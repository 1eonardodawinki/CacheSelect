"""Reference mathematics for propagating a Gated DeltaNet state edit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNBlockDeltaOperator:
    """Cached linear response of one unchanged GDN block."""

    transition: torch.Tensor
    output_responses: torch.Tensor


# Estimate recurrent checkpoint and block-operator storage without allocating it.
def estimate_gdn_delta_cache(
    *,
    block_size: int,
    value_heads: int,
    key_width: int,
    value_width: int,
    gdn_layers: int,
    resident_blocks: int,
    element_bytes: int,
) -> dict[str, int | float]:
    """Return per-block and model-wide byte costs for delta correction."""
    dimensions = (
        block_size,
        value_heads,
        key_width,
        value_width,
        gdn_layers,
        resident_blocks,
        element_bytes,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("all GDN cache dimensions must be positive")

    checkpoint_elements = value_heads * value_width * key_width
    transition_elements = value_heads * key_width * key_width
    response_elements = block_size * value_heads * key_width
    auxiliary_elements = transition_elements + response_elements
    existing_bytes_per_block_layer = checkpoint_elements * element_bytes
    auxiliary_bytes_per_block_layer = auxiliary_elements * element_bytes
    model_auxiliary_bytes = (
        auxiliary_bytes_per_block_layer * gdn_layers * resident_blocks
    )
    return {
        "checkpoint_elements_per_block_layer": checkpoint_elements,
        "transition_elements_per_block_layer": transition_elements,
        "response_elements_per_block_layer": response_elements,
        "existing_bytes_per_block_layer": existing_bytes_per_block_layer,
        "auxiliary_bytes_per_block_layer": auxiliary_bytes_per_block_layer,
        "combined_bytes_per_block_layer": (
            existing_bytes_per_block_layer + auxiliary_bytes_per_block_layer
        ),
        "model_auxiliary_bytes": model_auxiliary_bytes,
        "auxiliary_to_checkpoint_ratio": auxiliary_elements / checkpoint_elements,
    }


# Repeat each key/query head across the value heads that share it.
def _expand_grouped_heads(tensor: torch.Tensor, value_heads: int) -> torch.Tensor:
    key_heads = tensor.shape[1]
    if value_heads % key_heads != 0:
        raise ValueError("value-head count must be divisible by key-head count")
    return tensor.repeat_interleave(value_heads // key_heads, dim=1)


# Propagate one recurrent-state difference through one unchanged token.
def gdn_state_delta_step(
    state_delta: torch.Tensor,
    key: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact GDN delta recurrence for fixed token coefficients."""
    if state_delta.ndim != 3 or key.ndim != 2:
        raise ValueError("state_delta must be [HV,V,K] and key must be [HV,K]")
    if state_delta.shape[0] != key.shape[0] or state_delta.shape[2] != key.shape[1]:
        raise ValueError("state and expanded key dimensions do not agree")
    if log_decay.shape != state_delta.shape[:1] or beta.shape != state_delta.shape[:1]:
        raise ValueError("log_decay and beta must contain one scalar per value head")

    decayed_delta = state_delta * log_decay.exp()[:, None, None]
    state_read = torch.einsum("hvk,hk->hv", decayed_delta, key)
    # The value-vector term cancels because both executions see the same token.
    correction = beta[:, None, None] * state_read[:, :, None] * key[:, None, :]
    return decayed_delta - correction


# Propagate an edit delta and return its state/output effect at every later token.
def propagate_gdn_state_delta(
    initial_state_delta: torch.Tensor,
    keys: torch.Tensor,
    queries: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token state and output deltas under frozen coefficients."""
    if keys.ndim != 3 or queries.shape != keys.shape:
        raise ValueError("keys and queries must have matching [T,H,K] shapes")
    token_count, _, key_width = keys.shape
    value_heads, _, state_key_width = initial_state_delta.shape
    if state_key_width != key_width:
        raise ValueError("state, key, and query widths must agree")
    if log_decays.shape != (token_count, value_heads):
        raise ValueError("log_decays must have shape [T,HV]")
    if betas.shape != (token_count, value_heads):
        raise ValueError("betas must have shape [T,HV]")

    expanded_keys = _expand_grouped_heads(keys, value_heads)
    expanded_queries = _expand_grouped_heads(queries, value_heads)
    output_scale = scale if scale is not None else 1.0 / math.sqrt(key_width)
    state_delta = initial_state_delta
    state_deltas: list[torch.Tensor] = []
    output_deltas: list[torch.Tensor] = []
    for token_index in range(token_count):
        state_delta = gdn_state_delta_step(
            state_delta,
            expanded_keys[token_index],
            log_decays[token_index],
            betas[token_index],
        )
        state_deltas.append(state_delta)
        output_deltas.append(
            torch.einsum("hvk,hk->hv", state_delta, expanded_queries[token_index])
            * output_scale
        )
    if not state_deltas:
        return (
            initial_state_delta.new_empty((0, *initial_state_delta.shape)),
            initial_state_delta.new_empty(
                (0, value_heads, initial_state_delta.shape[1])
            ),
        )
    return torch.stack(state_deltas), torch.stack(output_deltas)


# Precompute one block's final-state and per-token response to an input delta.
def build_gdn_block_delta_operator(
    keys: torch.Tensor,
    queries: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
    *,
    scale: float | None = None,
) -> GDNBlockDeltaOperator:
    """Compose fixed token coefficients into a reusable block operator."""
    if keys.ndim != 3 or queries.shape != keys.shape:
        raise ValueError("keys and queries must have matching [T,H,K] shapes")
    token_count, key_heads, key_width = keys.shape
    if log_decays.ndim != 2 or betas.shape != log_decays.shape:
        raise ValueError("log_decays and betas must have matching [T,HV] shapes")
    if log_decays.shape[0] != token_count:
        raise ValueError("coefficient and token counts must agree")
    value_heads = log_decays.shape[1]
    if value_heads % key_heads != 0:
        raise ValueError("value-head count must be divisible by key-head count")

    expanded_keys = _expand_grouped_heads(keys, value_heads)
    expanded_queries = _expand_grouped_heads(queries, value_heads)
    output_scale = scale if scale is not None else 1.0 / math.sqrt(key_width)
    transition = (
        torch.eye(key_width, dtype=keys.dtype, device=keys.device)
        .expand(value_heads, key_width, key_width)
        .clone()
    )
    output_responses: list[torch.Tensor] = []
    for token_index in range(token_count):
        key = expanded_keys[token_index]
        transition_read = torch.einsum("hkl,hl->hk", transition, key)
        # Use the rank-one update directly instead of multiplying two KxK matrices.
        transition = log_decays[token_index].exp()[:, None, None] * (
            transition
            - betas[token_index, :, None, None]
            * transition_read[:, :, None]
            * key[:, None, :]
        )
        output_responses.append(
            torch.einsum("hkl,hl->hk", transition, expanded_queries[token_index])
            * output_scale
        )
    responses = (
        torch.stack(output_responses)
        if output_responses
        else keys.new_empty((0, value_heads, key_width))
    )
    return GDNBlockDeltaOperator(transition, responses)


# Apply a cached block operator to one edit-induced incoming state difference.
def apply_gdn_block_delta_operator(
    initial_state_delta: torch.Tensor,
    operator: GDNBlockDeltaOperator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the corrected final state and every corrected block output."""
    if initial_state_delta.ndim != 3:
        raise ValueError("initial_state_delta must have shape [HV,V,K]")
    if operator.transition.shape != (
        initial_state_delta.shape[0],
        initial_state_delta.shape[2],
        initial_state_delta.shape[2],
    ):
        raise ValueError("state and block-transition dimensions do not agree")
    final_state_delta = torch.einsum(
        "hvk,hkl->hvl", initial_state_delta, operator.transition
    )
    output_deltas = torch.einsum(
        "hvk,thk->thv", initial_state_delta, operator.output_responses
    )
    return final_state_delta, output_deltas
