"""Reference mathematics for propagating a Gated DeltaNet state edit."""

from __future__ import annotations

import math

import torch


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
