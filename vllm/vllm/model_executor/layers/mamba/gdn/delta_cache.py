# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded sidecar storage for cached GDN block-delta operators."""

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNDeltaCacheEntry:
    """Tensor views and slot metadata for one resident block operator."""

    slot: int
    transition: torch.Tensor
    output_responses: torch.Tensor
    state_bias: torch.Tensor
    output_biases: torch.Tensor


@dataclass(frozen=True)
class GDNDeltaShadowComparison:
    """Numerical divergence between cached-operator and full block execution."""

    output_relative_l2: float
    output_max_absolute_error: float
    final_state_relative_l2: float
    final_state_max_absolute_error: float


@dataclass(frozen=True)
class GDNDeltaBlockShadowResult:
    """Shadow outcome for one target logical block in a prefill batch."""

    sequence_index: int
    target_block_index: int
    source_contextual_hash: bytes
    reason: str
    comparison: GDNDeltaShadowComparison | None = None


# Measure an error relative to the full-computation tensor's overall magnitude.
def _relative_l2_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    difference_norm = torch.linalg.vector_norm(candidate - reference)
    reference_norm = torch.linalg.vector_norm(reference)
    denominator = reference_norm.clamp_min(torch.finfo(torch.float32).eps)
    return float((difference_norm / denominator).item())


# Compare a cached affine result with full computation without changing either.
def compare_gdn_affine_shadow(
    initial_state: torch.Tensor,
    entry: GDNDeltaCacheEntry,
    reference_outputs: torch.Tensor,
    reference_final_state: torch.Tensor,
) -> GDNDeltaShadowComparison:
    """Apply one operator in shadow mode and summarize its numerical error."""
    approximate_state, approximate_outputs = apply_gdn_affine_operator(
        initial_state,
        entry,
    )
    reference_outputs = reference_outputs.float()
    reference_final_state = reference_final_state.float()
    if approximate_outputs.shape != reference_outputs.shape:
        raise ValueError("reference outputs do not match the cached block shape")
    if approximate_state.shape != reference_final_state.shape:
        raise ValueError("reference state does not match the cached state shape")

    output_difference = (approximate_outputs - reference_outputs).abs()
    state_difference = (approximate_state - reference_final_state).abs()
    return GDNDeltaShadowComparison(
        output_relative_l2=_relative_l2_error(
            approximate_outputs,
            reference_outputs,
        ),
        output_max_absolute_error=float(output_difference.max().item()),
        final_state_relative_l2=_relative_l2_error(
            approximate_state,
            reference_final_state,
        ),
        final_state_max_absolute_error=float(state_difference.max().item()),
    )


# Compare every fully scheduled candidate block against its normal GDN result.
def compare_completed_gdn_blocks_shadow(
    sidecar: "GDNDeltaOperatorSidecar",
    *,
    state_cache: torch.Tensor,
    full_outputs: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    query_start_locations: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    reuse_candidates: tuple[tuple[tuple[int, bytes], ...], ...],
    sequence_index_offset: int = 0,
) -> tuple[GDNDeltaBlockShadowResult, ...]:
    """Resolve checkpoint/token rows and compare candidates without mutation."""
    if state_cache.ndim != 4:
        raise ValueError("state_cache must have shape [slots,HV,V,K]")
    if full_outputs.ndim != 3:
        raise ValueError("full_outputs must have shape [tokens,HV,V]")
    if checkpoint_state_indices.ndim != 2:
        raise ValueError("checkpoint indices must have shape [sequences,blocks]")

    query_starts = query_start_locations.detach().cpu().tolist()
    computed_counts = num_computed_tokens.detach().cpu().tolist()
    sequence_count = len(query_starts) - 1
    if len(computed_counts) != sequence_count:
        raise ValueError("computed counts must contain one value per sequence")
    if len(reuse_candidates) != sequence_count:
        raise ValueError("reuse candidates must contain one tuple per sequence")
    if checkpoint_state_indices.shape[0] != sequence_count:
        raise ValueError("checkpoint rows must contain one row per sequence")
    if query_starts[0] != 0 or query_starts[-1] != full_outputs.shape[0]:
        raise ValueError("query starts must span the flattened output tensor")

    results = []
    block_size = sidecar.block_size
    for sequence_index, sequence_candidates in enumerate(reuse_candidates):
        query_start = int(query_starts[sequence_index])
        query_end = int(query_starts[sequence_index + 1])
        computed = int(computed_counts[sequence_index])
        for target_block_index, source_hash in sequence_candidates:
            entry = sidecar.lookup(source_hash)
            if entry is None:
                results.append(
                    GDNDeltaBlockShadowResult(
                        sequence_index + sequence_index_offset,
                        target_block_index,
                        source_hash,
                        "operator_not_resident",
                    )
                )
                continue
            if target_block_index <= 0 or target_block_index >= (
                checkpoint_state_indices.shape[1]
            ):
                results.append(
                    GDNDeltaBlockShadowResult(
                        sequence_index + sequence_index_offset,
                        target_block_index,
                        source_hash,
                        "invalid_target_block",
                    )
                )
                continue

            local_start = target_block_index * block_size - computed
            output_start = query_start + local_start
            output_end = output_start + block_size
            if local_start < 0 or output_end > query_end:
                results.append(
                    GDNDeltaBlockShadowResult(
                        sequence_index + sequence_index_offset,
                        target_block_index,
                        source_hash,
                        "block_not_fully_scheduled",
                    )
                )
                continue

            incoming_slot = int(
                checkpoint_state_indices[
                    sequence_index, target_block_index - 1
                ].item()
            )
            final_slot = int(
                checkpoint_state_indices[
                    sequence_index, target_block_index
                ].item()
            )
            if min(incoming_slot, final_slot) < 0 or max(
                incoming_slot, final_slot
            ) >= state_cache.shape[0]:
                results.append(
                    GDNDeltaBlockShadowResult(
                        sequence_index + sequence_index_offset,
                        target_block_index,
                        source_hash,
                        "invalid_checkpoint_slot",
                    )
                )
                continue

            comparison = compare_gdn_affine_shadow(
                state_cache[incoming_slot],
                entry,
                full_outputs[output_start:output_end],
                state_cache[final_slot],
            )
            results.append(
                GDNDeltaBlockShadowResult(
                    sequence_index + sequence_index_offset,
                    target_block_index,
                    source_hash,
                    "compared",
                    comparison,
                )
            )
    return tuple(results)


# Apply one cached affine block operator to a new incoming recurrent state.
def apply_gdn_affine_operator(
    initial_state: torch.Tensor,
    entry: GDNDeltaCacheEntry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the corrected final state and per-token recurrent outputs."""
    if initial_state.ndim != 3:
        raise ValueError("initial_state must have shape [HV,V,K]")
    value_heads, value_width, key_width = initial_state.shape
    if entry.transition.shape != (value_heads, key_width, key_width):
        raise ValueError("initial state and transition dimensions do not agree")
    if entry.state_bias.shape != initial_state.shape:
        raise ValueError("initial state and state bias dimensions do not agree")
    if entry.output_responses.shape[1:] != (value_heads, key_width):
        raise ValueError("initial state and output response dimensions do not agree")
    if entry.output_biases.shape != (
        entry.output_responses.shape[0],
        value_heads,
        value_width,
    ):
        raise ValueError("output response and bias dimensions do not agree")

    initial_state = initial_state.float()
    final_state = (
        torch.einsum("hvk,hkl->hvl", initial_state, entry.transition)
        + entry.state_bias
    )
    outputs = (
        torch.einsum("hvk,thk->thv", initial_state, entry.output_responses)
        + entry.output_biases
    )
    return final_state, outputs


# Repeat grouped key/query heads across the value heads that consume them.
def _expand_gdn_grouped_heads(
    tensor: torch.Tensor,
    value_heads: int,
) -> torch.Tensor:
    key_heads = tensor.shape[1]
    if value_heads % key_heads != 0:
        raise ValueError("value-head count must be divisible by key-head count")
    return tensor.repeat_interleave(value_heads // key_heads, dim=1)


# Compose one unchanged GDN block into final-state and output response tensors.
def build_gdn_delta_operator(
    keys: torch.Tensor,
    queries: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the linear response to an incoming recurrent-state difference."""
    if keys.ndim != 3 or queries.shape != keys.shape:
        raise ValueError("keys and queries must have matching [T,H,K] shapes")
    token_count, _, key_width = keys.shape
    if log_decays.ndim != 2 or betas.shape != log_decays.shape:
        raise ValueError("log_decays and betas must have matching [T,HV] shapes")
    if log_decays.shape[0] != token_count:
        raise ValueError("coefficient and token counts must agree")

    value_heads = log_decays.shape[1]
    expanded_keys = _expand_gdn_grouped_heads(keys, value_heads).float()
    expanded_queries = _expand_gdn_grouped_heads(queries, value_heads).float()
    log_decays = log_decays.float()
    betas = betas.float()
    transition = torch.eye(
        key_width,
        dtype=torch.float32,
        device=keys.device,
    ).expand(value_heads, key_width, key_width).clone()
    responses = torch.empty(
        token_count,
        value_heads,
        key_width,
        dtype=torch.float32,
        device=keys.device,
    )
    output_scale = key_width**-0.5
    for token_index in range(token_count):
        key = expanded_keys[token_index]
        transition_read = torch.einsum("hkl,hl->hk", transition, key)
        transition = log_decays[token_index].exp()[:, None, None] * (
            transition
            - betas[token_index, :, None, None]
            * transition_read[:, :, None]
            * key[:, None, :]
        )
        responses[token_index] = (
            torch.einsum(
                "hkl,hl->hk",
                transition,
                expanded_queries[token_index],
            )
            * output_scale
        )
    return transition, responses


# Compose the full affine state/output transform for one unchanged GDN block.
def build_gdn_affine_operator(
    keys: torch.Tensor,
    queries: torch.Tensor,
    values: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a new incoming state to exact block states and recurrent outputs."""
    transition, responses = build_gdn_delta_operator(
        keys,
        queries,
        log_decays,
        betas,
    )
    token_count, _, key_width = keys.shape
    value_heads = log_decays.shape[1]
    if values.ndim != 3 or values.shape[:2] != (token_count, value_heads):
        raise ValueError("values must have shape [T,HV,V]")

    expanded_keys = _expand_gdn_grouped_heads(keys, value_heads).float()
    expanded_queries = _expand_gdn_grouped_heads(queries, value_heads).float()
    values = values.float()
    log_decays = log_decays.float()
    betas = betas.float()
    state_bias = torch.zeros(
        value_heads,
        values.shape[2],
        key_width,
        dtype=torch.float32,
        device=keys.device,
    )
    output_biases = torch.empty(
        token_count,
        value_heads,
        values.shape[2],
        dtype=torch.float32,
        device=keys.device,
    )
    output_scale = key_width**-0.5
    for token_index in range(token_count):
        key = expanded_keys[token_index]
        state_bias = (
            state_bias * log_decays[token_index].exp()[:, None, None]
        )
        prior_read = torch.einsum("hvk,hk->hv", state_bias, key)
        innovation = betas[token_index, :, None] * (
            values[token_index] - prior_read
        )
        state_bias = state_bias + innovation[:, :, None] * key[:, None, :]
        output_biases[token_index] = (
            torch.einsum(
                "hvk,hk->hv",
                state_bias,
                expanded_queries[token_index],
            )
            * output_scale
        )
    return transition, responses, state_bias, output_biases


# Cache operators for every complete logical block covered by one prefill batch.
def store_completed_gdn_delta_operators(
    sidecar: "GDNDeltaOperatorSidecar",
    *,
    keys: torch.Tensor,
    queries: torch.Tensor,
    values: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
    query_start_locations: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    contextual_block_hashes: tuple[tuple[bytes, ...], ...],
) -> tuple[tuple[int, int], ...]:
    """Store complete block operators and return (sequence, block) pairs."""
    if keys.ndim != 3 or queries.shape != keys.shape:
        raise ValueError("keys and queries must have matching [T,H,K] shapes")
    if log_decays.ndim != 2 or betas.shape != log_decays.shape:
        raise ValueError("log_decays and betas must have matching [T,HV] shapes")
    if keys.shape[0] != log_decays.shape[0]:
        raise ValueError("token coefficient tensors must have the same length")
    if values.ndim != 3 or values.shape[:2] != log_decays.shape:
        raise ValueError("values must have shape [T,HV,V]")

    query_starts = query_start_locations.detach().cpu().tolist()
    computed_counts = num_computed_tokens.detach().cpu().tolist()
    sequence_count = len(query_starts) - 1
    if len(computed_counts) != sequence_count:
        raise ValueError("num_computed_tokens must contain one value per sequence")
    if len(contextual_block_hashes) != sequence_count:
        raise ValueError("contextual hashes must contain one tuple per sequence")
    if query_starts[0] != 0 or query_starts[-1] != keys.shape[0]:
        raise ValueError("query starts must span the flattened coefficient tensors")

    block_size = sidecar.block_size
    stored_blocks: list[tuple[int, int]] = []
    for sequence_index in range(sequence_count):
        query_start = int(query_starts[sequence_index])
        query_end = int(query_starts[sequence_index + 1])
        computed = int(computed_counts[sequence_index])
        scheduled = query_end - query_start
        first_complete_block = (computed + block_size - 1) // block_size
        final_complete_block = (computed + scheduled) // block_size
        hashes = contextual_block_hashes[sequence_index]
        for block_index in range(first_complete_block, final_complete_block):
            if block_index >= len(hashes):
                # Generated tokens may complete blocks not present in prompt hashes.
                continue
            local_start = block_index * block_size - computed
            local_end = local_start + block_size
            if local_start < 0 or local_end > scheduled:
                continue
            coefficient_start = query_start + local_start
            coefficient_end = query_start + local_end
            transition, responses, state_bias, output_biases = (
                build_gdn_affine_operator(
                    keys[coefficient_start:coefficient_end],
                    queries[coefficient_start:coefficient_end],
                    values[coefficient_start:coefficient_end],
                    log_decays[coefficient_start:coefficient_end],
                    betas[coefficient_start:coefficient_end],
                )
            )
            sidecar.store(
                hashes[block_index],
                transition,
                responses,
                state_bias,
                output_biases,
            )
            stored_blocks.append((sequence_index, block_index))
    return tuple(stored_blocks)


class GDNDeltaOperatorSidecar:
    """Fixed-capacity LRU storage kept outside ordinary Mamba cache pages."""

    # Allocate the bounded tensor storage and initialize its CPU lookup table.
    def __init__(
        self,
        *,
        capacity: int,
        block_size: int,
        value_heads: int,
        key_width: int,
        value_width: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        dimensions = (
            capacity,
            block_size,
            value_heads,
            key_width,
            value_width,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("all GDN sidecar dimensions must be positive")
        if not dtype.is_floating_point:
            raise ValueError("GDN sidecar tensors require a floating-point dtype")

        self.capacity = capacity
        self.block_size = block_size
        self.value_heads = value_heads
        self.key_width = key_width
        self.value_width = value_width
        self.transitions = torch.empty(
            capacity,
            value_heads,
            key_width,
            key_width,
            dtype=dtype,
            device=device,
        )
        self.output_responses = torch.empty(
            capacity,
            block_size,
            value_heads,
            key_width,
            dtype=dtype,
            device=device,
        )
        self.state_biases = torch.empty(
            capacity,
            value_heads,
            value_width,
            key_width,
            dtype=dtype,
            device=device,
        )
        self.output_biases = torch.empty(
            capacity,
            block_size,
            value_heads,
            value_width,
            dtype=dtype,
            device=device,
        )
        self._key_to_slot: OrderedDict[bytes, int] = OrderedDict()
        self._free_slots = list(range(capacity))

    # Report how many block operators currently occupy the bounded cache.
    def __len__(self) -> int:
        return len(self._key_to_slot)

    # Return resident keys in least-to-most-recently-used order for auditing.
    def resident_keys(self) -> tuple[bytes, ...]:
        return tuple(self._key_to_slot)

    # Check complete residency without changing the LRU order.
    def contains_many(self, block_hashes: tuple[bytes, ...]) -> bool:
        return all(block_hash in self._key_to_slot for block_hash in block_hashes)

    # Insert or refresh one operator, evicting the least-recent key if needed.
    def store(
        self,
        block_hash: bytes,
        transition: torch.Tensor,
        output_responses: torch.Tensor,
        state_bias: torch.Tensor,
        output_biases: torch.Tensor,
    ) -> int:
        if not block_hash:
            raise ValueError("block_hash must be non-empty")
        expected_transition = (self.value_heads, self.key_width, self.key_width)
        expected_responses = (self.block_size, self.value_heads, self.key_width)
        expected_state_bias = (self.value_heads, self.value_width, self.key_width)
        expected_output_biases = (
            self.block_size,
            self.value_heads,
            self.value_width,
        )
        if transition.shape != expected_transition:
            raise ValueError(
                f"transition must have shape {expected_transition}, "
                f"received {tuple(transition.shape)}"
            )
        if output_responses.shape != expected_responses:
            raise ValueError(
                f"output_responses must have shape {expected_responses}, "
                f"received {tuple(output_responses.shape)}"
            )
        if state_bias.shape != expected_state_bias:
            raise ValueError(
                f"state_bias must have shape {expected_state_bias}, "
                f"received {tuple(state_bias.shape)}"
            )
        if output_biases.shape != expected_output_biases:
            raise ValueError(
                f"output_biases must have shape {expected_output_biases}, "
                f"received {tuple(output_biases.shape)}"
            )

        slot = self._key_to_slot.pop(block_hash, None)
        if slot is None:
            if self._free_slots:
                slot = self._free_slots.pop(0)
            else:
                _, slot = self._key_to_slot.popitem(last=False)
        self.transitions[slot].copy_(transition)
        self.output_responses[slot].copy_(output_responses)
        self.state_biases[slot].copy_(state_bias)
        self.output_biases[slot].copy_(output_biases)
        self._key_to_slot[block_hash] = slot
        return slot

    # Resolve one block hash and refresh its LRU position without copying data.
    def lookup(self, block_hash: bytes) -> GDNDeltaCacheEntry | None:
        slot = self._key_to_slot.pop(block_hash, None)
        if slot is None:
            return None
        self._key_to_slot[block_hash] = slot
        return GDNDeltaCacheEntry(
            slot=slot,
            transition=self.transitions[slot],
            output_responses=self.output_responses[slot],
            state_bias=self.state_biases[slot],
            output_biases=self.output_biases[slot],
        )

    # Resolve a complete operator set atomically or leave the LRU untouched.
    def lookup_many(
        self,
        block_hashes: tuple[bytes, ...],
    ) -> tuple[GDNDeltaCacheEntry, ...] | None:
        slots = tuple(self._key_to_slot.get(block_hash) for block_hash in block_hashes)
        if any(slot is None for slot in slots):
            return None
        entries = []
        for block_hash, optional_slot in zip(block_hashes, slots):
            assert optional_slot is not None
            self._key_to_slot.move_to_end(block_hash)
            entries.append(
                GDNDeltaCacheEntry(
                    slot=optional_slot,
                    transition=self.transitions[optional_slot],
                    output_responses=self.output_responses[optional_slot],
                    state_bias=self.state_biases[optional_slot],
                    output_biases=self.output_biases[optional_slot],
                )
            )
        return tuple(entries)

    # Remove all logical entries while retaining the allocated GPU buffers.
    def clear(self) -> None:
        self._key_to_slot.clear()
        self._free_slots = list(range(self.capacity))
