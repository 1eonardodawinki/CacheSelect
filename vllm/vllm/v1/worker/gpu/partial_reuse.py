# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve CacheSelect mappings and build worker copy and repair plans."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from vllm.config.cache import CacheSelectRepairSelector
    from vllm.v1.core.partial_reuse import PartialReusePlan
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.utils import AttentionGroup

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.partial_reuse import CacheSelectRepairMetrics
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata


@dataclass(frozen=True)
class ResolvedPartialReuseCandidate:
    source_block_index: int
    target_block_index: int
    source_block_id: int
    target_block_id: int
    source_resident: bool
    requires_repair: bool
    block_displacement: int = 0
    nearest_changed_block_distance: int | None = None


@dataclass(frozen=True)
class PartialReuseCopyInstruction:
    source_block_id: int
    target_block_id: int
    target_block_index: int
    requires_repair: bool


@dataclass(frozen=True)
class PartialReuseRepairInstruction:
    source_block_id: int
    target_block_id: int
    target_block_index: int
    target_token_indices: tuple[int, ...]


@dataclass(frozen=True)
class PartialReuseBatchDecision:
    eligible: bool
    reason: str
    request_id: str | None = None
    reused_batch_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class PartialReuseComputeSpan:
    start_row: int
    end_row: int


@dataclass(frozen=True)
class PartialReuseCompactedBatch:
    compute_rows: tuple[int, ...]
    compute_spans: tuple[PartialReuseComputeSpan, ...]
    span_inputs: tuple[PartialReuseSpanInputs, ...]
    span_attention_inputs: tuple[PartialReuseSpanAttentionInputs, ...]
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mappings: torch.Tensor
    query_start_locations: tuple[int, ...]


@dataclass(frozen=True)
class PartialReuseSpanInputs:
    span: PartialReuseComputeSpan
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mappings: torch.Tensor
    query_start_locations: tuple[int, int]
    sequence_length: int


@dataclass(frozen=True)
class PartialReuseSpanAttentionInputs:
    span: PartialReuseComputeSpan
    num_tokens: int
    query_start_loc_cpu: torch.Tensor
    query_start_loc_gpu: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    positions: torch.Tensor


class PartialReuseRepairSelector(Protocol):
    # Select target tokens that must be recomputed after block reuse.
    def select(
        self,
        candidates: Sequence[ResolvedPartialReuseCandidate],
        block_size: int,
    ) -> tuple[PartialReuseRepairInstruction, ...]: ...


# Convert logical target positions into physical V2 runner block IDs.
def resolve_target_block_ids(
    plan: PartialReusePlan,
    target_block_ids: Sequence[Sequence[int]],
) -> tuple[ResolvedPartialReuseCandidate, ...]:
    """Resolve logical target positions to physical block IDs."""
    # The locator currently indexes only the first KV-cache group, so reject
    # multi-group layouts instead of producing an unsafe cross-group mapping.
    if len(target_block_ids) != 1:
        raise ValueError("partial reuse currently requires one KV cache group")

    target_group = target_block_ids[0]
    resolved = []
    for candidate in plan.candidates:
        target_index = candidate.target_block_index
        if target_index < 0 or target_index >= len(target_group):
            raise ValueError(
                f"target block index {target_index} is outside the request block table"
            )
        resolved.append(
            ResolvedPartialReuseCandidate(
                source_block_index=candidate.source_block_index,
                target_block_index=target_index,
                source_block_id=candidate.source_block_id,
                target_block_id=target_group[target_index],
                source_resident=candidate.source_resident,
                requires_repair=candidate.requires_repair,
                block_displacement=candidate.block_displacement,
                nearest_changed_block_distance=(
                    candidate.nearest_changed_block_distance
                ),
            )
        )
    return tuple(resolved)


# Convert approved mappings into source-to-destination copy instructions.
def build_partial_reuse_copy_instructions(
    candidates: Sequence[ResolvedPartialReuseCandidate],
) -> tuple[PartialReuseCopyInstruction, ...]:
    """Build a copy plan without applying it to GPU memory."""
    return tuple(
        PartialReuseCopyInstruction(
            source_block_id=candidate.source_block_id,
            target_block_id=candidate.target_block_id,
            target_block_index=candidate.target_block_index,
            requires_repair=candidate.requires_repair,
        )
        for candidate in candidates
    )


# Convert CacheSelect instructions into vLLM's physical block-copy format.
def build_kv_cache_block_copies(
    instructions: Sequence[PartialReuseCopyInstruction],
) -> tuple[KVCacheBlockCopy, ...]:
    return tuple(
        KVCacheBlockCopy(
            src_block_id=instruction.source_block_id,
            dst_block_id=instruction.target_block_id,
        )
        for instruction in instructions
    )


# Record block copies that were actually submitted by the GPU worker.
def record_copy_execution(
    metrics: CacheSelectRepairMetrics,
    copied_blocks: int,
    block_size: int,
) -> CacheSelectRepairMetrics:
    if copied_blocks < 0:
        raise ValueError("copied_blocks must be non-negative")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    copied_tokens = copied_blocks * block_size
    if copied_tokens > metrics.candidate_tokens:
        raise ValueError("copied tokens exceed candidate tokens")
    return replace(
        metrics,
        copied_blocks=copied_blocks,
        copied_tokens=copied_tokens,
    )


# Record one advisory batch decision in request-level worker metrics.
def record_batch_execution_decision(
    metrics: CacheSelectRepairMetrics,
    *,
    eligible: bool,
    reason: str,
    reused_batch_rows: int,
    compute_batch_rows: int,
) -> CacheSelectRepairMetrics:
    if reused_batch_rows < 0:
        raise ValueError("reused_batch_rows must be non-negative")
    if compute_batch_rows < 0:
        raise ValueError("compute_batch_rows must be non-negative")
    if eligible and reused_batch_rows == 0:
        raise ValueError("eligible execution requires reusable batch rows")
    if eligible and compute_batch_rows == 0:
        raise ValueError("eligible execution requires compute batch rows")
    return replace(
        metrics,
        execution_eligible=eligible,
        execution_reason=reason,
        reused_batch_rows=reused_batch_rows,
        compute_batch_rows=compute_batch_rows,
    )


# Record successful construction of the advisory compact model batch.
def record_compacted_batch_construction(
    metrics: CacheSelectRepairMetrics,
    compacted_rows: int,
    compute_span_count: int,
) -> CacheSelectRepairMetrics:
    if compacted_rows < 1:
        raise ValueError("compacted batch must contain at least one row")
    if compute_span_count < 1:
        raise ValueError("compacted batch must contain at least one compute span")
    if compacted_rows != metrics.compute_batch_rows:
        raise ValueError("compacted rows must match selected compute rows")
    return replace(
        metrics,
        compute_span_count=compute_span_count,
        compacted_batch_built=True,
    )


# Record successful backend metadata construction for every compute span.
def record_span_attention_metadata_construction(
    metrics: CacheSelectRepairMetrics,
    metadata_count: int,
) -> CacheSelectRepairMetrics:
    if metadata_count < 1:
        raise ValueError("partial reuse requires at least one metadata object")
    if metadata_count != metrics.compute_span_count:
        raise ValueError("span metadata count must match compute span count")
    return replace(
        metrics,
        span_metadata_built=True,
        span_metadata_count=metadata_count,
    )


# Select copied prompt-token positions that the repair policy leaves reusable.
def build_reused_token_indices(
    copy_instructions: Sequence[PartialReuseCopyInstruction],
    repair_instructions: Sequence[PartialReuseRepairInstruction],
    block_size: int,
) -> tuple[int, ...]:
    if block_size < 1:
        raise ValueError("block_size must be positive")

    target_block_indices = [
        instruction.target_block_index for instruction in copy_instructions
    ]
    if len(target_block_indices) != len(set(target_block_indices)):
        raise ValueError("copy plan contains duplicate target blocks")

    candidate_tokens = {
        token_index
        for block_index in target_block_indices
        for token_index in range(
            block_index * block_size,
            (block_index + 1) * block_size,
        )
    }
    repair_tokens: set[int] = set()
    for instruction in repair_instructions:
        if instruction.target_block_index not in target_block_indices:
            raise ValueError("repair plan references a non-candidate block")
        block_start = instruction.target_block_index * block_size
        block_end = block_start + block_size
        if any(
            token_index < block_start or token_index >= block_end
            for token_index in instruction.target_token_indices
        ):
            raise ValueError("repair token lies outside its target block")
        repair_tokens.update(instruction.target_token_indices)

    return tuple(sorted(candidate_tokens - repair_tokens))


# Translate reusable prompt positions into rows of one flattened input batch.
def map_reused_tokens_to_batch_rows(
    req_ids: Sequence[str],
    query_start_locations: Sequence[int],
    num_computed_tokens: Sequence[int],
    num_scheduled_tokens: Sequence[int],
    reused_token_indices: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    num_reqs = len(req_ids)
    if len(query_start_locations) != num_reqs + 1:
        raise ValueError(
            "query_start_locations must contain one boundary per request"
        )
    if len(num_computed_tokens) != num_reqs:
        raise ValueError("num_computed_tokens must match req_ids")
    if len(num_scheduled_tokens) != num_reqs:
        raise ValueError("num_scheduled_tokens must match req_ids")

    rows_by_request: dict[str, tuple[int, ...]] = {}
    for index, req_id in enumerate(req_ids):
        batch_start = int(query_start_locations[index])
        batch_end = int(query_start_locations[index + 1])
        scheduled_tokens = int(num_scheduled_tokens[index])
        if batch_end - batch_start != scheduled_tokens:
            raise ValueError("query boundaries disagree with scheduled token counts")

        prompt_start = int(num_computed_tokens[index])
        prompt_end = prompt_start + scheduled_tokens
        prompt_indices = tuple(reused_token_indices.get(req_id, ()))
        if prompt_indices != tuple(sorted(set(prompt_indices))):
            raise ValueError("reused token indices must be sorted and unique")
        if any(prompt_index < 0 for prompt_index in prompt_indices):
            raise ValueError("reused token indices must be non-negative")

        rows = tuple(
            batch_start + prompt_index - prompt_start
            for prompt_index in prompt_indices
            if prompt_start <= prompt_index < prompt_end
        )
        if rows:
            rows_by_request[req_id] = rows
    return rows_by_request


# Decide whether one batch fits the deliberately narrow first execution scope.
def assess_partial_reuse_batch(
    *,
    execution_enabled: bool,
    req_ids: Sequence[str],
    is_prefilling: Sequence[bool],
    reused_batch_rows: Mapping[str, Sequence[int]],
    single_gpu: bool,
    supported_kv_layout: bool,
    speculative_decoding: bool,
    multimodal_model: bool,
    encoder_decoder_model: bool,
    pooling_model: bool,
) -> PartialReuseBatchDecision:
    if len(is_prefilling) != len(req_ids):
        raise ValueError("is_prefilling must match req_ids")
    if not reused_batch_rows:
        reason = "execution_disabled" if not execution_enabled else "no_reusable_rows"
        return PartialReuseBatchDecision(False, reason)
    if len(reused_batch_rows) != 1:
        return PartialReuseBatchDecision(False, "multiple_reuse_requests")

    request_id, rows = next(iter(reused_batch_rows.items()))
    decision_details = {
        "request_id": request_id,
        "reused_batch_rows": tuple(rows),
    }
    if request_id not in req_ids:
        raise ValueError("reused rows reference a request outside the batch")
    if not execution_enabled:
        return PartialReuseBatchDecision(
            False,
            "execution_disabled",
            **decision_details,
        )
    if len(req_ids) != 1:
        return PartialReuseBatchDecision(
            False,
            "batched_requests_unsupported",
            **decision_details,
        )
    if not rows:
        raise ValueError("reused batch row selection cannot be empty")
    if not bool(is_prefilling[0]):
        return PartialReuseBatchDecision(
            False,
            "non_prefill_request",
            **decision_details,
        )
    if not single_gpu:
        return PartialReuseBatchDecision(
            False,
            "parallelism_unsupported",
            **decision_details,
        )
    if not supported_kv_layout:
        return PartialReuseBatchDecision(
            False,
            "kv_layout_unsupported",
            **decision_details,
        )
    if speculative_decoding:
        return PartialReuseBatchDecision(
            False,
            "speculative_decoding_unsupported",
            **decision_details,
        )
    if multimodal_model:
        return PartialReuseBatchDecision(
            False,
            "multimodal_unsupported",
            **decision_details,
        )
    if encoder_decoder_model:
        return PartialReuseBatchDecision(
            False,
            "encoder_decoder_unsupported",
            **decision_details,
        )
    if pooling_model:
        return PartialReuseBatchDecision(
            False,
            "pooling_unsupported",
            **decision_details,
        )
    return PartialReuseBatchDecision(
        True,
        "eligible",
        **decision_details,
    )


# Select flattened batch rows that still require model computation.
def build_partial_reuse_compute_rows(
    decision: PartialReuseBatchDecision,
    num_tokens: int,
) -> tuple[int, ...]:
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")

    all_rows = tuple(range(num_tokens))
    if not decision.eligible:
        return all_rows

    reused_rows = decision.reused_batch_rows
    if reused_rows != tuple(sorted(set(reused_rows))):
        raise ValueError("reused batch rows must be sorted and unique")
    if any(row < 0 or row >= num_tokens for row in reused_rows):
        raise ValueError("reused batch row is outside the input batch")

    reused_row_set = set(reused_rows)
    compute_rows = tuple(row for row in all_rows if row not in reused_row_set)
    if not compute_rows:
        raise ValueError("partial reuse must retain at least one compute row")
    return compute_rows


# Group retained compute rows into contiguous half-open execution spans.
def build_partial_reuse_compute_spans(
    compute_rows: Sequence[int],
    num_rows: int,
) -> tuple[PartialReuseComputeSpan, ...]:
    selected_rows = _validate_compaction_rows(compute_rows, num_rows)
    spans: list[PartialReuseComputeSpan] = []
    span_start = selected_rows[0]
    previous_row = span_start
    for row in selected_rows[1:]:
        if row != previous_row + 1:
            spans.append(PartialReuseComputeSpan(span_start, previous_row + 1))
            span_start = row
        previous_row = row
    spans.append(PartialReuseComputeSpan(span_start, previous_row + 1))
    return tuple(spans)


# Validate selected rows against one unpadded flattened input.
def _validate_compaction_rows(
    compute_rows: Sequence[int],
    num_rows: int,
) -> tuple[int, ...]:
    selected_rows = tuple(compute_rows)
    if not selected_rows:
        raise ValueError("partial reuse must retain at least one compute row")
    if selected_rows != tuple(sorted(set(selected_rows))):
        raise ValueError("compute rows must be sorted and unique")
    if any(row < 0 or row >= num_rows for row in selected_rows):
        raise ValueError("compute row is outside the compacted input")
    return selected_rows


# Build validated row indices on the tensor being compacted.
def _build_compaction_row_indices(
    compute_rows: Sequence[int],
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    selected_rows = _validate_compaction_rows(compute_rows, num_rows)
    return torch.tensor(selected_rows, dtype=torch.long, device=device)


# Select token IDs and absolute positions for the rows that still need compute.
def compact_partial_reuse_model_inputs(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    compute_rows: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 1 or positions.ndim != 1:
        raise ValueError("partial reuse model inputs must be one-dimensional")
    if input_ids.numel() != positions.numel():
        raise ValueError("input IDs and positions must contain the same rows")
    if input_ids.device != positions.device:
        raise ValueError("input IDs and positions must use the same device")

    # index_select preserves the supplied absolute positions after row packing.
    row_indices = _build_compaction_row_indices(
        compute_rows,
        input_ids.numel(),
        input_ids.device,
    )
    return (
        input_ids.index_select(0, row_indices),
        positions.index_select(0, row_indices),
    )


# Select KV-cache write destinations for the same compacted token rows.
def compact_partial_reuse_slot_mappings(
    slot_mappings: torch.Tensor,
    compute_rows: Sequence[int],
) -> torch.Tensor:
    if slot_mappings.ndim != 2:
        raise ValueError("slot mappings must have cache-group and token dimensions")
    row_indices = _build_compaction_row_indices(
        compute_rows,
        slot_mappings.shape[1],
        slot_mappings.device,
    )
    return slot_mappings.index_select(1, row_indices)


# Rebuild request boundaries after selected rows are packed together.
def compact_partial_reuse_query_start_locations(
    query_start_locations: Sequence[int],
    compute_rows: Sequence[int],
) -> tuple[int, ...]:
    boundaries = tuple(query_start_locations)
    if len(boundaries) < 2:
        raise ValueError("query boundaries require at least one request")
    if boundaries[0] != 0:
        raise ValueError("query boundaries must start at zero")
    if boundaries != tuple(sorted(boundaries)):
        raise ValueError("query boundaries must be non-decreasing")

    selected_rows = _validate_compaction_rows(compute_rows, boundaries[-1])
    # Each new boundary is the count of retained rows before the old boundary.
    return tuple(bisect_left(selected_rows, boundary) for boundary in boundaries)


# Build one aligned advisory batch from all compacted model inputs.
def build_partial_reuse_compacted_batch(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    slot_mappings: torch.Tensor,
    block_tables: Sequence[torch.Tensor],
    query_start_locations: Sequence[int],
    compute_rows: Sequence[int],
    initial_computed_tokens: int,
) -> PartialReuseCompactedBatch:
    boundaries = tuple(query_start_locations)
    if len(boundaries) < 2:
        raise ValueError("query boundaries require at least one request")
    if boundaries[-1] != input_ids.numel():
        raise ValueError("query boundaries must cover every model input row")
    if slot_mappings.ndim != 2:
        raise ValueError("slot mappings must have cache-group and token dimensions")
    if slot_mappings.shape[1] != input_ids.numel():
        raise ValueError("slot mappings must cover every model input row")

    selected_rows = _validate_compaction_rows(compute_rows, input_ids.numel())
    compacted_ids, compacted_positions = compact_partial_reuse_model_inputs(
        input_ids,
        positions,
        selected_rows,
    )
    compute_spans = build_partial_reuse_compute_spans(
        selected_rows,
        input_ids.numel(),
    )
    span_inputs = build_partial_reuse_span_input_sequence(
        input_ids,
        positions,
        slot_mappings,
        compute_spans,
        initial_computed_tokens,
    )
    return PartialReuseCompactedBatch(
        compute_rows=selected_rows,
        compute_spans=compute_spans,
        span_inputs=span_inputs,
        span_attention_inputs=build_partial_reuse_span_attention_input_sequence(
            span_inputs,
            block_tables,
        ),
        input_ids=compacted_ids,
        positions=compacted_positions,
        slot_mappings=compact_partial_reuse_slot_mappings(
            slot_mappings,
            selected_rows,
        ),
        query_start_locations=compact_partial_reuse_query_start_locations(
            boundaries,
            selected_rows,
        ),
    )


# Build the model inputs and resulting context length for one compute span.
def build_partial_reuse_span_inputs(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    slot_mappings: torch.Tensor,
    span: PartialReuseComputeSpan,
    initial_computed_tokens: int,
) -> PartialReuseSpanInputs:
    if initial_computed_tokens < 0:
        raise ValueError("initial_computed_tokens must be non-negative")
    if span.start_row < 0 or span.start_row >= span.end_row:
        raise ValueError("compute span must be a non-empty forward range")
    if span.end_row > input_ids.numel():
        raise ValueError("compute span extends beyond the model input")

    span_rows = tuple(range(span.start_row, span.end_row))
    span_input_ids, span_positions = compact_partial_reuse_model_inputs(
        input_ids,
        positions,
        span_rows,
    )
    span_row_count = span.end_row - span.start_row
    return PartialReuseSpanInputs(
        span=span,
        input_ids=span_input_ids,
        positions=span_positions,
        slot_mappings=compact_partial_reuse_slot_mappings(
            slot_mappings,
            span_rows,
        ),
        query_start_locations=(0, span_row_count),
        # Earlier scheduled rows are either freshly computed or reused KV.
        sequence_length=initial_computed_tokens + span.end_row,
    )


# Build all compute-span inputs in the order required by causal attention.
def build_partial_reuse_span_input_sequence(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    slot_mappings: torch.Tensor,
    spans: Sequence[PartialReuseComputeSpan],
    initial_computed_tokens: int,
) -> tuple[PartialReuseSpanInputs, ...]:
    ordered_spans = tuple(spans)
    if not ordered_spans:
        raise ValueError("partial reuse requires at least one compute span")
    previous_end = 0
    for span in ordered_spans:
        if span.start_row < previous_end:
            raise ValueError("compute spans must be ordered and non-overlapping")
        previous_end = span.end_row
    return tuple(
        build_partial_reuse_span_inputs(
            input_ids,
            positions,
            slot_mappings,
            span,
            initial_computed_tokens,
        )
        for span in ordered_spans
    )


# Translate one compute span into the plain decoder model's call signature.
def build_partial_reuse_span_model_inputs(
    span_inputs: PartialReuseSpanInputs,
) -> dict[str, Any]:
    """Build advisory model keyword arguments for one isolated span."""
    if span_inputs.input_ids.ndim != 1 or span_inputs.positions.ndim != 1:
        raise ValueError("span model inputs must be one-dimensional")
    if span_inputs.input_ids.numel() != span_inputs.positions.numel():
        raise ValueError("span input IDs and positions must contain the same rows")
    if span_inputs.input_ids.numel() == 0:
        raise ValueError("span model inputs cannot be empty")
    expected_rows = span_inputs.span.end_row - span_inputs.span.start_row
    if span_inputs.input_ids.numel() != expected_rows:
        raise ValueError("span model inputs must cover the complete compute span")

    # Attention metadata is installed separately through set_forward_context.
    return {
        "input_ids": span_inputs.input_ids,
        "positions": span_inputs.positions,
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }


# Build the standard attention-builder inputs for one isolated compute span.
def build_partial_reuse_span_attention_inputs(
    span_inputs: PartialReuseSpanInputs,
    block_tables: Sequence[torch.Tensor],
) -> PartialReuseSpanAttentionInputs:
    tables = tuple(block_tables)
    if not tables:
        raise ValueError("partial reuse requires at least one block table")
    if any(table.ndim != 2 or table.shape[0] < 1 for table in tables):
        raise ValueError("block tables must contain the isolated request")
    if span_inputs.slot_mappings.shape[0] != len(tables):
        raise ValueError("slot mappings must match the KV cache groups")

    num_tokens = span_inputs.input_ids.numel()
    if span_inputs.query_start_locations != (0, num_tokens):
        raise ValueError("span query boundaries must cover every span token")
    if span_inputs.sequence_length < num_tokens:
        raise ValueError("span sequence length cannot be shorter than its query")

    query_boundaries = span_inputs.query_start_locations
    return PartialReuseSpanAttentionInputs(
        span=span_inputs.span,
        num_tokens=num_tokens,
        query_start_loc_cpu=torch.tensor(query_boundaries, dtype=torch.int32),
        query_start_loc_gpu=torch.tensor(
            query_boundaries,
            dtype=torch.int32,
            device=span_inputs.positions.device,
        ),
        seq_lens=torch.tensor(
            [span_inputs.sequence_length],
            dtype=torch.int32,
            device=span_inputs.positions.device,
        ),
        max_query_len=num_tokens,
        max_seq_len=span_inputs.sequence_length,
        block_tables=tuple(table[:1] for table in tables),
        slot_mappings=span_inputs.slot_mappings,
        positions=span_inputs.positions,
    )


# Build attention inputs for every compute span in causal execution order.
def build_partial_reuse_span_attention_input_sequence(
    span_inputs: Sequence[PartialReuseSpanInputs],
    block_tables: Sequence[torch.Tensor],
) -> tuple[PartialReuseSpanAttentionInputs, ...]:
    ordered_inputs = tuple(span_inputs)
    if not ordered_inputs:
        raise ValueError("partial reuse requires at least one span input")
    return tuple(
        build_partial_reuse_span_attention_inputs(item, block_tables)
        for item in ordered_inputs
    )


# Ask vLLM's backend builders for advisory metadata for every compute span.
def build_partial_reuse_span_attention_metadata(
    span_attention_inputs: Sequence[PartialReuseSpanAttentionInputs],
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
) -> tuple[dict[str, Any], ...]:
    ordered_inputs = tuple(span_attention_inputs)
    if not ordered_inputs:
        raise ValueError("partial reuse requires span attention inputs")
    return tuple(
        build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=1,
            num_tokens=item.num_tokens,
            query_start_loc_gpu=item.query_start_loc_gpu,
            query_start_loc_cpu=item.query_start_loc_cpu,
            max_query_len=item.max_query_len,
            seq_lens=item.seq_lens,
            max_seq_len=item.max_seq_len,
            block_tables=item.block_tables,
            slot_mappings=item.slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=torch.tensor(
                [item.max_seq_len],
                dtype=torch.int32,
            ),
            positions=item.positions,
            is_prefilling=torch.tensor([True]),
        )
        for item in ordered_inputs
    )


# Build a safe fallback that repairs every token in each affected target block.
def build_full_block_repair_instructions(
    candidates: Sequence[ResolvedPartialReuseCandidate],
    block_size: int,
) -> tuple[PartialReuseRepairInstruction, ...]:
    """Represent conservative repair without scheduling recomputation."""
    if block_size < 1:
        raise ValueError("block_size must be positive")

    instructions = []
    for candidate in candidates:
        if not candidate.requires_repair:
            continue
        token_start = candidate.target_block_index * block_size
        instructions.append(
            PartialReuseRepairInstruction(
                source_block_id=candidate.source_block_id,
                target_block_id=candidate.target_block_id,
                target_block_index=candidate.target_block_index,
                target_token_indices=tuple(
                    range(token_start, token_start + block_size)
                ),
            )
        )
    return tuple(instructions)


class FullBlockRepairSelector:
    # Select every token in blocks whose reused KV may depend on changed context.
    def select(
        self,
        candidates: Sequence[ResolvedPartialReuseCandidate],
        block_size: int,
    ) -> tuple[PartialReuseRepairInstruction, ...]:
        return build_full_block_repair_instructions(candidates, block_size)


class EditProximityRepairSelector:
    """Repair whole blocks near edits and experimentally reuse farther blocks."""

    # Configure the largest edit distance that still triggers block repair.
    def __init__(self, max_block_distance: int) -> None:
        if max_block_distance < 0:
            raise ValueError("max_block_distance must be non-negative")
        self.max_block_distance = max_block_distance

    # Select nearby candidates, while treating missing geometry conservatively.
    def select(
        self,
        candidates: Sequence[ResolvedPartialReuseCandidate],
        block_size: int,
    ) -> tuple[PartialReuseRepairInstruction, ...]:
        selected_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.requires_repair
            and (
                candidate.nearest_changed_block_distance is None
                or candidate.nearest_changed_block_distance
                <= self.max_block_distance
            )
        )
        return build_full_block_repair_instructions(
            selected_candidates, block_size
        )


# Construct the configured repair selector while keeping policy wiring centralized.
def create_repair_selector(
    selector_name: CacheSelectRepairSelector,
    edit_radius: int,
) -> PartialReuseRepairSelector:
    if selector_name == "full_block":
        return FullBlockRepairSelector()
    if selector_name == "edit_proximity":
        return EditProximityRepairSelector(edit_radius)
    raise ValueError(f"unknown CacheSelect repair selector: {selector_name}")


# Summarize one selector decision for request-level observability.
def summarize_repair_selection(
    selector_name: str,
    candidates: Sequence[ResolvedPartialReuseCandidate],
    repair_instructions: Sequence[PartialReuseRepairInstruction],
    block_size: int,
) -> CacheSelectRepairMetrics:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    candidate_tokens = len(candidates) * block_size
    repair_required_tokens = (
        sum(candidate.requires_repair for candidate in candidates) * block_size
    )
    repair_tokens = sum(
        len(instruction.target_token_indices)
        for instruction in repair_instructions
    )
    if repair_tokens > repair_required_tokens:
        raise ValueError("repair plan exceeds tokens marked for repair")
    return CacheSelectRepairMetrics(
        selector=selector_name,
        candidate_tokens=candidate_tokens,
        repair_tokens=repair_tokens,
        skipped_repair_tokens=repair_required_tokens - repair_tokens,
    )
