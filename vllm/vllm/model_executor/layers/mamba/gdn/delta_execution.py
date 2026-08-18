# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed token-span planning for experimental GDN affine execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from vllm.model_executor.layers.mamba.gdn.delta_cache import (
    GDNDeltaOperatorSidecar,
    apply_gdn_affine_operator,
)


@dataclass(frozen=True)
class GDNDeltaExecutionSpan:
    """One contiguous recurrent-compute or affine-reuse token interval."""

    sequence_index: int
    start_token: int
    end_token: int
    mode: Literal["recompute", "affine_reuse"]
    target_block_index: int | None = None
    source_contextual_hash: bytes | None = None


@dataclass(frozen=True)
class GDNDeltaSequenceExecutionPlan:
    """Complete, gap-free execution plan for one scheduled request interval."""

    sequence_index: int
    scheduled_start_token: int
    scheduled_end_token: int
    spans: tuple[GDNDeltaExecutionSpan, ...]
    reused_block_indices: tuple[int, ...]
    skipped_candidate_block_indices: tuple[int, ...]


@dataclass(frozen=True)
class GDNDeltaSequenceExecutionResult:
    """Outputs, final state and work split from one executed sequence plan."""

    outputs: torch.Tensor
    final_state: torch.Tensor
    recomputed_tokens: int
    reused_tokens: int


@dataclass(frozen=True)
class GDNDeltaActiveAdmission:
    """Fail-closed decision for entering behavior-changing GDN execution."""

    eligible: bool
    reason: str


GDNRecomputeSpan = Callable[
    [int, int, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


# Admit only the narrow eager single-prefill path implemented by the prototype.
def assess_gdn_delta_active_admission(
    *,
    execution_mode: str,
    mamba_cache_mode: str,
    prefill_backend: str,
    num_prefills: int,
    num_decodes: int,
    num_spec_decodes: int,
    reuse_candidates: Sequence[Sequence[tuple[int, bytes]]],
) -> GDNDeltaActiveAdmission:
    """Explain why active recurrence reuse may or may not run for this batch."""
    if execution_mode != "active":
        return GDNDeltaActiveAdmission(False, "shadow_mode")
    if mamba_cache_mode != "all":
        return GDNDeltaActiveAdmission(False, "checkpoint_mode_required")
    if prefill_backend != "triton":
        return GDNDeltaActiveAdmission(False, "triton_backend_required")
    if num_spec_decodes != 0:
        return GDNDeltaActiveAdmission(False, "speculative_decode_unsupported")
    if num_decodes != 0:
        return GDNDeltaActiveAdmission(False, "mixed_decode_unsupported")
    if num_prefills != 1 or len(reuse_candidates) != 1:
        return GDNDeltaActiveAdmission(False, "single_prefill_required")
    if not reuse_candidates[0]:
        return GDNDeltaActiveAdmission(False, "no_candidates")
    return GDNDeltaActiveAdmission(True, "eligible")


# Partition every scheduled request into normal recurrence and full-block reuse.
def build_gdn_delta_execution_plans(
    *,
    num_computed_tokens: Sequence[int],
    num_scheduled_tokens: Sequence[int],
    block_size: int,
    reuse_candidates: Sequence[Sequence[tuple[int, bytes]]],
) -> tuple[GDNDeltaSequenceExecutionPlan, ...]:
    """Return plans that cover every scheduled token exactly once."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    sequence_count = len(num_computed_tokens)
    if len(num_scheduled_tokens) != sequence_count:
        raise ValueError("scheduled counts must contain one value per sequence")
    if len(reuse_candidates) != sequence_count:
        raise ValueError("reuse candidates must contain one tuple per sequence")

    plans = []
    for sequence_index in range(sequence_count):
        scheduled_start = num_computed_tokens[sequence_index]
        scheduled_count = num_scheduled_tokens[sequence_index]
        if isinstance(scheduled_start, bool) or not isinstance(scheduled_start, int):
            raise TypeError("computed token counts must be integers")
        if isinstance(scheduled_count, bool) or not isinstance(scheduled_count, int):
            raise TypeError("scheduled token counts must be integers")
        if scheduled_start < 0 or scheduled_count < 0:
            raise ValueError("token counts must be nonnegative")
        scheduled_end = scheduled_start + scheduled_count

        candidate_by_block: dict[int, bytes] = {}
        for block_index, source_hash in reuse_candidates[sequence_index]:
            if (
                isinstance(block_index, bool)
                or not isinstance(block_index, int)
                or block_index < 0
            ):
                raise ValueError("candidate block indices must be nonnegative integers")
            if not isinstance(source_hash, bytes) or not source_hash:
                raise ValueError("candidate source hashes must be nonempty bytes")
            if block_index in candidate_by_block:
                raise ValueError("candidate target block indices must be unique")
            candidate_by_block[block_index] = source_hash

        reusable = []
        skipped = []
        for block_index in sorted(candidate_by_block):
            block_start = block_index * block_size
            block_end = block_start + block_size
            if block_start >= scheduled_start and block_end <= scheduled_end:
                reusable.append(block_index)
            else:
                skipped.append(block_index)

        spans = []
        cursor = scheduled_start
        for block_index in reusable:
            block_start = block_index * block_size
            block_end = block_start + block_size
            if cursor < block_start:
                spans.append(
                    GDNDeltaExecutionSpan(
                        sequence_index,
                        cursor,
                        block_start,
                        "recompute",
                    )
                )
            spans.append(
                GDNDeltaExecutionSpan(
                    sequence_index,
                    block_start,
                    block_end,
                    "affine_reuse",
                    block_index,
                    candidate_by_block[block_index],
                )
            )
            cursor = block_end
        if cursor < scheduled_end:
            spans.append(
                GDNDeltaExecutionSpan(
                    sequence_index,
                    cursor,
                    scheduled_end,
                    "recompute",
                )
            )

        plans.append(
            GDNDeltaSequenceExecutionPlan(
                sequence_index=sequence_index,
                scheduled_start_token=scheduled_start,
                scheduled_end_token=scheduled_end,
                spans=tuple(spans),
                reused_block_indices=tuple(reusable),
                skipped_candidate_block_indices=tuple(skipped),
            )
        )
    return tuple(plans)


# Execute one plan sequentially after atomically resolving every reuse operator.
def execute_gdn_delta_sequence_plan(
    plan: GDNDeltaSequenceExecutionPlan,
    *,
    initial_state: torch.Tensor,
    sidecar: GDNDeltaOperatorSidecar,
    recompute_span: GDNRecomputeSpan,
) -> GDNDeltaSequenceExecutionResult | None:
    """Return None before execution when any planned operator is unavailable."""
    reuse_hashes = tuple(
        span.source_contextual_hash
        for span in plan.spans
        if span.mode == "affine_reuse"
    )
    if any(source_hash is None for source_hash in reuse_hashes):
        raise ValueError("affine reuse spans must name a source operator")
    resolved_entries = sidecar.lookup_many(reuse_hashes)  # type: ignore[arg-type]
    if resolved_entries is None:
        return None

    state = initial_state
    outputs = []
    entry_index = 0
    recomputed_tokens = 0
    reused_tokens = 0
    for span in plan.spans:
        token_count = span.end_token - span.start_token
        if token_count <= 0:
            raise ValueError("execution spans must contain at least one token")
        if span.mode == "recompute":
            state, span_outputs = recompute_span(
                span.start_token,
                span.end_token,
                state,
            )
            recomputed_tokens += token_count
        else:
            entry = resolved_entries[entry_index]
            entry_index += 1
            state, span_outputs = apply_gdn_affine_operator(state, entry)
            reused_tokens += token_count
        if state.shape != initial_state.shape:
            raise ValueError("span execution returned an incompatible recurrent state")
        expected_output_shape = (
            token_count,
            initial_state.shape[0],
            initial_state.shape[1],
        )
        if span_outputs.shape != expected_output_shape:
            raise ValueError("span execution returned incompatible recurrent outputs")
        outputs.append(span_outputs)

    if entry_index != len(resolved_entries):
        raise RuntimeError("not every resolved GDN operator was consumed")
    if outputs:
        combined_outputs = torch.cat(outputs, dim=0)
    else:
        combined_outputs = torch.empty(
            (0, initial_state.shape[0], initial_state.shape[1]),
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
    expected_tokens = plan.scheduled_end_token - plan.scheduled_start_token
    if combined_outputs.shape[0] != expected_tokens:
        raise RuntimeError("execution plan did not cover its scheduled interval")
    return GDNDeltaSequenceExecutionResult(
        outputs=combined_outputs,
        final_state=state,
        recomputed_tokens=recomputed_tokens,
        reused_tokens=reused_tokens,
    )
