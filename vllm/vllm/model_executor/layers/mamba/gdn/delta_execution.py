# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed token-span planning for experimental GDN affine execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


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
