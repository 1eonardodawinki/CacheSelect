# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve CacheSelect mappings and build inert worker copy plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.config.cache import CacheSelectRepairSelector
    from vllm.v1.core.partial_reuse import PartialReusePlan

from vllm.v1.core.partial_reuse import CacheSelectRepairMetrics


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


# Convert approved mappings into inert source-to-destination copy instructions.
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
