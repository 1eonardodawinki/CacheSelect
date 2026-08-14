"""Runtime feature values for one partial-reuse candidate block."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Sequence

from cacheselect.reuse_opportunity import CandidateBlock, ReuseOpportunity


@dataclass(frozen=True)
class ChangedTokenRegion:
    """The smallest enclosing edit between two token sequences."""

    previous_start: int
    previous_end: int
    current_start: int
    current_end: int


# Count identical leading tokens before the first edit.
def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


# Count identical trailing tokens without overlapping the shared prefix.
def _common_suffix_length(
    left: Sequence[int], right: Sequence[int], prefix_length: int
) -> int:
    available = min(len(left), len(right)) - prefix_length
    count = 0
    while count < available and left[-1 - count] == right[-1 - count]:
        count += 1
    return count


# Locate the one enclosing token region that contains every prompt edit.
def locate_changed_token_region(
    previous_tokens: Sequence[int], current_tokens: Sequence[int]
) -> ChangedTokenRegion:
    prefix_length = _common_prefix_length(previous_tokens, current_tokens)
    suffix_length = _common_suffix_length(
        previous_tokens, current_tokens, prefix_length
    )
    return ChangedTokenRegion(
        previous_start=prefix_length,
        previous_end=len(previous_tokens) - suffix_length,
        current_start=prefix_length,
        current_end=len(current_tokens) - suffix_length,
    )


@dataclass(frozen=True)
class CandidateBlockFeatures:
    """Cheap model inputs describing one changed-prompt/candidate-block pair.

    This runtime structure deliberately contains no repair label. Benchmark
    code will attach ground truth later so labels cannot leak into serving.
    """

    previous_token_count: int
    current_token_count: int
    previous_changed_token_count: int
    current_changed_token_count: int
    block_size: int
    candidate_block_index: int
    candidate_position_ratio: float
    relative_block_offset: int
    nearest_changed_block_distance: int
    source_displacement_blocks: float
    candidate_share_of_native_recompute: float
    same_position_match: bool
    requires_repacking: bool
    changed_candidate_token_overlap_ratio: float | None
    introduced_candidate_token_overlap_ratio: float | None
    removed_candidate_token_overlap_ratio: float | None
    changed_candidate_token_jaccard: float | None


# Convert a possibly empty current edit interval into inclusive block bounds.
def _changed_block_bounds(
    region: ChangedTokenRegion, block_size: int
) -> tuple[int, int]:
    first_block = region.current_start // block_size
    if region.current_start == region.current_end:
        return first_block, first_block
    return first_block, (region.current_end - 1) // block_size


# Measure signed block distance from a candidate to the nearest changed block.
def _relative_block_offset(
    candidate_block_index: int, first_changed_block: int, last_changed_block: int
) -> int:
    if candidate_block_index < first_changed_block:
        return candidate_block_index - first_changed_block
    if candidate_block_index > last_changed_block:
        return candidate_block_index - last_changed_block
    return 0


# Pick the identical source occurrence closest to the candidate's new position.
def _nearest_source_start(candidate: CandidateBlock) -> int:
    return min(
        candidate.previous_starts,
        key=lambda start: (abs(start - candidate.current_start), start),
    )


# Extract geometry-only vectors for every content-identical candidate block.
def extract_candidate_block_geometry(
    previous_tokens: Sequence[int],
    current_tokens: Sequence[int],
    opportunity: ReuseOpportunity,
) -> tuple[CandidateBlockFeatures, ...]:
    if opportunity.previous_token_count != len(previous_tokens):
        raise ValueError("previous token count does not match reuse opportunity")
    if opportunity.current_token_count != len(current_tokens):
        raise ValueError("current token count does not match reuse opportunity")

    region = locate_changed_token_region(previous_tokens, current_tokens)
    first_changed_block, last_changed_block = _changed_block_bounds(
        region, opportunity.block_size
    )
    vectors = []
    for candidate in opportunity.candidate_blocks:
        relative_offset = _relative_block_offset(
            candidate.current_block_index,
            first_changed_block,
            last_changed_block,
        )
        source_start = _nearest_source_start(candidate)
        vectors.append(
            CandidateBlockFeatures(
                previous_token_count=len(previous_tokens),
                current_token_count=len(current_tokens),
                previous_changed_token_count=(
                    region.previous_end - region.previous_start
                ),
                current_changed_token_count=(region.current_end - region.current_start),
                block_size=opportunity.block_size,
                candidate_block_index=candidate.current_block_index,
                candidate_position_ratio=(
                    candidate.current_start / max(len(current_tokens), 1)
                ),
                relative_block_offset=relative_offset,
                nearest_changed_block_distance=abs(relative_offset),
                source_displacement_blocks=(
                    (candidate.current_start - source_start) / opportunity.block_size
                ),
                candidate_share_of_native_recompute=(
                    opportunity.candidate_share_of_native_recompute
                ),
                same_position_match=candidate.same_position_match,
                requires_repacking=candidate.requires_repacking,
                changed_candidate_token_overlap_ratio=None,
                introduced_candidate_token_overlap_ratio=None,
                removed_candidate_token_overlap_ratio=None,
                changed_candidate_token_jaccard=None,
            )
        )
    return tuple(vectors)


# Retain token multiplicity when isolating additions or removals from an edit.
def _token_difference(
    left_tokens: Sequence[int], right_tokens: Sequence[int]
) -> tuple[int, ...]:
    difference = Counter(left_tokens) - Counter(right_tokens)
    return tuple(difference.elements())


# Measure how much of one token region also appears inside a candidate block.
def _token_overlap_ratio(
    source_tokens: Sequence[int], candidate_tokens: Sequence[int]
) -> float:
    if not source_tokens:
        return 0.0
    overlap = Counter(source_tokens) & Counter(candidate_tokens)
    return sum(overlap.values()) / len(source_tokens)


# Measure set-level token similarity without overcounting repeated filler text.
def _token_jaccard(left_tokens: Sequence[int], right_tokens: Sequence[int]) -> float:
    left = set(left_tokens)
    right = set(right_tokens)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


# Add lexical edit-to-block relationships to the geometry-only feature vectors.
def extract_candidate_block_features(
    previous_tokens: Sequence[int],
    current_tokens: Sequence[int],
    opportunity: ReuseOpportunity,
) -> tuple[CandidateBlockFeatures, ...]:
    geometry = extract_candidate_block_geometry(
        previous_tokens, current_tokens, opportunity
    )
    region = locate_changed_token_region(previous_tokens, current_tokens)
    previous_changed = previous_tokens[region.previous_start : region.previous_end]
    current_changed = current_tokens[region.current_start : region.current_end]
    introduced = _token_difference(current_changed, previous_changed)
    removed = _token_difference(previous_changed, current_changed)

    vectors = []
    for row, candidate in zip(geometry, opportunity.candidate_blocks, strict=True):
        candidate_tokens = current_tokens[
            candidate.current_start : candidate.current_start + opportunity.block_size
        ]
        vectors.append(
            replace(
                row,
                changed_candidate_token_overlap_ratio=_token_overlap_ratio(
                    current_changed, candidate_tokens
                ),
                introduced_candidate_token_overlap_ratio=_token_overlap_ratio(
                    introduced, candidate_tokens
                ),
                removed_candidate_token_overlap_ratio=_token_overlap_ratio(
                    removed, candidate_tokens
                ),
                changed_candidate_token_jaccard=_token_jaccard(
                    current_changed, candidate_tokens
                ),
            )
        )
    return tuple(vectors)
