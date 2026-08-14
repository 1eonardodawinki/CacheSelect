"""Runtime feature values for one partial-reuse candidate block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


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
    changed_candidate_token_overlap_ratio: float
    introduced_candidate_token_overlap_ratio: float
    removed_candidate_token_overlap_ratio: float
    changed_candidate_token_jaccard: float
