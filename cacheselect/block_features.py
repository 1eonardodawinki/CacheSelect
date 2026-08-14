"""Runtime feature values for one partial-reuse candidate block."""

from __future__ import annotations

from dataclasses import dataclass


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
