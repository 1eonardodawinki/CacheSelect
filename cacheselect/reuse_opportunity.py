"""Deterministic estimates of reuse available beyond native prefix caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import Match, SequenceMatcher
from typing import Any, Sequence


@dataclass(frozen=True)
class MatchingTokenSpan:
    """One monotonic exact-token match between two rendered prompts."""

    previous_start: int
    current_start: int
    token_count: int


@dataclass(frozen=True)
class CandidateBlock:
    """One current block whose token content exists in the previous prompt.

    Token identity makes the block a partial-reuse candidate, not a safe KV
    hit. Its KV state may still need repair because its preceding context can
    have changed.
    """

    current_block_index: int
    current_start: int
    previous_starts: tuple[int, ...]
    aligned_previous_starts: tuple[int, ...]
    same_position_match: bool

    @property
    def has_whole_source_block(self) -> bool:
        """Whether an identical source occurrence starts on a block boundary."""
        return bool(self.aligned_previous_starts)

    @property
    def moved(self) -> bool:
        """Whether the content is unavailable at the same prompt position."""
        return not self.same_position_match

    @property
    def requires_repacking(self) -> bool:
        """Whether all source occurrences cross physical block boundaries."""
        return not self.has_whole_source_block

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "has_whole_source_block": self.has_whole_source_block,
                "moved": self.moved,
                "requires_repacking": self.requires_repacking,
            }
        )
        return payload


@dataclass(frozen=True)
class ReuseOpportunity:
    """Measured native reuse and content-identical post-prefix candidates."""

    block_size: int
    previous_token_count: int
    current_token_count: int
    exact_common_prefix_tokens: int
    native_cached_tokens: int
    prefix_alignment_loss_tokens: int
    common_suffix_tokens: int
    monotonic_post_edit_matching_tokens: int
    recomputed_full_block_count: int
    candidate_block_count: int
    candidate_token_count: int
    whole_source_block_count: int
    repacking_required_block_count: int
    moved_candidate_block_count: int
    candidate_share_of_native_recompute: float
    monotonic_matching_spans: tuple[MatchingTokenSpan, ...]
    candidate_blocks: tuple[CandidateBlock, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["monotonic_matching_spans"] = [
            asdict(span) for span in self.monotonic_matching_spans
        ]
        payload["candidate_blocks"] = [
            block.to_dict() for block in self.candidate_blocks
        ]
        return payload


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def _common_suffix_length(
    left: Sequence[int],
    right: Sequence[int],
    *,
    common_prefix_tokens: int,
) -> int:
    available = min(len(left), len(right)) - common_prefix_tokens
    count = 0
    while count < available and left[-1 - count] == right[-1 - count]:
        count += 1
    return count


def _matching_spans(
    previous_tokens: Sequence[int],
    current_tokens: Sequence[int],
) -> tuple[MatchingTokenSpan, ...]:
    matcher = SequenceMatcher(
        None,
        list(previous_tokens),
        list(current_tokens),
        autojunk=False,
    )
    return tuple(
        MatchingTokenSpan(
            previous_start=match.a,
            current_start=match.b,
            token_count=match.size,
        )
        for match in matcher.get_matching_blocks()
        if match != Match(len(previous_tokens), len(current_tokens), 0)
        and match.size > 0
    )


def _post_edit_matching_tokens(
    spans: Sequence[MatchingTokenSpan],
    *,
    exact_common_prefix_tokens: int,
) -> int:
    total = 0
    for span in spans:
        span_end = span.current_start + span.token_count
        total += max(0, span_end - max(span.current_start, exact_common_prefix_tokens))
    return total


def _content_occurrences(
    tokens: Sequence[int],
    *,
    width: int,
) -> dict[tuple[int, ...], tuple[int, ...]]:
    mutable: dict[tuple[int, ...], list[int]] = {}
    for start in range(0, len(tokens) - width + 1):
        block = tuple(tokens[start : start + width])
        mutable.setdefault(block, []).append(start)
    return {block: tuple(starts) for block, starts in mutable.items()}


def analyze_reuse_opportunity(
    previous_tokens: Sequence[int],
    current_tokens: Sequence[int],
    *,
    native_cached_tokens: int,
    block_size: int = 16,
) -> ReuseOpportunity:
    """Quantify exact token content that APC leaves after its prefix hit.

    Candidate blocks are current-position-aligned full blocks, beyond native
    APC's measured hit, whose exact token tuple occurs anywhere in the previous
    prompt. This content-only match deliberately ignores contextual KV
    validity. It measures opportunity for a future repair algorithm rather
    than claiming that the blocks are immediately reusable.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not 0 <= native_cached_tokens <= len(current_tokens):
        raise ValueError(
            "native_cached_tokens must be between zero and current token count"
        )

    exact_prefix = _common_prefix_length(previous_tokens, current_tokens)
    if native_cached_tokens > exact_prefix:
        raise ValueError("native_cached_tokens exceeds the exact common prefix")

    common_suffix = _common_suffix_length(
        previous_tokens,
        current_tokens,
        common_prefix_tokens=exact_prefix,
    )
    spans = _matching_spans(previous_tokens, current_tokens)
    post_edit_matching = _post_edit_matching_tokens(
        spans,
        exact_common_prefix_tokens=exact_prefix,
    )

    previous_occurrences = _content_occurrences(
        previous_tokens,
        width=block_size,
    )
    first_recomputed_block = (
        (native_cached_tokens + block_size - 1) // block_size * block_size
    )
    current_full_block_end = len(current_tokens) // block_size * block_size
    candidate_blocks: list[CandidateBlock] = []
    for current_start in range(
        first_recomputed_block,
        current_full_block_end,
        block_size,
    ):
        token_block = tuple(current_tokens[current_start : current_start + block_size])
        previous_starts = previous_occurrences.get(token_block)
        if previous_starts is None:
            continue
        aligned_starts = tuple(
            start for start in previous_starts if start % block_size == 0
        )
        candidate_blocks.append(
            CandidateBlock(
                current_block_index=current_start // block_size,
                current_start=current_start,
                previous_starts=previous_starts,
                aligned_previous_starts=aligned_starts,
                same_position_match=current_start in previous_starts,
            )
        )

    recomputed_full_blocks = max(
        0,
        (current_full_block_end - first_recomputed_block) // block_size,
    )
    candidate_count = len(candidate_blocks)
    candidate_tokens = candidate_count * block_size
    native_recompute_tokens = len(current_tokens) - native_cached_tokens

    return ReuseOpportunity(
        block_size=block_size,
        previous_token_count=len(previous_tokens),
        current_token_count=len(current_tokens),
        exact_common_prefix_tokens=exact_prefix,
        native_cached_tokens=native_cached_tokens,
        prefix_alignment_loss_tokens=exact_prefix - native_cached_tokens,
        common_suffix_tokens=common_suffix,
        monotonic_post_edit_matching_tokens=post_edit_matching,
        recomputed_full_block_count=recomputed_full_blocks,
        candidate_block_count=candidate_count,
        candidate_token_count=candidate_tokens,
        whole_source_block_count=sum(
            block.has_whole_source_block for block in candidate_blocks
        ),
        repacking_required_block_count=sum(
            block.requires_repacking for block in candidate_blocks
        ),
        moved_candidate_block_count=sum(block.moved for block in candidate_blocks),
        candidate_share_of_native_recompute=(
            candidate_tokens / native_recompute_tokens
            if native_recompute_tokens
            else 0.0
        ),
        monotonic_matching_spans=spans,
        candidate_blocks=tuple(candidate_blocks),
    )
