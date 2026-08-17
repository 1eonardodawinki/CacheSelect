"""Cheap context signals for candidate KV-cache blocks."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from cacheselect.reuse_opportunity import ReuseOpportunity


@dataclass(frozen=True)
class TokenEditSpan:
    """One non-equal token interval between two prompt versions."""

    previous_start: int
    previous_end: int
    current_start: int
    current_end: int


@dataclass(frozen=True)
class CandidateContextFeatures:
    """Additional prompt-only signals for one reuse candidate."""

    edit_span_count: int
    edit_span_count_before_candidate: int
    previous_changed_tokens_before_candidate: int
    current_changed_tokens_before_candidate: int
    preceding_context_match_tokens: int
    preceding_matching_run_blocks: int
    following_matching_run_blocks: int
    matching_run_length_blocks: int
    source_occurrence_count: int
    aligned_source_occurrence_count: int


# Preserve separate insertions, deletions and replacements instead of one envelope.
def locate_token_edit_spans(
    previous_tokens: Sequence[int], current_tokens: Sequence[int]
) -> tuple[TokenEditSpan, ...]:
    matcher = SequenceMatcher(
        None,
        previous_tokens,
        current_tokens,
        autojunk=False,
    )
    return tuple(
        TokenEditSpan(previous_start, previous_end, current_start, current_end)
        for tag, previous_start, previous_end, current_start, current_end in matcher.get_opcodes()
        if tag != "equal"
    )


# Count identical context tokens immediately before one old/new block occurrence.
def _preceding_match_tokens(
    previous_tokens: Sequence[int],
    current_tokens: Sequence[int],
    *,
    previous_start: int,
    current_start: int,
) -> int:
    matched = 0
    limit = min(previous_start, current_start)
    while (
        matched < limit
        and previous_tokens[previous_start - matched - 1]
        == current_tokens[current_start - matched - 1]
    ):
        matched += 1
    return matched


# Treat a zero-width deletion at the block boundary as an earlier context edit.
def _edit_precedes_candidate(edit: TokenEditSpan, candidate_start: int) -> bool:
    return edit.current_start < candidate_start or (
        edit.current_start == candidate_start and edit.current_start == edit.current_end
    )


# Count full matching blocks around one candidate/source occurrence pair.
def _matching_block_run(
    previous_tokens: Sequence[int],
    current_tokens: Sequence[int],
    *,
    previous_start: int,
    current_start: int,
    block_size: int,
) -> tuple[int, int]:
    preceding = 0
    while (
        previous_start - (preceding + 1) * block_size >= 0
        and current_start - (preceding + 1) * block_size >= 0
    ):
        step = preceding + 1
        previous_block_start = previous_start - step * block_size
        current_block_start = current_start - step * block_size
        if tuple(
            previous_tokens[previous_block_start : previous_block_start + block_size]
        ) != tuple(
            current_tokens[current_block_start : current_block_start + block_size]
        ):
            break
        preceding += 1

    following = 0
    while previous_start + (following + 2) * block_size <= len(
        previous_tokens
    ) and current_start + (following + 2) * block_size <= len(current_tokens):
        step = following + 1
        previous_block_start = previous_start + step * block_size
        current_block_start = current_start + step * block_size
        if tuple(
            previous_tokens[previous_block_start : previous_block_start + block_size]
        ) != tuple(
            current_tokens[current_block_start : current_block_start + block_size]
        ):
            break
        following += 1
    return preceding, following


# Measure separate edits and source-context agreement for every candidate block.
def extract_candidate_context_features(
    previous_tokens: Sequence[int],
    current_tokens: Sequence[int],
    opportunity: ReuseOpportunity,
) -> tuple[CandidateContextFeatures, ...]:
    if opportunity.previous_token_count != len(previous_tokens):
        raise ValueError("previous token count does not match reuse opportunity")
    if opportunity.current_token_count != len(current_tokens):
        raise ValueError("current token count does not match reuse opportunity")
    edits = locate_token_edit_spans(previous_tokens, current_tokens)
    rows = []
    for candidate in opportunity.candidate_blocks:
        current_start = candidate.current_start
        edits_before = tuple(
            edit for edit in edits if _edit_precedes_candidate(edit, current_start)
        )
        preceding_matches = (
            _preceding_match_tokens(
                previous_tokens,
                current_tokens,
                previous_start=previous_start,
                current_start=current_start,
            )
            for previous_start in candidate.previous_starts
        )
        # Prefer executable aligned occurrences when measuring stable block runs.
        source_starts = candidate.aligned_previous_starts or candidate.previous_starts
        preceding_run, following_run = max(
            (
                _matching_block_run(
                    previous_tokens,
                    current_tokens,
                    previous_start=previous_start,
                    current_start=current_start,
                    block_size=opportunity.block_size,
                )
                for previous_start in source_starts
            ),
            key=lambda run: (sum(run), run[0]),
        )
        rows.append(
            CandidateContextFeatures(
                edit_span_count=len(edits),
                edit_span_count_before_candidate=len(edits_before),
                previous_changed_tokens_before_candidate=sum(
                    edit.previous_end - edit.previous_start for edit in edits_before
                ),
                current_changed_tokens_before_candidate=sum(
                    min(edit.current_end, current_start) - edit.current_start
                    for edit in edits_before
                ),
                preceding_context_match_tokens=max(preceding_matches, default=0),
                preceding_matching_run_blocks=preceding_run,
                following_matching_run_blocks=following_run,
                matching_run_length_blocks=preceding_run + 1 + following_run,
                source_occurrence_count=len(candidate.previous_starts),
                aligned_source_occurrence_count=len(candidate.aligned_previous_starts),
            )
        )
    return tuple(rows)
