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
                source_occurrence_count=len(candidate.previous_starts),
                aligned_source_occurrence_count=len(candidate.aligned_previous_starts),
            )
        )
    return tuple(rows)
