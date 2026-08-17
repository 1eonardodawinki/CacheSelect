from unittest import TestCase

from cacheselect.context_features import (
    extract_candidate_context_features,
    locate_token_edit_spans,
)
from cacheselect.reuse_opportunity import analyze_reuse_opportunity


class ContextFeatureTests(TestCase):
    # Keep two distant replacements separate and count only edits before a block.
    def test_extracts_edit_and_preceding_context_features(self):
        previous = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        current = (1, 2, 30, 40, 5, 6, 7, 8, 90, 100)
        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=2,
            block_size=2,
        )

        edits = locate_token_edit_spans(previous, current)
        rows = extract_candidate_context_features(previous, current, opportunity)

        self.assertEqual(len(edits), 2)
        self.assertEqual(
            [
                candidate.current_block_index
                for candidate in opportunity.candidate_blocks
            ],
            [2, 3],
        )
        self.assertEqual(rows[0].edit_span_count_before_candidate, 1)
        self.assertEqual(rows[0].previous_changed_tokens_before_candidate, 2)
        self.assertEqual(rows[0].current_changed_tokens_before_candidate, 2)
        self.assertEqual(rows[0].preceding_context_match_tokens, 0)
        self.assertEqual(rows[1].preceding_context_match_tokens, 2)

    # Report repeated source content so the selector can detect ambiguous matches.
    def test_counts_aligned_source_occurrences(self):
        previous = (1, 2, 5, 6, 5, 6, 9, 10)
        current = (1, 3, 5, 6, 9, 10)
        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=0,
            block_size=2,
        )

        rows = extract_candidate_context_features(previous, current, opportunity)
        feature_by_block = {
            candidate.current_block_index: row
            for candidate, row in zip(opportunity.candidate_blocks, rows, strict=True)
        }

        repeated = feature_by_block[1]
        self.assertEqual(repeated.source_occurrence_count, 2)
        self.assertEqual(repeated.aligned_source_occurrence_count, 2)

    # Count a deletion immediately before a candidate as changed prior context.
    def test_counts_boundary_deletion_before_candidate(self):
        previous = (1, 2, 3, 4, 5, 6)
        current = (1, 2, 5, 6)
        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=2,
            block_size=2,
        )

        rows = extract_candidate_context_features(previous, current, opportunity)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].edit_span_count_before_candidate, 1)
        self.assertEqual(rows[0].previous_changed_tokens_before_candidate, 2)
        self.assertEqual(rows[0].current_changed_tokens_before_candidate, 0)
