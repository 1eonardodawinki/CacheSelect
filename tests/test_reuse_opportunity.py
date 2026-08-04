from unittest import TestCase

from cacheselect.reuse_opportunity import analyze_reuse_opportunity


class ReuseOpportunityTests(TestCase):
    def test_changed_middle_exposes_unchanged_suffix_blocks(self):
        previous = list(range(8)) + list(range(20, 24)) + list(range(40, 48))
        current = list(range(8)) + list(range(30, 34)) + list(range(40, 48))

        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=8,
            block_size=4,
        )

        self.assertEqual(opportunity.exact_common_prefix_tokens, 8)
        self.assertEqual(opportunity.common_suffix_tokens, 8)
        self.assertEqual(opportunity.monotonic_post_edit_matching_tokens, 8)
        self.assertEqual(opportunity.candidate_block_count, 2)
        self.assertEqual(opportunity.candidate_token_count, 8)
        self.assertEqual(opportunity.whole_source_block_count, 2)
        self.assertEqual(opportunity.repacking_required_block_count, 0)

    def test_location_independent_matching_detects_reordered_blocks(self):
        block_a = [1, 2, 3, 4]
        block_b = [5, 6, 7, 8]
        block_c = [9, 10, 11, 12]
        previous = block_a + block_b + block_c
        current = block_a + block_c + block_b

        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=4,
            block_size=4,
        )

        self.assertEqual(opportunity.candidate_block_count, 2)
        self.assertEqual(opportunity.moved_candidate_block_count, 2)
        self.assertEqual(opportunity.whole_source_block_count, 2)
        self.assertEqual(
            [block.current_start for block in opportunity.candidate_blocks],
            [4, 8],
        )

    def test_unaligned_source_is_marked_for_repacking(self):
        previous = [1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 23]
        current = [1, 2, 3, 4, 99, 10, 11, 12, 13, 20, 21, 22, 23]

        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=4,
            block_size=4,
        )

        self.assertEqual(opportunity.candidate_block_count, 1)
        candidate = opportunity.candidate_blocks[0]
        self.assertEqual(candidate.current_start, 8)
        self.assertEqual(candidate.previous_starts, (7,))
        self.assertTrue(candidate.requires_repacking)
        self.assertFalse(candidate.has_whole_source_block)

    def test_prefix_alignment_loss_is_reported(self):
        previous = list(range(12))
        current = list(range(10)) + [99, 100]

        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=8,
            block_size=4,
        )

        self.assertEqual(opportunity.exact_common_prefix_tokens, 10)
        self.assertEqual(opportunity.prefix_alignment_loss_tokens, 2)

    def test_invalid_native_hit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact common prefix"):
            analyze_reuse_opportunity(
                [1, 2, 3],
                [1, 9, 3],
                native_cached_tokens=2,
                block_size=1,
            )

        with self.assertRaisesRegex(ValueError, "block_size"):
            analyze_reuse_opportunity(
                [1],
                [1],
                native_cached_tokens=1,
                block_size=0,
            )
