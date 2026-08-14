from unittest import TestCase

from cacheselect.block_features import (
    extract_candidate_block_geometry,
    locate_changed_token_region,
)
from cacheselect.reuse_opportunity import analyze_reuse_opportunity


class ChangedTokenRegionTests(TestCase):
    # Verify replacements, insertions, deletions, and exact matches stay bounded.
    def test_locates_enclosing_changed_region(self):
        cases = (
            ([1, 2, 3, 4], [1, 2, 9, 4], (2, 3, 2, 3)),
            ([1, 2, 3, 4], [1, 2, 8, 9, 3, 4], (2, 2, 2, 4)),
            ([1, 2, 8, 9, 3, 4], [1, 2, 3, 4], (2, 4, 2, 2)),
            ([1, 2, 3], [1, 2, 3], (3, 3, 3, 3)),
        )

        for previous, current, expected in cases:
            with self.subTest(previous=previous, current=current):
                region = locate_changed_token_region(previous, current)
                self.assertEqual(
                    (
                        region.previous_start,
                        region.previous_end,
                        region.current_start,
                        region.current_end,
                    ),
                    expected,
                )


class CandidateBlockGeometryTests(TestCase):
    # Verify unchanged blocks are measured outward from a replacement block.
    def test_extracts_edit_distance_and_prompt_position(self):
        previous = list(range(4)) + [10, 11, 12, 13] + list(range(20, 28))
        current = list(range(4)) + [40, 41, 42, 43] + list(range(20, 28))
        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=4,
            block_size=4,
        )

        vectors = extract_candidate_block_geometry(previous, current, opportunity)

        self.assertEqual([row.candidate_block_index for row in vectors], [2, 3])
        self.assertEqual(
            [row.nearest_changed_block_distance for row in vectors], [1, 2]
        )
        self.assertEqual([row.relative_block_offset for row in vectors], [1, 2])
        self.assertEqual([row.source_displacement_blocks for row in vectors], [0, 0])
        self.assertEqual(vectors[0].candidate_position_ratio, 0.5)
        self.assertTrue(all(row.same_position_match for row in vectors))
        self.assertTrue(
            all(row.changed_candidate_token_overlap_ratio is None for row in vectors)
        )

    # Verify reordered blocks expose signed movement from their source positions.
    def test_extracts_source_displacement_for_reordered_blocks(self):
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

        vectors = extract_candidate_block_geometry(previous, current, opportunity)

        self.assertEqual([row.source_displacement_blocks for row in vectors], [-1, 1])
        self.assertTrue(all(not row.same_position_match for row in vectors))
