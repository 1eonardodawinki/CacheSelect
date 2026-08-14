from unittest import TestCase

from cacheselect.block_features import locate_changed_token_region


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
