import unittest

from benchmarks.hybrid_gdn_breakeven import (
    build_hybrid_gdn_breakeven_conditions,
)


class HybridGDNBreakEvenTests(unittest.TestCase):
    # Interleave reuse sizes once per repetition to reduce time-order bias.
    def test_builds_interleaved_conditions(self) -> None:
        conditions = build_hybrid_gdn_breakeven_conditions((1, 4), 2)

        self.assertEqual(
            [(item.reused_block_count, item.repetition) for item in conditions],
            [(1, 1), (4, 1), (1, 2), (4, 2)],
        )

    # Reject duplicate sizes because they make repetition accounting ambiguous.
    def test_rejects_duplicate_reused_block_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_hybrid_gdn_breakeven_conditions((2, 2), 3)


if __name__ == "__main__":
    unittest.main()
