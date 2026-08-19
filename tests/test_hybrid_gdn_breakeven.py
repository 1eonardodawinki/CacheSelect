import unittest

from benchmarks.hybrid_gdn_breakeven import (
    build_hybrid_gdn_breakeven_conditions,
    calibrate_hybrid_gdn_breakeven_prompt,
)


# Turn the controlled prompt text into predictable token geometry for unit tests.
def _fake_tokenize(prompt: str) -> tuple[int, ...]:
    """Represent the marker, repeated filler, and instruction as integer tokens."""
    marker = 11 if "marker A" in prompt else 12
    filler_count = prompt.count("Stable context follows")
    return (90, marker, *(30 for _ in range(filler_count * 2)), 70, 71)


class HybridGDNBreakEvenTests(unittest.TestCase):
    # Interleave reuse sizes once per repetition to reduce time-order bias.
    def test_builds_interleaved_conditions(self) -> None:
        conditions = build_hybrid_gdn_breakeven_conditions((1, 4), 2)

        self.assertEqual(
            [(item.reused_block_count, item.repetition) for item in conditions],
            [(1, 1), (4, 1), (1, 2), (4, 2)],
        )

    # Calibrate one changed block followed by the exact requested shared blocks.
    def test_calibrates_exact_reusable_block_geometry(self) -> None:
        pair = calibrate_hybrid_gdn_breakeven_prompt(
            reused_block_count=4,
            block_size=4,
            tokenize_prompt=_fake_tokenize,
        )

        self.assertEqual(pair.full_block_count, 5)
        self.assertEqual(pair.reused_block_count, 4)
        self.assertGreaterEqual(pair.shared_suffix_tokens, 16)
        self.assertLess(pair.shared_prefix_tokens, 4)
        self.assertNotEqual(pair.source_prompt, pair.target_prompt)

    # Reject duplicate sizes because they make repetition accounting ambiguous.
    def test_rejects_duplicate_reused_block_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_hybrid_gdn_breakeven_conditions((2, 2), 3)


if __name__ == "__main__":
    unittest.main()
