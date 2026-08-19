import unittest

from benchmarks.analyze_hybrid_gdn_breakeven import (
    analyze_hybrid_gdn_breakeven,
    render_hybrid_gdn_breakeven_markdown,
)


# Build one internally consistent matrix with a break-even at four blocks.
def _summary() -> dict:
    """Return a compact completed result for analyzer tests."""
    cells = []
    for count, wall, ttft in ((1, 0.9, 0.8), (4, 1.2, 1.3)):
        cells.append(
            {
                "reused_block_count": count,
                "repetitions": 3,
                "mean_reuse_fraction": count / (count + 1),
                "median_speedup": wall,
                "median_ttft_speedup": ttft,
                "all_outputs_exact": True,
                "break_even_met": count == 4,
            }
        )
    return {
        "experiment": "hybrid_gdn_reuse_breakeven",
        "run_id": "run",
        "model": "Qwen/Qwen3.5-9B",
        "block_size": 64,
        "trial_count": 6,
        "all_outputs_exact": True,
        "first_break_even_reused_block_count": 4,
        "cells": cells,
    }


class AnalyzeHybridGDNBreakEvenTests(unittest.TestCase):
    # Produce a conservative threshold and a readable evidence table.
    def test_analyzes_and_renders_break_even(self) -> None:
        analysis = analyze_hybrid_gdn_breakeven(_summary())
        markdown = render_hybrid_gdn_breakeven_markdown(analysis)

        self.assertEqual(analysis["decision"], "MEASURED_BREAK_EVEN")
        self.assertEqual(analysis["recommended_minimum_reused_blocks"], 4)
        self.assertIn("| 4 | 80.0% | 1.200x | 1.300x | yes | yes |", markdown)

    # Reject hand-edited artifacts whose headline contradicts their cells.
    def test_rejects_inconsistent_first_break_even(self) -> None:
        summary = _summary()
        summary["first_break_even_reused_block_count"] = 1

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            analyze_hybrid_gdn_breakeven(summary)


if __name__ == "__main__":
    unittest.main()
