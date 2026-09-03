import unittest

from benchmarks.run_span_breakeven import _prompts, summarize
from cacheselect.reuse_opportunity import analyze_reuse_opportunity


class SpanBreakEvenTest(unittest.TestCase):
    def test_builds_one_exact_aligned_gap(self) -> None:
        source, target = _prompts(2, 3, 16)

        self.assertEqual(len(source), 128)
        self.assertEqual(source[48:80], target[48:80])
        self.assertNotEqual(source[:48], target[:48])
        self.assertNotEqual(source[80:], target[80:])
        opportunity = analyze_reuse_opportunity(
            source, target, native_cached_tokens=0, block_size=16
        )
        self.assertEqual(
            [block.current_block_index for block in opportunity.candidate_blocks],
            [3, 4],
        )

    def test_can_shift_source_gap_to_require_repacking(self) -> None:
        source, target = _prompts(2, 3, 16, source_offset_tokens=8)

        self.assertEqual(len(source), len(target))
        opportunity = analyze_reuse_opportunity(
            source, target, native_cached_tokens=0, block_size=16
        )
        self.assertEqual(
            [block.previous_starts for block in opportunity.candidate_blocks],
            [(56,), (72,)],
        )
        self.assertTrue(
            all(block.requires_repacking for block in opportunity.candidate_blocks)
        )

    def test_reports_first_median_split_win(self) -> None:
        rows = [
            {
                "gap_blocks": gap,
                "gap_tokens": gap * 16,
                "merged_ttft_ms": merged,
                "split_ttft_ms": split,
            }
            for gap, merged, split in (
                (1, 10.0, 12.0),
                (1, 11.0, 13.0),
                (2, 15.0, 14.0),
                (2, 16.0, 13.0),
            )
        ]

        result = summarize(rows)

        self.assertEqual(result["first_measured_split_win_blocks"], 2)
        self.assertEqual(result["cells"][0]["median_split_delta_ms"], 2.0)
        self.assertEqual(result["cells"][1]["split_faster_repetitions"], 2)


if __name__ == "__main__":
    unittest.main()
