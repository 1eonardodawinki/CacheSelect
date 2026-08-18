import unittest

from benchmarks.hybrid_gdn_matrix import (
    build_hybrid_gdn_matrix_conditions,
    summarize_hybrid_gdn_matrix,
)


class HybridGDNMatrixTests(unittest.TestCase):
    # Interleave lengths once per repetition in a deterministic order.
    def test_builds_interleaved_matrix_conditions(self) -> None:
        conditions = build_hybrid_gdn_matrix_conditions((64, 160), 2)

        self.assertEqual(
            [(item.record_count, item.repetition) for item in conditions],
            [(64, 1), (160, 1), (64, 2), (160, 2)],
        )

    # Aggregate repeated speed and correctness evidence per length/scenario.
    def test_summarizes_matrix_cells(self) -> None:
        analyses = []
        for speedup in (1.2, 1.4):
            analyses.append(
                {
                    "record_count": 64,
                    "all_outputs_exact": True,
                    "comparisons": [
                        {
                            "scenario": "early_edit",
                            "speedup": speedup,
                            "ttft_speedup": speedup + 0.5,
                            "exact_output_match": True,
                            "native_cached_tokens": 64,
                            "reused_layer_tokens": 3072,
                        }
                    ],
                }
            )

        summary = summarize_hybrid_gdn_matrix(analyses)

        self.assertTrue(summary["all_outputs_exact"])
        self.assertEqual(summary["run_count"], 2)
        self.assertAlmostEqual(summary["cells"][0]["mean_speedup"], 1.3)
        self.assertAlmostEqual(summary["cells"][0]["mean_ttft_speedup"], 1.8)
        self.assertEqual(summary["cells"][0]["repetitions"], 2)

    # Reject duplicated lengths that would make matrix accounting ambiguous.
    def test_rejects_duplicate_record_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_hybrid_gdn_matrix_conditions((64, 64), 2)


if __name__ == "__main__":
    unittest.main()
