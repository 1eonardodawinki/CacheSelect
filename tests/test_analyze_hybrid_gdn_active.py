import unittest

from benchmarks.analyze_hybrid_gdn_active import analyze_hybrid_gdn_active


class AnalyzeHybridGDNActiveTests(unittest.TestCase):
    # Summarize exact active outputs and their latency relative to references.
    def test_analyzes_complete_active_run(self) -> None:
        rows = []
        for scenario, reference_seconds, active_seconds in (
            ("early_edit", 1.2, 0.8),
            ("middle_edit", 1.0, 0.5),
        ):
            rows.extend(
                [
                    {
                        "scenario": scenario,
                        "role": "reference",
                        "output_text": "same answer",
                        "client_wall_seconds": reference_seconds,
                        "time_to_first_token_ms": reference_seconds * 500,
                    },
                    {
                        "scenario": scenario,
                        "role": "target",
                        "output_text": "same answer",
                        "client_wall_seconds": active_seconds,
                        "time_to_first_token_ms": active_seconds * 400,
                        "cached_tokens": 128,
                        "gdn_delta_reuse": {
                            "active_complete": True,
                            "candidate_block_count": 2,
                            "active_executed_layer_count": 24,
                            "active_reused_layer_tokens": 3072,
                            "active_recomputed_layer_tokens": 1536,
                        },
                    },
                ]
            )
        summary = {
            "run_id": "active-run",
            "model": "Qwen/Qwen3.5-9B",
            "record_count": 160,
            "gdn_delta_active": {"passed": True},
            "observations": rows,
        }

        analysis = analyze_hybrid_gdn_active(summary)

        self.assertTrue(analysis["all_outputs_exact"])
        self.assertEqual(analysis["mean_speedup"], 1.75)
        self.assertEqual(analysis["mean_ttft_speedup"], 2.1875)
        self.assertEqual(analysis["comparisons"][0]["candidate_blocks"], 2)

    # Reject end-to-end timings that cannot isolate the accelerated prefill phase.
    def test_rejects_missing_ttft_measurement(self) -> None:
        summary = {
            "run_id": "active-run",
            "model": "Qwen/Qwen3.5-9B",
            "record_count": 160,
            "gdn_delta_active": {"passed": True},
            "observations": [
                {
                    "scenario": scenario,
                    "role": role,
                    "output_text": "same answer",
                    "client_wall_seconds": 1.0,
                    "cached_tokens": 128,
                    "gdn_delta_reuse": {
                        "active_complete": True,
                        "candidate_block_count": 2,
                        "active_executed_layer_count": 24,
                        "active_reused_layer_tokens": 3072,
                        "active_recomputed_layer_tokens": 1536,
                    },
                }
                for scenario in ("early_edit", "middle_edit")
                for role in ("reference", "target")
            ],
        }

        with self.assertRaisesRegex(ValueError, "time-to-first-token"):
            analyze_hybrid_gdn_active(summary)

    # Reject a summary that cannot prove every layer used the active path.
    def test_rejects_incomplete_active_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "passing active execution"):
            analyze_hybrid_gdn_active(
                {"gdn_delta_active": {"passed": False}, "observations": []}
            )


if __name__ == "__main__":
    unittest.main()
