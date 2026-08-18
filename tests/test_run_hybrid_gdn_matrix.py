import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.run_hybrid_gdn_matrix import run_hybrid_gdn_matrix


class RunHybridGDNMatrixTests(unittest.TestCase):
    # Execute conditions in interleaved order and persist the consolidated result.
    def test_runs_matrix_against_one_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            # Return the minimum fail-closed evidence required by the runner.
            def fake_run(**kwargs):
                return {
                    "record_count": kwargs["record_count"],
                    "request_ledger": str(kwargs["request_log_dir"] / "requests.jsonl"),
                    "gdn_delta_active": {"passed": True},
                    "reference_validation_passed": True,
                }

            # Preserve the context identity while avoiding HTTP in this unit test.
            def fake_analyze(result):
                return {
                    "record_count": result["record_count"],
                    "all_outputs_exact": True,
                    "comparisons": [
                        {
                            "scenario": "early_edit",
                            "speedup": 1.25,
                            "ttft_speedup": 1.5,
                            "exact_output_match": True,
                            "native_cached_tokens": 64,
                            "reused_layer_tokens": 2048,
                        }
                    ],
                }

            with (
                patch(
                    "benchmarks.run_hybrid_gdn_matrix.run_hybrid_apc_baseline",
                    side_effect=fake_run,
                ) as run,
                patch(
                    "benchmarks.run_hybrid_gdn_matrix.analyze_hybrid_gdn_active",
                    side_effect=fake_analyze,
                ),
            ):
                matrix = run_hybrid_gdn_matrix(
                    base_url="http://127.0.0.1:8000",
                    model="test-model",
                    run_id="matrix-run",
                    output_root=root / "results",
                    request_log_root=root / "logs",
                    record_counts=(64, 160),
                    repetitions=2,
                )

            self.assertEqual(
                [call.kwargs["record_count"] for call in run.call_args_list],
                [64, 160, 64, 160],
            )
            self.assertEqual(matrix["run_count"], 4)
            self.assertTrue(matrix["all_outputs_exact"])
            self.assertTrue((root / "results" / "matrix-summary.json").is_file())
            self.assertEqual(len(matrix["runs"]), 4)

    # Stop before aggregation when active execution was not proven.
    def test_rejects_inactive_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "benchmarks.run_hybrid_gdn_matrix.run_hybrid_apc_baseline",
                return_value={
                    "gdn_delta_active": {"passed": False},
                    "reference_validation_passed": True,
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "did not execute"):
                    run_hybrid_gdn_matrix(
                        base_url="http://127.0.0.1:8000",
                        model="test-model",
                        run_id="inactive",
                        output_root=root / "results",
                        request_log_root=root / "logs",
                        record_counts=(64,),
                        repetitions=1,
                    )


if __name__ == "__main__":
    unittest.main()
