import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.run_hybrid_apc_baseline import (
    PREFIX_HIT_METRIC,
    build_hybrid_apc_scenarios,
    parse_prometheus_counter,
    run_hybrid_apc_baseline,
)
from observability.request_recorder import validate_ledger


class HybridAPCBaselineTest(unittest.TestCase):
    # Confirm the four transitions isolate the intended kinds of prompt change.
    def test_builds_controlled_scenarios(self) -> None:
        scenarios = {item.name: item for item in build_hybrid_apc_scenarios()}

        self.assertEqual(
            set(scenarios), {"exact", "append_only", "early_edit", "middle_edit"}
        )
        self.assertEqual(
            scenarios["exact"].source_prompt, scenarios["exact"].target_prompt
        )
        self.assertTrue(
            scenarios["append_only"].target_prompt.startswith(
                scenarios["append_only"].source_prompt.split("\n", 1)[0]
            )
        )
        self.assertNotEqual(
            scenarios["early_edit"].source_prompt,
            scenarios["early_edit"].target_prompt,
        )
        self.assertNotEqual(
            scenarios["middle_edit"].source_prompt,
            scenarios["middle_edit"].target_prompt,
        )

    # Confirm Prometheus label sets are parsed without matching created gauges.
    def test_parses_prometheus_counter(self) -> None:
        metrics = (
            f'{PREFIX_HIT_METRIC}{{engine="0"}} 528.0\n'
            f'{PREFIX_HIT_METRIC}_created{{engine="0"}} 99.0\n'
            f'{PREFIX_HIT_METRIC}{{engine="1"}} 1056.0\n'
        )

        self.assertEqual(parse_prometheus_counter(metrics, PREFIX_HIT_METRIC), 1584.0)

    # Confirm the runner records all eight requests and exact counter deltas.
    def test_records_source_and_target_requests(self) -> None:
        counters = iter((index * 100, index * 16) for index in range(16))

        # Return deterministic fake counters for each before/after measurement.
        def fake_counters(_url: str, _timeout: float) -> tuple[int, int]:
            return next(counters)

        # Return the minimal successful vLLM response needed by the runner.
        def fake_post(_url: str, _payload: dict, _timeout: float) -> dict:
            return {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "summary.json"
            with (
                patch(
                    "benchmarks.run_hybrid_apc_baseline._read_prefix_counters",
                    side_effect=fake_counters,
                ),
                patch(
                    "benchmarks.run_hybrid_apc_baseline._post_json",
                    side_effect=fake_post,
                ),
            ):
                result = run_hybrid_apc_baseline(
                    base_url="http://example.test:8000",
                    model="test-model",
                    run_id="hybrid-test",
                    request_log_dir=root / "logs",
                    output=output,
                )

            self.assertEqual(len(result["observations"]), 8)
            self.assertTrue(result["ledger_complete"])
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text())["run_id"], "hybrid-test")
            ledger = validate_ledger(result["request_ledger"])
            self.assertEqual((ledger.started, ledger.completed), (8, 8))
            self.assertTrue(
                all(row["queried_tokens"] == 100 for row in result["observations"])
            )
            self.assertTrue(
                all(row["cached_tokens"] == 16 for row in result["observations"])
            )

    # Confirm optional references are isolated, recorded, and compared.
    def test_compares_checkpoint_outputs_with_uncached_references(self) -> None:
        counters = iter((index * 100, index * 16) for index in range(24))

        # Return deterministic fake counters for all twelve requests.
        def fake_counters(_url: str, _timeout: float) -> tuple[int, int]:
            return next(counters)

        # Return identical deterministic text for reference and reused requests.
        def fake_post(_url: str, _payload: dict, _timeout: float) -> dict:
            return {
                "choices": [
                    {
                        "message": {"content": "stable output"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "benchmarks.run_hybrid_apc_baseline._read_prefix_counters",
                    side_effect=fake_counters,
                ),
                patch(
                    "benchmarks.run_hybrid_apc_baseline._post_json",
                    side_effect=fake_post,
                ),
            ):
                result = run_hybrid_apc_baseline(
                    base_url="http://example.test:8000",
                    model="test-model",
                    run_id="hybrid-validation",
                    request_log_dir=root / "logs",
                    output=root / "summary.json",
                    validate_against_reference=True,
                )

        self.assertEqual(len(result["observations"]), 12)
        self.assertEqual(len(result["reference_checks"]), 4)
        self.assertTrue(
            all(check["exact_output_match"] for check in result["reference_checks"])
        )


if __name__ == "__main__":
    unittest.main()
