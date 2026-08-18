import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.run_hybrid_apc_baseline import (
    PREFIX_HIT_METRIC,
    assess_gdn_delta_preflight_observability,
    assess_gdn_delta_shadow_execution,
    assess_hybrid_token_alignment,
    assess_hybrid_checkpoint_reuse,
    build_hybrid_apc_scenarios,
    parse_prometheus_counter,
    run_hybrid_apc_baseline,
)
from observability.request_recorder import validate_ledger


class HybridAPCBaselineTest(unittest.TestCase):
    # Detect an equal-length edit with an unchanged suffix at fixed positions.
    def test_assesses_token_aligned_edit(self) -> None:
        assessment = assess_hybrid_token_alignment(
            [1, 2, 3, 4, 5, 6],
            [1, 2, 9, 4, 5, 6],
        )

        self.assertTrue(assessment["passed"])
        self.assertEqual(assessment["shared_prefix_tokens"], 2)
        self.assertEqual(assessment["shared_suffix_tokens"], 3)

    # Reject a text edit that shifted every later token position.
    def test_rejects_token_length_shift(self) -> None:
        assessment = assess_hybrid_token_alignment(
            [1, 2, 3, 4, 5],
            [1, 2, 8, 9, 4, 5],
        )

        self.assertFalse(assessment["passed"])
        self.assertFalse(assessment["same_token_count"])

    # Abort before generation when the live tokenizer reports shifted edits.
    def test_rejects_misaligned_workload_before_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "benchmarks.run_hybrid_apc_baseline.inspect_hybrid_token_alignment",
                    return_value={
                        "passed": False,
                        "details": {"early_edit": {"same_token_count": False}},
                    },
                ),
                patch(
                    "benchmarks.run_hybrid_apc_baseline._run_recorded_request"
                ) as run_request,
            ):
                with self.assertRaisesRegex(RuntimeError, "not token aligned"):
                    run_hybrid_apc_baseline(
                        base_url="http://example.test:8000",
                        model="test-model",
                        run_id="misaligned",
                        request_log_dir=root / "logs",
                        output=root / "summary.json",
                        require_token_aligned_edits=True,
                    )

            run_request.assert_not_called()

    # Require every candidate-layer pair to produce finite shadow divergence.
    def test_assesses_complete_gdn_delta_shadow_execution(self) -> None:
        rows = [
            {
                "scenario": scenario,
                "role": "target",
                "gdn_delta_reuse": {
                    "preflight_eligible": True,
                    "shadow_expected_count": 48,
                    "shadow_compared_count": 48,
                    "shadow_complete": True,
                    "shadow_max_output_relative_l2": 0.08,
                    "shadow_max_final_state_relative_l2": 0.03,
                },
            }
            for scenario in ("early_edit", "middle_edit")
        ]

        assessment = assess_gdn_delta_shadow_execution(rows)

        self.assertTrue(assessment["passed"])
        self.assertEqual(
            assessment["details"]["early_edit"]["completed_comparisons"], 48
        )

    # Require both edited scenarios to expose plan and preflight evidence.
    def test_assesses_gdn_delta_preflight_observability(self) -> None:
        rows = [
            {
                "scenario": scenario,
                "role": "target",
                "gdn_delta_reuse": {
                    "plan": {"reason": "aligned_candidates"},
                    "preflight_reason": "eligible",
                },
            }
            for scenario in ("early_edit", "middle_edit")
        ]

        assessment = assess_gdn_delta_preflight_observability(rows)

        self.assertTrue(assessment["passed"])

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
        self.assertEqual(
            len(scenarios["early_edit"].source_prompt),
            len(scenarios["early_edit"].target_prompt),
        )
        self.assertEqual(
            len(scenarios["middle_edit"].source_prompt),
            len(scenarios["middle_edit"].target_prompt),
        )
        self.assertIn("marker A", scenarios["early_edit"].source_prompt)
        self.assertIn("marker B", scenarios["early_edit"].target_prompt)

    # Distinguish block checkpoint reuse from the old all-or-nothing hybrid path.
    def test_accepts_progressively_longer_edited_prefix_hits(self) -> None:
        cached = {
            "exact": 2112,
            "append_only": 2112,
            "early_edit": 128,
            "middle_edit": 1216,
        }
        rows = [
            {"scenario": scenario, "role": "target", "cached_tokens": tokens}
            for scenario, tokens in cached.items()
        ]

        assessment = assess_hybrid_checkpoint_reuse(rows)

        self.assertTrue(assessment["passed"])
        self.assertEqual(assessment["cached_tokens"], cached)

    # Reject the previous align-mode behavior that missed both edited prefixes.
    def test_rejects_zero_edit_prefix_hits(self) -> None:
        rows = [
            {"scenario": scenario, "role": "target", "cached_tokens": tokens}
            for scenario, tokens in {
                "exact": 2112,
                "append_only": 2112,
                "early_edit": 0,
                "middle_edit": 0,
            }.items()
        ]

        self.assertFalse(assess_hybrid_checkpoint_reuse(rows)["passed"])

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
                patch(
                    "benchmarks.run_hybrid_apc_baseline.inspect_hybrid_token_alignment",
                    return_value={"passed": True, "details": {}},
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
            self.assertFalse(result["reference_validation_passed"])
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
                patch(
                    "benchmarks.run_hybrid_apc_baseline.inspect_hybrid_token_alignment",
                    return_value={"passed": True, "details": {}},
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
        self.assertTrue(result["reference_validation_passed"])


if __name__ == "__main__":
    unittest.main()
