import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.analyze_policy_reuse_sweep import analyze_policy_reuse_sweep


def _write_result(root: Path, target: float, selected: int) -> Path:
    result = root / str(target)
    result.mkdir()
    cases = [
        {
            "transition_id": transition,
            "valid_reference": True,
            "quality_passed": quality,
            "exact_output_match": exact,
            "requires_manual_review": not exact,
            "reference_ttft_ms": 3.0,
            "policy_ttft_ms": 2.0,
            "reference_wall_seconds": 2.0,
            "policy_wall_seconds": 1.0,
        }
        for transition, quality, exact in (("a", True, True), ("b", False, False))
    ]
    (result / "summary.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "candidate_tokens": 100,
                "selected_reuse_tokens": selected,
                "selected_reuse_share": selected / 100,
                "executed_cached_tokens": selected,
                "aggregate_ttft_speedup": 1.5,
            }
        ),
        encoding="utf-8",
    )
    (result / "mlp-model.json").write_text(
        json.dumps(
            {
                "selected_repair_threshold": target / 2,
                "reuse_budget": {"target_reuse_rate": target},
            }
        ),
        encoding="utf-8",
    )
    return result


class AnalyzePolicyReuseSweepTests(TestCase):
    def test_combines_matched_runs_in_budget_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = analyze_policy_reuse_sweep(
                [_write_result(root, 0.5, 40), _write_result(root, 0.1, 8)]
            )

        self.assertEqual(report["matched_case_count"], 2)
        self.assertEqual(
            [point["target_reuse_rate"] for point in report["points"]],
            [0.1, 0.5],
        )
        point = report["points"][0]
        self.assertEqual(point["semantic_error_rate"], 0.5)
        self.assertEqual(point["exact_match_rate"], 0.5)
        self.assertEqual(point["aggregate_wall_speedup"], 2.0)

    def test_rejects_unmatched_cases(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_result(root, 0.1, 8)
            second = _write_result(root, 0.5, 40)
            summary_path = second / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["cases"][1]["transition_id"] = "different"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "same transitions"):
                analyze_policy_reuse_sweep([first, second])

    def test_compares_cache_policy_with_native_apc(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _write_result(root, 0.1, 8)
            baseline = _write_result(root, 0.0, 0)
            report = analyze_policy_reuse_sweep([policy], baseline)

        self.assertEqual(report["native_apc"]["semantic_error_rate"], 0.5)
        self.assertEqual(report["points"][0]["ttft_speedup_vs_native_apc"], 1.0)
        self.assertEqual(report["points"][0]["wall_speedup_vs_native_apc"], 1.0)
