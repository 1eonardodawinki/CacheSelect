"""Combine matched end-to-end policy runs across frozen reuse budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _summarize(summary: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"incomplete policy result: {result_dir}")
    valid = [case for case in cases if case["valid_reference"]]
    valid_count = len(valid)
    reference_ttft = sum(float(case["reference_ttft_ms"]) for case in valid)
    policy_ttft = sum(float(case["policy_ttft_ms"]) for case in valid)
    reference_wall = sum(float(case["reference_wall_seconds"]) for case in valid)
    policy_wall = sum(float(case["policy_wall_seconds"]) for case in valid)
    quality_passes = sum(bool(case["quality_passed"]) for case in valid)
    exact_matches = sum(bool(case["exact_output_match"]) for case in valid)
    return {
        "case_ids": tuple(sorted(str(case["transition_id"]) for case in cases)),
        "case_count": len(cases),
        "valid_reference_count": valid_count,
        "semantic_error_count": valid_count - quality_passes,
        "semantic_error_rate": (
            (valid_count - quality_passes) / valid_count if valid_count else None
        ),
        "exact_match_rate": exact_matches / valid_count if valid_count else None,
        "manual_review_cases": sum(
            bool(case["requires_manual_review"]) for case in cases
        ),
        "candidate_tokens": int(summary["candidate_tokens"]),
        "selected_reuse_tokens": int(summary["selected_reuse_tokens"]),
        "selected_reuse_share": float(summary["selected_reuse_share"]),
        "executed_cached_tokens": int(summary["executed_cached_tokens"]),
        "aggregate_ttft_speedup": (
            reference_ttft / policy_ttft if policy_ttft else None
        ),
        "aggregate_wall_speedup": (
            reference_wall / policy_wall if policy_wall else None
        ),
        "policy_ttft_ms": policy_ttft,
        "policy_wall_seconds": policy_wall,
        "result_dir": str(result_dir),
    }


def analyze_policy_reuse_sweep(
    result_dirs: list[Path],
    baseline_result_dir: Path | None = None,
) -> dict[str, Any]:
    if not result_dirs:
        raise ValueError("provide at least one policy result directory")
    expected_cases = None
    points = []
    for result_dir in result_dirs:
        summary = _read(result_dir / "summary.json")
        model = _read(result_dir / "mlp-model.json")
        budget = model.get("reuse_budget") or {}
        target = budget.get("target_reuse_rate")
        if not isinstance(target, (int, float)):
            raise ValueError(f"incomplete policy result: {result_dir}")
        point = _summarize(summary, result_dir)
        case_ids = point.pop("case_ids")
        if expected_cases is None:
            expected_cases = case_ids
        elif case_ids != expected_cases:
            raise ValueError("policy runs do not contain the same transitions")
        points.append(
            {
                "target_reuse_rate": float(target),
                "repair_threshold": float(model["selected_repair_threshold"]),
                **point,
            }
        )
    points.sort(key=lambda point: point["target_reuse_rate"])
    targets = [point["target_reuse_rate"] for point in points]
    if len(targets) != len(set(targets)):
        raise ValueError("reuse targets must be unique")
    baseline = None
    if baseline_result_dir is not None:
        baseline = _summarize(
            _read(baseline_result_dir / "summary.json"), baseline_result_dir
        )
        if baseline.pop("case_ids") != expected_cases:
            raise ValueError("native APC and CacheSelect runs do not match")
        for point in points:
            point["ttft_speedup_vs_native_apc"] = (
                baseline["policy_ttft_ms"] / point["policy_ttft_ms"]
                if point["policy_ttft_ms"]
                else None
            )
            point["wall_speedup_vs_native_apc"] = (
                baseline["policy_wall_seconds"] / point["policy_wall_seconds"]
                if point["policy_wall_seconds"]
                else None
            )
    return {
        "schema_version": 1,
        "analysis": "end-to-end-reuse-budget-sweep",
        "matched_case_count": len(expected_cases or ()),
        "native_apc": baseline,
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", action="append", type=Path, required=True)
    parser.add_argument("--baseline-result-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_policy_reuse_sweep(
        args.result_dir, baseline_result_dir=args.baseline_result_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for point in report["points"]:
        print(
            f"reuse={point['target_reuse_rate']:.0%} "
            f"actual={point['selected_reuse_share']:.1%} "
            f"error={point['semantic_error_rate']:.1%} "
            f"ttft={point['aggregate_ttft_speedup']:.3f}x"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
