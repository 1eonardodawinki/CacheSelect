"""Summarize active Qwen hybrid GDN recurrence reuse against full references."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


# Return one uniquely identified observation from a hybrid benchmark summary.
def _observation(
    rows: list[dict[str, Any]],
    scenario: str,
    role: str,
) -> dict[str, Any]:
    """Resolve exactly one scenario/role row or reject an ambiguous artifact."""
    matches = [
        row
        for row in rows
        if row.get("scenario") == scenario and row.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {scenario}/{role} observation")
    return matches[0]


# Convert one completed active smoke summary into analysis-ready comparisons.
def analyze_hybrid_gdn_active(summary: dict[str, Any]) -> dict[str, Any]:
    """Compare active edited requests with their uncached reference requests."""
    active_gate = summary.get("gdn_delta_active")
    if not isinstance(active_gate, dict) or active_gate.get("passed") is not True:
        raise ValueError("hybrid summary does not contain passing active execution")
    rows = summary.get("observations")
    if not isinstance(rows, list):
        raise ValueError("hybrid summary observations are missing")
    record_count = summary.get("record_count")
    if not isinstance(record_count, int) or record_count < 20:
        raise ValueError("hybrid summary has an invalid record_count")

    comparisons = []
    for scenario in ("early_edit", "middle_edit"):
        reference = _observation(rows, scenario, "reference")
        target = _observation(rows, scenario, "target")
        metrics = target.get("gdn_delta_reuse")
        if not isinstance(metrics, dict) or metrics.get("active_complete") is not True:
            raise ValueError(f"{scenario} has incomplete active GDN evidence")
        reference_seconds = float(reference["client_wall_seconds"])
        active_seconds = float(target["client_wall_seconds"])
        if reference_seconds <= 0 or active_seconds <= 0:
            raise ValueError("request durations must be positive")
        reference_ttft = reference.get("time_to_first_token_ms")
        active_ttft = target.get("time_to_first_token_ms")
        if (
            not isinstance(reference_ttft, (int, float))
            or isinstance(reference_ttft, bool)
            or reference_ttft <= 0
            or not isinstance(active_ttft, (int, float))
            or isinstance(active_ttft, bool)
            or active_ttft <= 0
        ):
            raise ValueError("time-to-first-token measurements must be positive")
        comparisons.append(
            {
                "scenario": scenario,
                "exact_output_match": (
                    reference.get("output_text") == target.get("output_text")
                ),
                "reference_seconds": reference_seconds,
                "active_seconds": active_seconds,
                "speedup": reference_seconds / active_seconds,
                "reference_ttft_ms": float(reference_ttft),
                "active_ttft_ms": float(active_ttft),
                "ttft_speedup": float(reference_ttft) / float(active_ttft),
                "native_cached_tokens": int(target["cached_tokens"]),
                "candidate_blocks": int(metrics["candidate_block_count"]),
                "executed_layers": int(metrics["active_executed_layer_count"]),
                "reused_layer_tokens": int(metrics["active_reused_layer_tokens"]),
                "recomputed_layer_tokens": int(
                    metrics["active_recomputed_layer_tokens"]
                ),
            }
        )

    speedups = [row["speedup"] for row in comparisons]
    ttft_speedups = [row["ttft_speedup"] for row in comparisons]
    return {
        "schema_version": 1,
        "experiment": "hybrid_gdn_active_analysis",
        "source_run_id": summary.get("run_id"),
        "model": summary.get("model"),
        "record_count": record_count,
        "all_outputs_exact": all(row["exact_output_match"] for row in comparisons),
        "mean_speedup": statistics.fmean(speedups),
        "minimum_speedup": min(speedups),
        "mean_ttft_speedup": statistics.fmean(ttft_speedups),
        "minimum_ttft_speedup": min(ttft_speedups),
        "comparisons": comparisons,
    }


# Parse paths for the standalone analysis command.
def _parse_args() -> argparse.Namespace:
    """Return command-line arguments for one active smoke artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


# Load, analyze and save one active smoke result.
def main() -> None:
    """Write a compact JSON artifact beside the raw benchmark summary."""
    args = _parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze_hybrid_gdn_active(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
