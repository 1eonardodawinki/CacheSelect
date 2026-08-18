"""Summarize Qwen hybrid GDN affine shadow divergence by layer and block."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# Validate one finite, nonnegative divergence measurement.
def _read_error(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"invalid {field} in GDN shadow result")
    return float(value)


# Reduce comparable layer/block records without imposing a safety threshold.
def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty GDN shadow record set")
    output_errors = [_read_error(record, "output_relative_l2") for record in records]
    state_errors = [
        _read_error(record, "final_state_relative_l2") for record in records
    ]
    return {
        "comparison_count": len(records),
        "median_output_relative_l2": statistics.median(output_errors),
        "max_output_relative_l2": max(output_errors),
        "median_final_state_relative_l2": statistics.median(state_errors),
        "max_final_state_relative_l2": max(state_errors),
    }


# Convert one completed hybrid smoke summary into analysis-ready aggregates.
def analyze_hybrid_gdn_shadow(summary: dict[str, Any]) -> dict[str, Any]:
    shadow_gate = summary.get("gdn_delta_shadow")
    if not isinstance(shadow_gate, dict) or shadow_gate.get("passed") is not True:
        raise ValueError("hybrid summary does not contain a passing shadow run")

    observations = summary.get("observations")
    if not isinstance(observations, list):
        raise ValueError("hybrid summary observations are missing")
    edited_targets = [
        row
        for row in observations
        if row.get("role") == "target"
        and row.get("scenario") in {"early_edit", "middle_edit"}
    ]
    if {row.get("scenario") for row in edited_targets} != {
        "early_edit",
        "middle_edit",
    }:
        raise ValueError("both edited target scenarios are required")

    records: list[dict[str, Any]] = []
    for row in edited_targets:
        metrics = row.get("gdn_delta_reuse")
        if not isinstance(metrics, dict) or metrics.get("shadow_complete") is not True:
            raise ValueError("edited target has incomplete GDN shadow evidence")
        expected = metrics.get("shadow_expected_count")
        raw_results = metrics.get("shadow_results")
        if not isinstance(expected, int) or expected <= 0:
            raise ValueError("edited target has no expected shadow comparisons")
        if not isinstance(raw_results, list) or len(raw_results) != expected:
            raise ValueError("shadow result count differs from the expected count")
        for raw_result in raw_results:
            if (
                not isinstance(raw_result, dict)
                or raw_result.get("reason") != "compared"
            ):
                raise ValueError("shadow result was not successfully compared")
            record = dict(raw_result)
            record["scenario"] = row["scenario"]
            records.append(record)

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_block: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scenario = record["scenario"]
        layer_name = record.get("layer_name")
        block_index = record.get("target_block_index")
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("shadow result has no layer name")
        if (
            isinstance(block_index, bool)
            or not isinstance(block_index, int)
            or block_index < 0
        ):
            raise ValueError("shadow result has an invalid target block")
        by_scenario[scenario].append(record)
        by_layer[layer_name].append(record)
        by_block[(scenario, block_index)].append(record)

    return {
        "schema_version": 1,
        "experiment": "hybrid_gdn_shadow_analysis",
        "source_run_id": summary.get("run_id"),
        "model": summary.get("model"),
        "overall": _summarize_records(records),
        "by_scenario": {
            name: _summarize_records(items) for name, items in by_scenario.items()
        },
        "by_layer": [
            {"layer_name": name, **_summarize_records(items)}
            for name, items in by_layer.items()
        ],
        "by_block": [
            {
                "scenario": scenario,
                "target_block_index": block_index,
                **_summarize_records(items),
            }
            for (scenario, block_index), items in by_block.items()
        ],
    }


# Parse the standalone analysis command line.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


# Analyze one saved GPU summary and persist a compact JSON artifact.
def main() -> None:
    args = _parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze_hybrid_gdn_shadow(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {analysis['overall']['comparison_count']} GDN comparisons")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
