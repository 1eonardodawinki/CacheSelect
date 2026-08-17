"""Compare repeated uncached MTRAG reference runs for output stability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


# Load and validate one full-computation MTRAG reference artifact.
def _load_reference(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact.get("rows")
    if (
        artifact.get("analysis") != "mtrag-reference-quality-calibration"
        or not isinstance(rows, list)
        or artifact.get("request_count") != len(rows)
    ):
        raise ValueError(f"{path} is not a complete MTRAG reference artifact")
    return artifact


# Compare outputs for the same tasks under the same model configuration.
def compare_mtrag_reference_stability(
    baseline_path: Path,
    repeat_path: Path,
) -> dict[str, Any]:
    baseline = _load_reference(baseline_path)
    repeat = _load_reference(repeat_path)
    for field in ("model", "max_completion_tokens", "source_sha256", "manifest"):
        if baseline.get(field) != repeat.get(field):
            raise ValueError(f"reference runs disagree on {field}")

    # Index rows by task while rejecting duplicate or incomplete observations.
    def rows_by_task(artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        result = {}
        for row in artifact["rows"]:
            if not isinstance(row, Mapping):
                raise ValueError("reference row must be an object")
            task_id = row.get("task_id")
            output = row.get("output_text")
            if (
                not isinstance(task_id, str)
                or not task_id
                or task_id in result
                or not isinstance(output, str)
                or not output
            ):
                raise ValueError("reference row has invalid identity or output")
            result[task_id] = row
        return result

    baseline_rows = rows_by_task(baseline)
    repeat_rows = rows_by_task(repeat)
    if baseline_rows.keys() != repeat_rows.keys():
        raise ValueError("reference runs contain different task IDs")

    comparisons = []
    for task_id in sorted(baseline_rows):
        original = baseline_rows[task_id]
        repeated = repeat_rows[task_id]
        for field in (
            "split",
            "collection",
            "prompt_token_count",
            "expected_answer",
        ):
            if original.get(field) != repeated.get(field):
                raise ValueError(f"reference runs disagree on task {field}")
        original_output = original["output_text"]
        repeated_output = repeated["output_text"]
        comparisons.append(
            {
                "task_id": task_id,
                "split": original.get("split"),
                "collection": original.get("collection"),
                "exact_output_match": original_output == repeated_output,
                "baseline_output_sha256": hashlib.sha256(
                    original_output.encode()
                ).hexdigest(),
                "repeat_output_sha256": hashlib.sha256(
                    repeated_output.encode()
                ).hexdigest(),
            }
        )
    match_count = sum(row["exact_output_match"] for row in comparisons)
    return {
        "schema_version": 1,
        "analysis": "mtrag-reference-output-stability",
        "model": baseline.get("model"),
        "max_completion_tokens": baseline.get("max_completion_tokens"),
        "baseline_artifact": str(baseline_path),
        "repeat_artifact": str(repeat_path),
        "request_count": len(comparisons),
        "exact_match_count": match_count,
        "changed_count": len(comparisons) - match_count,
        "exact_match_rate": match_count / len(comparisons) if comparisons else 0.0,
        "comparisons": comparisons,
    }


# Parse two reference runs and save their task-by-task stability report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_mtrag_reference_stability(args.baseline, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Matched {report['exact_match_count']}/{report['request_count']} "
        "full-computation outputs"
    )


if __name__ == "__main__":
    main()
