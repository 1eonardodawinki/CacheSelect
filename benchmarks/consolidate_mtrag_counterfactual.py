"""Validate and combine completed or resumed MTRAG counterfactual runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from cacheselect.block_features import CandidateBlockFeatures


COUNT_FIELDS = (
    "trial_count",
    "valid_training_rows",
    "invalid_trials",
    "abstained_trials",
    "reference_drift_trials",
    "repair_labels",
    "reuse_labels",
)
TRAINING_COLUMNS = (
    "trace_id",
    "transition_id",
    "split",
    "decision",
    "label_source",
    "label_reason",
    *(field.name for field in fields(CandidateBlockFeatures)),
)
AUDIT_COLUMNS = (
    "mtrag_case_index",
    "mtrag_collection",
    "mtrag_conversation_id",
    "mtrag_current_task_id",
)
SUPPORTED_EXPERIMENTS = {
    "mtrag-counterfactual-pilot",
    "qwen3-reviewed-mtrag-counterfactual",
}


# Read one non-negative count without accepting booleans as integers.
def _count(record: dict[str, Any], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


# Load every completed case from one run, including an interrupted run without a summary.
def _load_run_cases(input_dir: Path) -> tuple[list[tuple[int, Path, dict]], int | None]:
    case_paths = sorted(
        input_dir.glob("case-*/summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("case-")),
    )
    if not case_paths:
        raise ValueError("MTRAG run contains no completed case summaries")
    local_indices = [int(path.parent.name.removeprefix("case-")) for path in case_paths]
    if local_indices != list(range(1, len(case_paths) + 1)):
        raise ValueError("MTRAG run has a gap in its completed local cases")

    summary_path = input_dir / "summary.json"
    run_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else None
    )
    if run_summary is not None:
        cases = run_summary.get("cases")
        if (
            run_summary.get("experiment") not in SUPPORTED_EXPERIMENTS
            or not isinstance(cases, list)
            or _count(run_summary, "case_count") != len(case_paths)
        ):
            raise ValueError("input is not a complete MTRAG counterfactual summary")
    else:
        cases = None

    start = run_summary.get("source_case_start", 1) if run_summary else 1
    if not isinstance(start, int) or isinstance(start, bool) or start < 1:
        raise ValueError("MTRAG run has an invalid source case start")
    loaded = []
    for local_index, path in enumerate(case_paths, start=1):
        case = json.loads(path.read_text(encoding="utf-8"))
        if cases is not None and case != cases[local_index - 1]:
            raise ValueError("top-level and per-case MTRAG summaries disagree")
        source_index = case.get("source_case_index", start + local_index - 1)
        if not isinstance(source_index, int) or isinstance(source_index, bool):
            raise ValueError("MTRAG case has an invalid source case index")
        loaded.append((source_index, path.parent, case))

    if run_summary is not None:
        totals = Counter()
        for _, _, case in loaded:
            totals.update({name: _count(case, name) for name in COUNT_FIELDS})
        if any(_count(run_summary, name) != totals[name] for name in COUNT_FIELDS):
            raise ValueError("MTRAG run totals do not match its cases")
    source_count = run_summary.get("source_case_count") if run_summary else None
    if source_count is not None and (
        not isinstance(source_count, int)
        or isinstance(source_count, bool)
        or source_count < 1
    ):
        raise ValueError("MTRAG run has an invalid source case count")
    return loaded, source_count


# Validate every case across one or more runs and merge its labelled training rows.
def consolidate_mtrag_counterfactual_results(
    input_dirs: Path | Sequence[Path],
    *,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    run_dirs = (input_dirs,) if isinstance(input_dirs, Path) else tuple(input_dirs)
    if not run_dirs:
        raise ValueError("at least one MTRAG run is required")
    indexed_cases: dict[int, tuple[Path, dict]] = {}
    source_counts = set()
    for run_dir in run_dirs:
        cases, source_count = _load_run_cases(run_dir)
        if source_count is not None:
            source_counts.add(source_count)
        for source_index, case_dir, case in cases:
            if source_index in indexed_cases:
                raise ValueError(f"duplicate MTRAG source case {source_index}")
            indexed_cases[source_index] = (case_dir, case)
    if len(source_counts) > 1:
        raise ValueError("MTRAG runs disagree on the source case count")
    expected_count = next(iter(source_counts), max(indexed_cases))
    expected_indices = set(range(1, expected_count + 1))
    missing = sorted(expected_indices - indexed_cases.keys())
    if missing:
        raise ValueError(f"MTRAG runs are missing source cases: {missing}")
    unexpected = sorted(indexed_cases.keys() - expected_indices)
    if unexpected:
        raise ValueError(f"MTRAG runs contain unexpected source cases: {unexpected}")

    totals: Counter[str] = Counter()
    merged: list[dict[str, str | int]] = []
    seen: set[tuple[str, str, str]] = set()
    skipped_indices = []
    for index in range(1, expected_count + 1):
        case_dir, case = indexed_cases[index]
        if not isinstance(case, dict):
            raise ValueError("MTRAG case summary must be an object")
        counts = {name: _count(case, name) for name in COUNT_FIELDS}
        if (
            counts["valid_training_rows"]
            + counts["invalid_trials"]
            + counts["abstained_trials"]
            != counts["trial_count"]
            or counts["valid_training_rows"]
            != counts["repair_labels"] + counts["reuse_labels"]
        ):
            raise ValueError("MTRAG case trial counts are inconsistent")
        totals.update(counts)

        collection = case.get("collection")
        task_id = case.get("current_task_id")
        if not isinstance(collection, str) or not collection:
            raise ValueError("MTRAG case has no collection")
        if not isinstance(task_id, str) or "<::>" not in task_id:
            raise ValueError("MTRAG case has no valid current task ID")
        status = case.get("status", "completed")
        if status == "skipped_reference_quality":
            if any(counts.values()):
                raise ValueError("skipped MTRAG case contains trial results")
            skipped_indices.append(index)
            continue
        if status != "completed":
            raise ValueError("MTRAG case has an unknown status")
        dataset = case_dir / "counterfactual-blocks.csv"
        with dataset.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != TRAINING_COLUMNS:
                raise ValueError("MTRAG case dataset has an unexpected schema")
            rows = list(reader)
        decisions = Counter(row["decision"] for row in rows)
        if (
            len(rows) != counts["valid_training_rows"]
            or decisions["repair"] != counts["repair_labels"]
            or decisions["reuse"] != counts["reuse_labels"]
        ):
            raise ValueError("MTRAG case dataset rows do not match its summary")

        for row in rows:
            if row["trace_id"] != case.get("trace_id") or row[
                "transition_id"
            ] != case.get("transition_id"):
                raise ValueError("MTRAG training row belongs to a different case")
            key = (row["trace_id"], row["transition_id"], row["candidate_block_index"])
            if key in seen:
                raise ValueError("MTRAG pilot contains a duplicate block label")
            seen.add(key)
            merged.append(
                {
                    **row,
                    "mtrag_case_index": index,
                    "mtrag_collection": collection,
                    "mtrag_conversation_id": task_id.split("<::>", 1)[0],
                    "mtrag_current_task_id": task_id,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[*TRAINING_COLUMNS, *AUDIT_COLUMNS])
        writer.writeheader()
        writer.writerows(merged)
    report = {
        "schema_version": 1,
        "analysis": "mtrag-counterfactual-consolidation",
        "source_runs": [str(path) for path in run_dirs],
        "output_dataset": str(output_path),
        "case_count": expected_count,
        "completed_case_count": expected_count - len(skipped_indices),
        "skipped_reference_case_count": len(skipped_indices),
        "skipped_reference_case_indices": skipped_indices,
        **totals,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# Parse paths and consolidate one or more copied or cluster-resident run directories.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if len(args.input_dir) > 1 and args.output_dir is None:
        parser.error("multiple inputs require --output-dir")
    output_dir = args.output_dir or args.input_dir[0]
    report = consolidate_mtrag_counterfactual_results(
        args.input_dir,
        output_path=output_dir / "mtrag-counterfactual-blocks.csv",
        report_path=output_dir / "consolidated-summary.json",
    )
    print(
        f"Consolidated {report['valid_training_rows']} labels from "
        f"{report['trial_count']} trials"
    )


if __name__ == "__main__":
    main()
