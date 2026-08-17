"""Validate and combine one completed MTRAG counterfactual pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


# Read one non-negative count without accepting booleans as integers.
def _count(record: dict[str, Any], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


# Validate every case and merge only its already-labelled training rows.
def consolidate_mtrag_counterfactual_results(
    input_dir: Path,
    *,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    source_summary = input_dir / "summary.json"
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    cases = summary.get("cases")
    if (
        summary.get("experiment") != "mtrag-counterfactual-pilot"
        or not isinstance(cases, list)
        or not cases
        or _count(summary, "case_count") != len(cases)
    ):
        raise ValueError("input is not a complete MTRAG counterfactual summary")

    totals: Counter[str] = Counter()
    merged: list[dict[str, str | int]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, case in enumerate(cases, start=1):
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
        dataset = input_dir / f"case-{index:02d}" / "counterfactual-blocks.csv"
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

    if any(_count(summary, name) != totals[name] for name in COUNT_FIELDS):
        raise ValueError("MTRAG pilot totals do not match its cases")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[*TRAINING_COLUMNS, *AUDIT_COLUMNS])
        writer.writeheader()
        writer.writerows(merged)
    report = {
        "schema_version": 1,
        "analysis": "mtrag-counterfactual-consolidation",
        "source_summary": str(source_summary),
        "output_dataset": str(output_path),
        "case_count": len(cases),
        **totals,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# Parse paths and consolidate one copied or cluster-resident pilot directory.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report = consolidate_mtrag_counterfactual_results(
        args.input_dir,
        output_path=args.input_dir / "mtrag-counterfactual-blocks.csv",
        report_path=args.input_dir / "consolidated-summary.json",
    )
    print(
        f"Consolidated {report['valid_training_rows']} labels from "
        f"{report['trial_count']} trials"
    )


if __name__ == "__main__":
    main()
