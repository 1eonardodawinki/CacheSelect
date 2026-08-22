import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.consolidate_mtrag_counterfactual import (
    TRAINING_COLUMNS,
    assemble_mtrag_counterfactual_results,
    consolidate_mtrag_counterfactual_results,
)


# Write one case summary and its schema-complete training table.
def _write_case(
    root: Path,
    decision: str | None,
    *,
    local_index: int = 1,
    source_index: int | None = None,
    status: str | None = None,
) -> dict[str, object]:
    suffix = source_index or local_index
    case = {
        "collection": "collection-1",
        "current_task_id": f"conversation-{suffix}<::>2",
        "trace_id": f"trace-{suffix}",
        "transition_id": f"transition-{suffix}",
        "trial_count": 1,
        "valid_training_rows": int(decision is not None),
        "invalid_trials": 0,
        "abstained_trials": int(decision is None),
        "reference_drift_trials": 0,
        "repair_labels": int(decision == "repair"),
        "reuse_labels": int(decision == "reuse"),
    }
    if source_index is not None:
        case["source_case_index"] = source_index
    if status is not None:
        case["status"] = status
    case_dir = root / f"case-{local_index:02d}"
    case_dir.mkdir()
    if status == "skipped_reference_quality":
        case.update(
            trial_count=0,
            valid_training_rows=0,
            abstained_trials=0,
            repair_labels=0,
            reuse_labels=0,
        )
        (case_dir / "summary.json").write_text(json.dumps(case), encoding="utf-8")
        return case
    with (case_dir / "counterfactual-blocks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=TRAINING_COLUMNS)
        writer.writeheader()
        if decision:
            row = {name: "0" for name in TRAINING_COLUMNS}
            row.update(
                trace_id=f"trace-{suffix}",
                transition_id=f"transition-{suffix}",
                decision=decision,
                candidate_block_index="7",
            )
            writer.writerow(row)
    (case_dir / "summary.json").write_text(json.dumps(case), encoding="utf-8")
    return case


# Write complete request lifecycles for one exact-output block trial.
def _write_ledger(root: Path, trial_id: str) -> None:
    log_dir = root / "request-logs"
    log_dir.mkdir()
    events = []
    for role in ("reference", "donor", "intervention"):
        request_id = f"{trial_id}:{role}"
        events.extend(
            [
                {"event": "request_started", "request_id": request_id},
                {
                    "event": "request_completed",
                    "request_id": request_id,
                    "output": {"text": "same"},
                },
            ]
        )
    (log_dir / "requests.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )


class ConsolidateMtragCounterfactualTests(TestCase):
    # Merge valid labels and reject an inconsistent pilot summary.
    def test_consolidates_and_validates_pilot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            case = _write_case(root, "reuse")
            summary = {
                "experiment": "mtrag-counterfactual-pilot",
                "case_count": 1,
                **{name: case[name] for name in (
                    "trial_count",
                    "valid_training_rows",
                    "invalid_trials",
                    "abstained_trials",
                    "reference_drift_trials",
                    "repair_labels",
                    "reuse_labels",
                )},
                "cases": [case],
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            report = consolidate_mtrag_counterfactual_results(
                root,
                output_path=root / "combined.csv",
                report_path=root / "report.json",
            )
            with (root / "combined.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))

            self.assertEqual(report["valid_training_rows"], 1)
            self.assertEqual(rows[0]["mtrag_conversation_id"], "conversation-1")
            case["valid_training_rows"] = 2
            summary["valid_training_rows"] = 2
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (root / "case-01" / "summary.json").write_text(
                json.dumps(case), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "trial counts"):
                consolidate_mtrag_counterfactual_results(
                    root,
                    output_path=root / "combined.csv",
                    report_path=root / "report.json",
                )

    # Recover an interrupted prefix and merge it with a resumed suffix and skip.
    def test_merges_resumed_runs_by_source_case_index(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "prefix"
            resumed = root / "resumed"
            prefix.mkdir()
            resumed.mkdir()
            _write_case(prefix, "reuse")
            skipped = _write_case(
                resumed,
                None,
                local_index=1,
                source_index=2,
                status="skipped_reference_quality",
            )
            completed = _write_case(
                resumed, "repair", local_index=2, source_index=3, status="completed"
            )
            resumed_summary = {
                "experiment": "qwen3-reviewed-mtrag-counterfactual",
                "case_count": 2,
                "source_case_start": 2,
                "source_case_count": 3,
                **{
                    name: skipped[name] + completed[name]
                    for name in (
                        "trial_count",
                        "valid_training_rows",
                        "invalid_trials",
                        "abstained_trials",
                        "reference_drift_trials",
                        "repair_labels",
                        "reuse_labels",
                    )
                },
                "cases": [skipped, completed],
            }
            (resumed / "summary.json").write_text(
                json.dumps(resumed_summary), encoding="utf-8"
            )

            report = consolidate_mtrag_counterfactual_results(
                (prefix, resumed),
                output_path=root / "combined.csv",
                report_path=root / "report.json",
            )
            with (root / "combined.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))

            self.assertEqual([row["mtrag_case_index"] for row in rows], ["1", "3"])
            self.assertEqual(report["skipped_reference_case_indices"], [2])
            self.assertEqual(report["case_count"], 3)

            _write_ledger(prefix, "trial-1")
            _write_ledger(resumed, "trial-3")
            assembled = root / "assembled"
            assembled_report = assemble_mtrag_counterfactual_results(
                (prefix, resumed), assembled
            )
            assembled_summary = json.loads(
                (assembled / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(assembled_report["recorded_requests"], 6)
            self.assertEqual(assembled_summary["case_count"], 3)
            self.assertEqual(
                assembled_summary["cases"][1]["status"],
                "skipped_reference_quality",
            )

            completed["source_case_index"] = 4
            (resumed / "case-02" / "summary.json").write_text(
                json.dumps(completed), encoding="utf-8"
            )
            resumed_summary["cases"][1] = completed
            (resumed / "summary.json").write_text(
                json.dumps(resumed_summary), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing source cases"):
                consolidate_mtrag_counterfactual_results(
                    (prefix, resumed),
                    output_path=root / "combined.csv",
                    report_path=root / "report.json",
                )
