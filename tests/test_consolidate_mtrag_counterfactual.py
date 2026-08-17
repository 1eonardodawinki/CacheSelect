import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.consolidate_mtrag_counterfactual import (
    TRAINING_COLUMNS,
    consolidate_mtrag_counterfactual_results,
)


# Write one case summary and its schema-complete training table.
def _write_case(root: Path, decision: str | None) -> dict[str, object]:
    case = {
        "collection": "collection-1",
        "current_task_id": "conversation-1<::>2",
        "trace_id": "trace-1",
        "transition_id": "transition-1",
        "trial_count": 1,
        "valid_training_rows": int(decision is not None),
        "invalid_trials": 0,
        "abstained_trials": int(decision is None),
        "reference_drift_trials": 0,
        "repair_labels": int(decision == "repair"),
        "reuse_labels": int(decision == "reuse"),
    }
    case_dir = root / "case-01"
    case_dir.mkdir()
    with (case_dir / "counterfactual-blocks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=TRAINING_COLUMNS)
        writer.writeheader()
        if decision:
            row = {name: "0" for name in TRAINING_COLUMNS}
            row.update(
                trace_id="trace-1",
                transition_id="transition-1",
                decision=decision,
                candidate_block_index="7",
            )
            writer.writerow(row)
    return case


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
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trial counts"):
                consolidate_mtrag_counterfactual_results(
                    root,
                    output_path=root / "combined.csv",
                    report_path=root / "report.json",
                )
