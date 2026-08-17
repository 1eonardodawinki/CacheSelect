import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from benchmarks.consolidate_mtrag_counterfactual import AUDIT_COLUMNS, TRAINING_COLUMNS
from benchmarks.curate_mtrag_counterfactual_dataset import (
    curate_mtrag_counterfactual_dataset,
)
from cacheselect.block_features import CandidateBlockFeatures


# Write the compact result artifacts needed to merge one automatic and one review row.
def _write_result_fixture(root: Path) -> None:
    log_dir = root / "request-logs"
    log_dir.mkdir()
    events = []
    for trial_id, block in (("exact-trial", 7), ("review-trial", 8)):
        request_id = f"{trial_id}:intervention"
        events.extend(
            [
                {
                    "event": "request_started",
                    "request_id": request_id,
                    "metadata": {
                        "counterfactual_trial_id": trial_id,
                        "counterfactual_role": "intervention",
                        "counterfactual_reuse_block_index": block,
                    },
                },
                {
                    "event": "request_completed",
                    "request_id": request_id,
                    "output": {"text": "answer"},
                    "metrics": {
                        "server_metrics": {
                            "cacheselect_partial_reuse_plan": {
                                "transition_id": "transition-1"
                            }
                        }
                    },
                },
            ]
        )
    ledger = log_dir / "requests.jsonl"
    ledger.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )

    review_path = root / "manual-review-blinded.json"
    key_path = root / "manual-review-key.json"
    review_path.write_text('{"frozen":true}\n', encoding="utf-8")
    key_path.write_text('{"sealed":true}\n', encoding="utf-8")
    audit = {
        "source_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "completed_blind_review_sha256": hashlib.sha256(
            review_path.read_bytes()
        ).hexdigest(),
        "identity_key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
        "reviewed_trials": 1,
        "abstentions": 0,
        "rows": [
            {
                "review_id": "review-1",
                "trial_id": "review-trial",
                "block_index": 8,
                "blind_verdict": "answer_a_better",
                "decision": "repair",
                "reason": "reference retained an important detail",
            }
        ],
    }
    (root / "manual-review-unblinded.json").write_text(json.dumps(audit))
    case = {
        "trace_id": "trace-1",
        "transition_id": "transition-1",
        "split": "train",
        "collection": "collection-1",
        "current_task_id": "conversation-1<::>2",
    }
    (root / "summary.json").write_text(
        json.dumps({"trial_count": 2, "cases": [case]}), encoding="utf-8"
    )
    row = {name: "0" for name in (*TRAINING_COLUMNS, *AUDIT_COLUMNS)}
    row.update(
        trace_id="trace-1",
        transition_id="transition-1",
        split="train",
        decision="reuse",
        candidate_block_index="7",
        mtrag_case_index="1",
    )
    with (root / "mtrag-counterfactual-blocks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=(*TRAINING_COLUMNS, *AUDIT_COLUMNS))
        writer.writeheader()
        writer.writerow(row)


class CurateMtragCounterfactualDatasetTests(TestCase):
    # Merge exact and manually reviewed labels without losing their provenance.
    def test_curates_all_pilot_trials(self):
        feature = {field.name: 0 for field in fields(CandidateBlockFeatures)}
        feature["candidate_block_index"] = 8
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_result_fixture(root)
            with patch(
                "benchmarks.curate_mtrag_counterfactual_dataset._reviewed_feature",
                return_value=(feature, "transition-1"),
            ):
                report = curate_mtrag_counterfactual_dataset(root)
            with (root / "mtrag-curated-blocks.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["reuse_labels"], 1)
        self.assertEqual(report["repair_labels"], 1)
        self.assertEqual(rows[0]["adjudication"], "exact_output_match")
        self.assertEqual(rows[1]["adjudication"], "blinded_semantic_review")
        self.assertEqual(rows[1]["trial_id"], "review-trial")

    # Preserve an unclear review in the audit without creating a training label.
    def test_excludes_abstained_review(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_result_fixture(root)
            audit_path = root / "manual-review-unblinded.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["abstentions"] = 1
            audit["rows"][0]["decision"] = "abstain"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            report = curate_mtrag_counterfactual_dataset(root)
            with (root / "mtrag-curated-blocks.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(report["source_trial_count"], 2)
        self.assertEqual(report["row_count"], 1)
        self.assertEqual(report["abstained_trials"], 1)
        self.assertEqual(rows[0]["trial_id"], "exact-trial")
