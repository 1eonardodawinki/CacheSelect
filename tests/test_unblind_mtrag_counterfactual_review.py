import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.unblind_mtrag_counterfactual_review import (
    unblind_mtrag_counterfactual_review,
)


# Write matching review, key, and ledger fixtures with representative verdicts.
def _write_review_fixture(root: Path) -> None:
    log_dir = root / "request-logs"
    log_dir.mkdir()
    ledger = log_dir / "requests.jsonl"
    ledger.write_text('{"event":"request_completed"}\n', encoding="utf-8")
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    rows = [
        ("review-1", "equivalent", "answer_a"),
        ("review-2", "answer_a_better", "answer_b"),
        ("review-3", "answer_b_better", "answer_b"),
        ("review-4", "unclear", "answer_a"),
    ]
    review = {
        "source_ledger_sha256": digest,
        "rows": [
            {"review_id": review_id, "verdict": verdict, "reason": "reviewed"}
            for review_id, verdict, _ in rows
        ],
    }
    key = {
        "source_ledger_sha256": digest,
        "rows": [
            {
                "review_id": review_id,
                "trial_id": f"trial-{index}",
                "block_index": index,
                "reference_slot": "answer_b" if intervention == "answer_a" else "answer_a",
                "intervention_slot": intervention,
            }
            for index, (review_id, _, intervention) in enumerate(rows, start=1)
        ],
    }
    (root / "manual-review-blinded.json").write_text(json.dumps(review))
    (root / "manual-review-key.json").write_text(json.dumps(key))


class UnblindMtragCounterfactualReviewTests(TestCase):
    # Map blind comparisons to causal decisions and preserve the frozen hashes.
    def test_unblinds_completed_review(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_review_fixture(root)
            audit = unblind_mtrag_counterfactual_review(root)
            saved = json.loads((root / "manual-review-unblinded.json").read_text())

        self.assertEqual(
            [row["decision"] for row in audit["rows"]],
            ["reuse", "repair", "reuse", "abstain"],
        )
        self.assertEqual(audit["reuse_decisions"], 2)
        self.assertEqual(audit["repair_decisions"], 1)
        self.assertEqual(audit["abstentions"], 1)
        self.assertEqual(saved, audit)

    # Refuse to unblind a row whose reviewer has not made a decision.
    def test_rejects_incomplete_review(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_review_fixture(root)
            review_path = root / "manual-review-blinded.json"
            review = json.loads(review_path.read_text())
            review["rows"][0]["verdict"] = ""
            review_path.write_text(json.dumps(review))

            with self.assertRaisesRegex(ValueError, "incomplete"):
                unblind_mtrag_counterfactual_review(root)
