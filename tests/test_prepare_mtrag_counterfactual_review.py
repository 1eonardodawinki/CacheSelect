import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.prepare_mtrag_counterfactual_review import (
    prepare_mtrag_counterfactual_review,
)


# Construct one complete trial lifecycle for a compact ledger fixture.
def _trial_events(trial_id: str, reference: str, intervention: str, block: int):
    evaluation = {
        "ground_truth": {"expected_answer": "expected"},
        "prompt_segments": [{"kind": "query", "content": "question"}],
    }
    events = []
    for role, text in (("reference", reference), ("intervention", intervention)):
        request_id = f"{trial_id}:{role}"
        events.extend(
            [
                {
                    "event": "request_started",
                    "request_id": request_id,
                    "metadata": {
                        "counterfactual_trial_id": trial_id,
                        "counterfactual_role": role,
                        "counterfactual_reuse_block_index": block,
                    },
                    "evaluation": evaluation,
                },
                {
                    "event": "request_completed",
                    "request_id": request_id,
                    "output": {"text": text},
                },
            ]
        )
    return events


class PrepareMtragCounterfactualReviewTests(TestCase):
    # Hide identities and answer roles while selecting only the non-exact trial.
    def test_prepares_blinded_review_and_sealed_key(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log_dir = root / "request-logs"
            log_dir.mkdir()
            events = [
                *_trial_events("exact-trial", "same", "same", 2),
                *_trial_events("abstained-trial", "reference", "changed", 7),
            ]
            (log_dir / "requests.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps({"trial_count": 2, "invalid_trials": 0, "abstained_trials": 1}),
                encoding="utf-8",
            )

            result = prepare_mtrag_counterfactual_review(root)
            review_text = (root / "manual-review-blinded.json").read_text()
            review = json.loads(review_text)
            key = json.loads((root / "manual-review-key.json").read_text())

        self.assertEqual(result["rows"], 1)
        self.assertNotIn("abstained-trial", review_text)
        self.assertNotIn('"block_index"', review_text)
        self.assertEqual({review["rows"][0]["answer_a"], review["rows"][0]["answer_b"]}, {"reference", "changed"})
        self.assertEqual(review["rows"][0]["verdict"], "")
        self.assertEqual(key["rows"][0]["trial_id"], "abstained-trial")
        self.assertEqual(key["rows"][0]["block_index"], 7)
