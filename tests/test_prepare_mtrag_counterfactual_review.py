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
            shared = _trial_events(
                "abstained-trial", "reference", "changed\u2028answer", 7
            )
            shared[0]["metadata"] = {
                "benchmark_request_id": "shared-reference",
                "counterfactual_role": "discovery_edit",
            }
            shared[2]["metadata"][
                "counterfactual_reference_request_id"
            ] = "shared-reference"
            shared.extend(
                [
                    {
                        "event": "request_started",
                        "request_id": "stability-reference",
                        "metadata": {
                            "counterfactual_role": "stability_reference",
                            "counterfactual_reference_request_id": "shared-reference",
                        },
                    },
                    {
                        "event": "request_completed",
                        "request_id": "stability-reference",
                        "output": {"text": "independent normal wording"},
                    },
                ]
            )
            events = [
                *_trial_events("exact-trial", "same", "same", 2),
                *shared,
            ]
            (log_dir / "requests.jsonl").write_text(
                "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
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
        self.assertEqual(
            {review["rows"][0]["answer_a"], review["rows"][0]["answer_b"]},
            {"reference", "changed\u2028answer"},
        )
        self.assertEqual(review["rows"][0]["verdict"], "")
        self.assertEqual(
            review["rows"][0]["independent_normal_answer"],
            "independent normal wording",
        )
        self.assertEqual(key["rows"][0]["trial_id"], "abstained-trial")
        self.assertEqual(key["rows"][0]["block_index"], 7)

    def test_excludes_wholly_rejected_reference_cases(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log_dir = root / "request-logs"
            log_dir.mkdir()
            invalid = _trial_events("invalid-trial", "bad", "changed", 1)
            invalid[0]["metadata"] = {
                "benchmark_request_id": "rejected:edited",
                "counterfactual_role": "discovery_edit",
            }
            invalid[2]["metadata"][
                "counterfactual_reference_request_id"
            ] = "rejected:edited"
            events = [
                *invalid,
                *_trial_events("valid-trial", "reference", "changed", 2),
            ]
            (log_dir / "requests.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "trial_count": 2,
                        "invalid_trials": 1,
                        "abstained_trials": 1,
                        "cases": [
                            {
                                "discovery_id": "rejected",
                                "trial_count": 1,
                                "invalid_trials": 1,
                            },
                            {
                                "discovery_id": "valid",
                                "trial_count": 1,
                                "invalid_trials": 0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = prepare_mtrag_counterfactual_review(root)
            key = json.loads((root / "manual-review-key.json").read_text())

        self.assertEqual(result["rows"], 1)
        self.assertEqual(key["rows"][0]["trial_id"], "valid-trial")
