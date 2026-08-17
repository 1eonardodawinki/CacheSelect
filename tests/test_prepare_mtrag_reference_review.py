import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.prepare_mtrag_reference_review import (
    prepare_mtrag_reference_review,
)


# Write one small reference artifact for a requested split.
def _artifact(path: Path, split: str, task_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "analysis": "mtrag-reference-quality-calibration",
                "rows": [
                    {
                        "task_id": task_id,
                        "split": split,
                        "collection": "RAG",
                        "output_text": "Model answer",
                        "expected_answer": "Expected answer",
                        "quality": {
                            "metrics": {"token_recall": 0.5, "rouge_l_f1": 0.4}
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class PrepareMtragReferenceReviewTests(TestCase):
    # Hide task and split identities while retaining them in a separate key.
    def test_prepares_blinded_review_and_key(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = root / "train.json"
            validation = root / "validation.json"
            _artifact(train, "train", "task-train")
            _artifact(validation, "validation", "task-validation")

            review, key = prepare_mtrag_reference_review((train, validation))

        self.assertEqual(review["row_count"], 2)
        self.assertEqual(key["row_count"], 2)
        self.assertNotIn("task_id", review["rows"][0])
        self.assertNotIn("split", review["rows"][0])
        self.assertTrue(all(row["verdict"] == "" for row in review["rows"]))
        self.assertEqual({row["split"] for row in key["rows"]}, {
            "train",
            "validation",
        })
