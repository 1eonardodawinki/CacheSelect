import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.review_mtrag_with_openai import MODEL, _apply, _load_progress


class OpenAIMtragReviewTests(unittest.TestCase):
    def test_loads_resumable_progress_and_applies_complete_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            judgments = [
                {
                    "group_id": "group-1",
                    "verdict": "equivalent",
                    "reason": "Same facts.",
                    "model": MODEL,
                },
                {
                    "group_id": "group-2",
                    "verdict": "answer_b_better",
                    "reason": "B is complete.",
                    "model": MODEL,
                },
            ]
            progress.write_text("".join(json.dumps(row) + "\n" for row in judgments))
            loaded = _load_progress(progress, {"group-1", "group-2"})
            compact = {
                "rows": [
                    {"group_id": "group-1", "verdict": "", "reason": ""},
                    {"group_id": "group-2", "verdict": "", "reason": ""},
                ]
            }

            _apply(compact, loaded, MODEL)

            self.assertEqual(
                [row["verdict"] for row in compact["rows"]],
                ["equivalent", "answer_b_better"],
            )


if __name__ == "__main__":
    unittest.main()
