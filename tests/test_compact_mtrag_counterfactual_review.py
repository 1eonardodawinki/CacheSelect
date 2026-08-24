import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.compact_mtrag_counterfactual_review import (
    apply_compact_review,
    prepare_compact_review,
)


class CompactMtragReviewTests(unittest.TestCase):
    def test_compacts_and_expands_swapped_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "review_id": f"review-{index}",
                    "question": "Q",
                    "expected_answer": "E",
                    "answer_a": answers[0],
                    "answer_b": answers[1],
                    "independent_normal_answer": None,
                    "verdict": "",
                    "reason": "",
                }
                for index, answers in enumerate(
                    (("Reference", "Candidate"), ("Candidate", "Reference")), 1
                )
            ]
            path = root / "manual-review-blinded.json"
            path.write_text(json.dumps({"rubric": "Compare", "rows": rows}))

            compact = prepare_compact_review(root)
            self.assertEqual((compact["trial_count"], compact["group_count"]), (2, 1))
            compact["rows"][0].update(verdict="answer_a_better", reason="Better")
            (root / "manual-review-compact.json").write_text(json.dumps(compact))

            counts = apply_compact_review(root)
            completed = json.loads(path.read_text())["rows"]
            self.assertEqual(
                [row["verdict"] for row in completed],
                ["answer_b_better", "answer_a_better"],
            )
            self.assertEqual(sum(counts.values()), 2)


if __name__ == "__main__":
    unittest.main()
