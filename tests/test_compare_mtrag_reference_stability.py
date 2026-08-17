import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.compare_mtrag_reference_stability import (
    compare_mtrag_reference_stability,
)


# Write a minimal full-computation reference artifact for stability tests.
def _write_reference(path: Path, outputs: dict[str, str]) -> None:
    rows = [
        {
            "task_id": task_id,
            "split": "train",
            "collection": "clapnq",
            "output_text": output,
        }
        for task_id, output in outputs.items()
    ]
    path.write_text(
        json.dumps(
            {
                "analysis": "mtrag-reference-quality-calibration",
                "model": "test-model",
                "max_completion_tokens": 32,
                "request_count": len(rows),
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


class CompareMtragReferenceStabilityTests(TestCase):
    # Count exact matches while retaining task-level hashes for changed outputs.
    def test_compares_repeated_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            repeat = root / "repeat.json"
            _write_reference(baseline, {"task-a": "same", "task-b": "first"})
            _write_reference(repeat, {"task-b": "second", "task-a": "same"})

            report = compare_mtrag_reference_stability(baseline, repeat)

        self.assertEqual(report["request_count"], 2)
        self.assertEqual(report["exact_match_count"], 1)
        self.assertEqual(report["changed_count"], 1)
        self.assertEqual(report["exact_match_rate"], 0.5)
        changed = [
            row for row in report["comparisons"] if not row["exact_output_match"]
        ]
        self.assertEqual(changed[0]["task_id"], "task-b")
        self.assertNotEqual(
            changed[0]["baseline_output_sha256"],
            changed[0]["repeat_output_sha256"],
        )

    # Refuse comparisons that do not cover exactly the same task identities.
    def test_rejects_different_task_sets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            repeat = root / "repeat.json"
            _write_reference(baseline, {"task-a": "answer"})
            _write_reference(repeat, {"task-b": "answer"})

            with self.assertRaisesRegex(ValueError, "different task IDs"):
                compare_mtrag_reference_stability(baseline, repeat)
