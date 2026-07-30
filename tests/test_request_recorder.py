import json
import tempfile
import unittest
from pathlib import Path

from observability.request_recorder import RequestRecorder, validate_ledger


class RequestRecorderTest(unittest.TestCase):
    def test_complete_request_preserves_full_input_output_and_separates_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RequestRecorder(
                run_id="unit-test",
                model="test-model",
                backend="test",
                log_dir=tmp,
            )
            pending = recorder.start(
                model_input={
                    "messages": [{"role": "user", "content": "full input"}],
                    "rendered_prompt": "full input",
                },
                sampling={"temperature": 0.0},
                metadata={"policy": "FULL"},
                evaluation={"ground_truth": "expected"},
            )
            pending.complete(
                output={"text": "full output", "token_ids": [1, 2]},
                metrics={"ttft_seconds": 0.1},
            )

            events = [
                json.loads(line)
                for line in Path(recorder.path).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([e["event"] for e in events], [
                "request_started",
                "request_completed",
            ])
            self.assertEqual(events[0]["model_input"]["rendered_prompt"], "full input")
            self.assertEqual(events[0]["evaluation"]["ground_truth"], "expected")
            self.assertNotIn("evaluation", events[0]["model_input"])
            self.assertEqual(events[1]["output"]["text"], "full output")
            self.assertEqual(events[0]["request_id"], events[1]["request_id"])

    def test_failed_request_preserves_input_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RequestRecorder(
                run_id="failure-test",
                model="test-model",
                backend="test",
                log_dir=tmp,
            )
            pending = recorder.start(model_input={"rendered_prompt": "input before crash"})
            try:
                raise ValueError("deliberate failure")
            except ValueError as error:
                pending.fail(error)

            events = [
                json.loads(line)
                for line in Path(recorder.path).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[0]["model_input"]["rendered_prompt"], "input before crash")
            self.assertEqual(events[1]["event"], "request_failed")
            self.assertEqual(events[1]["error"]["type"], "ValueError")
            self.assertIn("deliberate failure", events[1]["error"]["message"])

    def test_validation_detects_an_unfinished_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RequestRecorder(
                run_id="interrupted-test",
                model="test-model",
                backend="test",
                log_dir=tmp,
            )
            recorder.start(model_input={"rendered_prompt": "persisted input"})

            summary = validate_ledger(recorder.path)

            self.assertEqual(summary.started, 1)
            self.assertEqual(summary.completed, 0)
            self.assertEqual(len(summary.incomplete_request_ids), 1)
            self.assertFalse(summary.is_complete)


if __name__ == "__main__":
    unittest.main()
