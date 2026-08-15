import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmarks.block_dataset import DatasetSplit
from benchmarks.run_counterfactual_dataset import main
from benchmarks.schema import save_trace
from benchmarks.workloads import build_rag_trace


class CounterfactualDatasetCommandTests(TestCase):
    # Verify CLI settings reach the workflow and produce a durable summary.
    def test_runs_counterfactual_dataset_command(self):
        trace = build_rag_trace()
        transition_id = trace.transitions[0].transition_id
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trace_path = root / "trace.json"
            output_path = root / "counterfactual-blocks.csv"
            summary_path = root / "summary.json"
            save_trace(trace, trace_path)
            recorder = SimpleNamespace(path=root / "requests.jsonl")
            workflow_result = {
                "trace_id": trace.trace_id,
                "transition_id": transition_id,
                "split": "train",
                "valid_training_rows": 2,
                "request_ledger": str(recorder.path),
            }
            arguments = [
                "run_counterfactual_dataset",
                "--trace",
                str(trace_path),
                "--transition-id",
                transition_id,
                "--split",
                "train",
                "--model",
                "test-model",
                "--base-url",
                "http://vllm.test:8000/",
                "--output",
                str(output_path),
                "--summary-output",
                str(summary_path),
                "--run-id",
                "test-run",
            ]

            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "benchmarks.run_counterfactual_dataset.RequestRecorder",
                    return_value=recorder,
                ) as recorder_class,
                patch(
                    "benchmarks.run_counterfactual_dataset."
                    "run_counterfactual_dataset_workflow",
                    return_value=workflow_result,
                ) as workflow,
            ):
                main()
            summary = json.loads(summary_path.read_text())

        self.assertEqual(summary["run_id"], "test-run")
        self.assertEqual(summary["valid_training_rows"], 2)
        self.assertEqual(
            workflow.call_args.kwargs["url"],
            "http://vllm.test:8000/v1/chat/completions",
        )
        self.assertEqual(workflow.call_args.kwargs["split"], DatasetSplit.TRAIN)
        self.assertIs(workflow.call_args.kwargs["recorder"], recorder)
        self.assertEqual(
            recorder_class.call_args.kwargs["invocation_metadata"]["dataset_path"],
            str(output_path),
        )
