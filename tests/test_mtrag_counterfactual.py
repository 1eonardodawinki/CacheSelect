import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmarks.block_dataset import DatasetSplit
from benchmarks.counterfactual_trial import CounterfactualReferenceQualityError
from benchmarks.mtrag_counterfactual import run_mtrag_counterfactual_cases


# Build one compact case object for runner wiring tests.
def _case(index: int) -> SimpleNamespace:
    transition = SimpleNamespace(
        transition_id=f"transition-{index}",
        current_request_id=f"task-{index}",
    )
    trace = SimpleNamespace(
        trace_id=f"trace-{index}",
        transitions=[transition],
    )
    return SimpleNamespace(
        trace=trace,
        split=DatasetSplit.TRAIN,
        collection=f"collection-{index}",
        block_size=16,
        expected_candidate_block_indices=(1, 2, 3),
        expected_testable_block_indices=(1, 2),
        target_block_indices=(2,),
    )


class MtragCounterfactualTests(TestCase):
    # Run cases in order while forwarding their frozen plans and audited outputs.
    def test_runs_frozen_cases_through_existing_workflow(self):
        cases = (_case(1), _case(2))
        reference_outputs = {"task-1": "answer 1", "task-2": "answer 2"}
        recorder = SimpleNamespace(path=Path("requests.jsonl"))
        workflow_results = [
            {
                "trial_count": 1,
                "valid_training_rows": 1,
                "invalid_trials": 0,
                "abstained_trials": 0,
                "reference_drift_trials": 0,
                "repair_labels": 0,
                "reuse_labels": 1,
            },
            {
                "trial_count": 1,
                "valid_training_rows": 0,
                "invalid_trials": 0,
                "abstained_trials": 1,
                "reference_drift_trials": 1,
                "repair_labels": 0,
                "reuse_labels": 0,
            },
        ]
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                patch(
                    "benchmarks.mtrag_counterfactual.save_trace"
                ) as save_trace,
                patch(
                    "benchmarks.mtrag_counterfactual."
                    "run_counterfactual_dataset_workflow",
                    side_effect=workflow_results,
                ) as run_workflow,
            ):
                result = run_mtrag_counterfactual_cases(
                    cases,
                    reference_outputs=reference_outputs,
                    output_dir=output_dir,
                    url="http://server/v1/chat/completions",
                    model="test-model",
                    max_completion_tokens=384,
                    api_key=None,
                    timeout_seconds=300.0,
                    recorder=recorder,
                )

            first_summary = json.loads(
                (output_dir / "case-01" / "summary.json").read_text()
            )

        self.assertEqual(save_trace.call_count, 2)
        self.assertEqual(run_workflow.call_count, 2)
        first_call = run_workflow.call_args_list[0].kwargs
        self.assertEqual(first_call["selected_block_indices"], (2,))
        self.assertEqual(first_call["required_reference_output"], "answer 1")
        self.assertFalse(first_call["require_reference_output_match"])
        self.assertTrue(first_call["require_exact_output_match"])
        self.assertIs(first_call["recorder"], recorder)
        self.assertEqual(first_summary["current_task_id"], "task-1")
        self.assertEqual(first_summary["source_case_index"], 1)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["completed_case_count"], 2)
        self.assertEqual(result["skipped_reference_case_count"], 0)
        self.assertEqual(result["planned_target_blocks"], 2)
        self.assertEqual(result["trial_count"], 2)
        self.assertEqual(result["valid_training_rows"], 1)
        self.assertEqual(result["abstained_trials"], 1)
        self.assertEqual(result["reference_drift_trials"], 1)

    # Skip only an unusable uncached reference and continue with later cases.
    def test_records_bad_reference_and_continues(self):
        cases = (_case(1), _case(2))
        valid = {
            "trial_count": 1,
            "valid_training_rows": 1,
            "invalid_trials": 0,
            "abstained_trials": 0,
            "reference_drift_trials": 0,
            "repair_labels": 0,
            "reuse_labels": 1,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("benchmarks.mtrag_counterfactual.save_trace"),
                patch(
                    "benchmarks.mtrag_counterfactual."
                    "run_counterfactual_dataset_workflow",
                    side_effect=[
                        CounterfactualReferenceQualityError("bad reference"),
                        valid,
                    ],
                ),
            ):
                result = run_mtrag_counterfactual_cases(
                    cases,
                    reference_outputs={"task-1": "one", "task-2": "two"},
                    output_dir=root,
                    url="http://server/v1/chat/completions",
                    model="test-model",
                    max_completion_tokens=384,
                    api_key=None,
                    timeout_seconds=300.0,
                    recorder=SimpleNamespace(path=Path("requests.jsonl")),
                    source_case_start=66,
                )
            skipped = json.loads((root / "case-01" / "summary.json").read_text())

        self.assertEqual(skipped["status"], "skipped_reference_quality")
        self.assertEqual(skipped["source_case_index"], 66)
        self.assertEqual(result["source_case_end"], 67)
        self.assertEqual(result["completed_case_count"], 1)
        self.assertEqual(result["skipped_reference_case_count"], 1)
        self.assertEqual(result["skipped_target_blocks"], 1)
        self.assertEqual(result["trial_count"], 1)

    # Reject a case whose answer was never approved by the manual audit.
    def test_rejects_case_without_approved_reference(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "approved reference"):
                run_mtrag_counterfactual_cases(
                    (_case(1),),
                    reference_outputs={},
                    output_dir=Path(directory),
                    url="http://server/v1/chat/completions",
                    model="test-model",
                    max_completion_tokens=384,
                    api_key=None,
                    timeout_seconds=300.0,
                    recorder=SimpleNamespace(path=Path("requests.jsonl")),
                )
