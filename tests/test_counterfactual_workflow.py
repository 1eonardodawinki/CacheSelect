from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmarks.block_dataset import DatasetSplit, RepairDecision
from benchmarks.counterfactual_trial import CounterfactualCandidateDiscovery
from benchmarks.counterfactual_workflow import run_counterfactual_dataset_workflow
from benchmarks.workloads import build_rag_trace


class CounterfactualWorkflowTests(TestCase):
    # Verify the workflow connects discovery, trials, export, and ledger audit.
    def test_runs_complete_counterfactual_dataset_workflow(self):
        trace = build_rag_trace()
        transition = trace.transitions[0]
        discovery = CounterfactualCandidateDiscovery(
            trace.trace_id,
            transition.transition_id,
            4,
            (1, 2, 3),
            (1, 2),
            3,
        )
        discovery_run = SimpleNamespace(
            discovery_id="discovery-id",
            discovery=discovery,
            edited_observation={"output_text": "fresh full-compute answer"},
            approved_output_exact_match=False,
        )
        batch = SimpleNamespace(
            trials=(
                SimpleNamespace(
                    label=SimpleNamespace(decision=RepairDecision.REUSE),
                    label_result=SimpleNamespace(
                        valid_reference=True,
                        valid_execution=True,
                        decision=RepairDecision.REUSE,
                    ),
                    reference_output_exact_match=True,
                ),
                SimpleNamespace(
                    label=None,
                    label_result=SimpleNamespace(
                        valid_reference=True,
                        valid_execution=True,
                        decision=None,
                    ),
                    reference_output_exact_match=False,
                ),
            )
        )
        recorder = SimpleNamespace(path=Path("requests.jsonl"))
        ledger = SimpleNamespace(
            is_complete=True,
            failed=0,
            path=recorder.path,
            started=8,
        )

        with (
            patch(
                "benchmarks.counterfactual_workflow."
                "run_counterfactual_candidate_discovery",
                return_value=discovery_run,
            ) as run_discovery,
            patch(
                "benchmarks.counterfactual_workflow."
                "run_discovered_counterfactual_trials",
                return_value=batch,
            ) as run_trials,
            patch(
                "benchmarks.counterfactual_workflow."
                "save_counterfactual_training_dataset",
                return_value=1,
            ) as save_dataset,
            patch(
                "benchmarks.counterfactual_workflow.validate_ledger",
                return_value=ledger,
            ),
        ):
            result = run_counterfactual_dataset_workflow(
                trace=trace,
                transition_id=transition.transition_id,
                split=DatasetSplit.TRAIN,
                output_path=Path("counterfactual-blocks.csv"),
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                recorder=recorder,
            )

        self.assertEqual(result["testable_blocks"], 2)
        self.assertEqual(result["planned_blocks"], 2)
        self.assertEqual(result["valid_training_rows"], 1)
        self.assertEqual(result["invalid_trials"], 0)
        self.assertEqual(result["abstained_trials"], 1)
        self.assertEqual(result["reference_drift_trials"], 1)
        self.assertEqual(result["reuse_labels"], 1)
        self.assertFalse(result["approved_reference_exact_match"])
        self.assertEqual(
            run_discovery.call_args.kwargs["source_request"].request_id,
            transition.previous_request_id,
        )
        self.assertIs(run_trials.call_args.kwargs["recorder"], recorder)
        self.assertEqual(
            run_trials.call_args.kwargs["selected_block_indices"],
            (1, 2),
        )
        self.assertEqual(
            run_trials.call_args.kwargs["required_reference_output"],
            "fresh full-compute answer",
        )
        self.assertIs(
            run_trials.call_args.kwargs["reference_observation"],
            discovery_run.edited_observation,
        )
        self.assertEqual(save_dataset.call_args.kwargs["split"], DatasetSplit.TRAIN)

    # Stop before causal trials when live discovery drifts from the frozen pilot.
    def test_rejects_unexpected_live_candidates(self):
        trace = build_rag_trace()
        transition = trace.transitions[0]
        discovery_run = SimpleNamespace(
            discovery_id="discovery-id",
            discovery=CounterfactualCandidateDiscovery(
                trace.trace_id,
                transition.transition_id,
                16,
                (1, 2),
                (1, 2),
                None,
            ),
            edited_observation={"output_text": "fresh full-compute answer"},
            approved_output_exact_match=True,
        )

        with (
            patch(
                "benchmarks.counterfactual_workflow."
                "run_counterfactual_candidate_discovery",
                return_value=discovery_run,
            ),
            patch(
                "benchmarks.counterfactual_workflow."
                "run_discovered_counterfactual_trials"
            ) as run_trials,
        ):
            with self.assertRaisesRegex(RuntimeError, "frozen pilot"):
                run_counterfactual_dataset_workflow(
                    trace=trace,
                    transition_id=transition.transition_id,
                    split=DatasetSplit.TRAIN,
                    output_path=Path("counterfactual-blocks.csv"),
                    url="http://vllm.test/v1/chat/completions",
                    model="test-model",
                    max_completion_tokens=8,
                    api_key=None,
                    timeout_seconds=2.0,
                    recorder=SimpleNamespace(path=Path("requests.jsonl")),
                    expected_block_size=16,
                    expected_candidate_block_indices=(1,),
                    expected_testable_block_indices=(1,),
                )

        run_trials.assert_not_called()
