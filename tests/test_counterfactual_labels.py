import csv
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from benchmarks.block_dataset import (
    BlockRepairLabel,
    DatasetSplit,
    LabelSource,
    RepairDecision,
)
from benchmarks.counterfactual_dataset import (
    extract_counterfactual_trial_feature,
    save_counterfactual_training_dataset,
)
from benchmarks.counterfactual_labels import (
    CounterfactualExecutionEvidence,
    CounterfactualLabelResult,
    SingleBlockIntervention,
    build_single_block_interventions,
    counterfactual_block_label,
    score_single_block_intervention,
    validate_counterfactual_execution,
)
from benchmarks.counterfactual_trial import (
    CounterfactualCandidateDiscovery,
    CounterfactualDiscoveryRunResult,
    CounterfactualTrialBatchResult,
    CounterfactualTrialResult,
    build_discovered_counterfactual_interventions,
    extract_counterfactual_candidate_discovery,
    run_counterfactual_candidate_discovery,
    run_discovered_counterfactual_trials,
    run_single_block_counterfactual_trial,
)
from benchmarks.workloads import build_rag_trace


# Build the server evidence required before a block may receive a label.
def _successful_execution(
    intervention: SingleBlockIntervention,
    block_size: int = 4,
    reported_target: int | None = None,
    executed: bool = True,
) -> CounterfactualExecutionEvidence:
    candidate_count = len(intervention.candidate_block_indices)
    selected_target = intervention.reused_block_index
    if reported_target is not None:
        selected_target = reported_target
    return validate_counterfactual_execution(
        intervention,
        block_size=block_size,
        server_metrics={
            "cacheselect_repair_selector": "counterfactual_single_block",
            "cacheselect_partial_reuse_plan": {
                "transition_id": intervention.transition_id,
                "block_size": block_size,
                "counterfactual_reuse_block_index": selected_target,
                "candidates": [
                    {"target_block_index": index, "source_resident": True}
                    for index in intervention.candidate_block_indices
                ],
            },
            "cacheselect_candidate_tokens": candidate_count * block_size,
            "cacheselect_repair_tokens": (candidate_count - 1) * block_size,
            "cacheselect_skipped_repair_tokens": block_size,
            "cacheselect_copied_blocks": candidate_count,
            "cacheselect_copied_tokens": candidate_count * block_size,
            "cacheselect_execution_eligible": executed,
            "cacheselect_execution_reason": "eligible" if executed else "fallback",
            "cacheselect_reused_batch_rows": block_size,
            "cacheselect_compacted_batch_built": executed,
            "cacheselect_span_metadata_built": executed,
            "cacheselect_compacted_batch_executed": executed,
        },
    )


class CounterfactualLabelTests(TestCase):
    # Verify a bad donor answer does not block safe candidate discovery.
    def test_runs_counterfactual_candidate_discovery(self):
        trace = build_rag_trace()
        source, edited = trace.requests[:2]
        source_observation = {
            "cached_tokens": 0,
            "runtime_policy": {"policy": "FULL_RECOMPUTE"},
            "server_metrics": {},
            "quality": {"passed": False},
        }
        edited_observation = {
            "output_text": edited.ground_truth.expected_answer,
            "finish_reason": "stop",
            "prompt_token_ids": list(range(12)),
            "prompt_token_count": 12,
            "quality": {"mode": "requirements", "passed": True},
            "server_metrics": {
                "cacheselect_partial_reuse_plan": {
                    "transition_id": trace.transitions[0].transition_id,
                    "block_size": 4,
                    "native_cached_tokens": 0,
                    "candidate_block_count": 2,
                    "candidate_token_count": 8,
                    "candidates": [
                        {"target_block_index": 1, "source_resident": True},
                        {"target_block_index": 2, "source_resident": True},
                    ],
                },
                "cacheselect_repair_selector": "full_block",
                "cacheselect_candidate_tokens": 8,
                "cacheselect_repair_tokens": 8,
                "cacheselect_skipped_repair_tokens": 0,
                "cacheselect_compacted_batch_executed": False,
            },
        }

        with patch(
            "benchmarks.counterfactual_trial._observe_request",
            side_effect=(source_observation, edited_observation),
        ) as observe:
            result = run_counterfactual_candidate_discovery(
                trace_id=trace.trace_id,
                transition=trace.transitions[0],
                source_request=source,
                edited_request=edited,
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                recorder=object(),
                required_edited_output="previously approved wording",
            )

        self.assertIsInstance(result, CounterfactualDiscoveryRunResult)
        self.assertEqual(result.discovery.testable_block_indices, (1,))
        self.assertFalse(result.approved_output_exact_match)
        calls = observe.call_args_list
        self.assertEqual(calls[0].kwargs["cache_salt"], calls[1].kwargs["cache_salt"])
        self.assertEqual(
            calls[1].kwargs["vllm_xargs"]["cacheselect_source_request_id"],
            calls[0].args[0].request_id,
        )

    # Verify a trial reconstructs the tested block's model-ready feature row.
    def test_extracts_counterfactual_trial_feature(self):
        intervention = SingleBlockIntervention("trace", "transition", (1, 2), 1)
        score = CounterfactualLabelResult(
            intervention, False, False, None, False, 0.0, "unused"
        )
        previous_tokens = list(range(16))
        current_tokens = [90, 91, 92, 93] + previous_tokens[2:6] + previous_tokens[8:]
        trial = CounterfactualTrialResult(
            "trial",
            intervention,
            {},
            {"prompt_token_ids": previous_tokens},
            {
                "prompt_token_ids": current_tokens,
                "server_metrics": {
                    "cacheselect_partial_reuse_plan": {
                        "block_size": 4,
                        "native_cached_tokens": 0,
                    }
                },
            },
            CounterfactualExecutionEvidence(False, "unused", None),
            score,
            None,
        )

        features = extract_counterfactual_trial_feature(trial, block_size=4)

        self.assertEqual(features.candidate_block_index, 1)
        self.assertTrue(features.requires_repacking)

    # Verify causal labels are persisted in the shared selector CSV schema.
    def test_saves_counterfactual_training_dataset(self):
        intervention = SingleBlockIntervention("trace", "transition", (1, 2, 3), 2)
        label = BlockRepairLabel(
            RepairDecision.REUSE,
            LabelSource.COUNTERFACTUAL_EXECUTION,
            "Quality remained correct.",
        )
        previous_tokens = list(range(16))
        current_tokens = previous_tokens[:4] + [90, 91, 92, 93] + previous_tokens[8:]
        trial = CounterfactualTrialResult(
            "trial",
            intervention,
            {},
            {"prompt_token_ids": previous_tokens},
            {
                "prompt_token_ids": current_tokens,
                "server_metrics": {
                    "cacheselect_partial_reuse_plan": {
                        "block_size": 4,
                        "native_cached_tokens": 4,
                    }
                },
            },
            CounterfactualExecutionEvidence(True, "executed", 4),
            CounterfactualLabelResult(
                intervention, True, True, RepairDecision.REUSE, True, 1.0, "safe"
            ),
            label,
        )
        batch = CounterfactualTrialBatchResult(
            CounterfactualCandidateDiscovery(
                "trace", "transition", 4, (1, 2, 3), (1, 2), 3
            ),
            (trial,),
            (2,),
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "counterfactual-blocks.csv"
            count = save_counterfactual_training_dataset(
                batch, split=DatasetSplit.TRAIN, path=output
            )
            with output.open(newline="") as input_file:
                rows = list(csv.DictReader(input_file))

        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["candidate_block_index"], "2")
        self.assertEqual(rows[0]["decision"], "reuse")
        self.assertEqual(rows[0]["label_source"], "counterfactual_execution")

        with TemporaryDirectory() as temporary_directory:
            empty_output = Path(temporary_directory) / "abstained-blocks.csv"
            empty_batch = replace(batch, reference_stable=False)
            empty_count = save_counterfactual_training_dataset(
                empty_batch,
                split=DatasetSplit.TRAIN,
                path=empty_output,
            )
            with empty_output.open(newline="") as input_file:
                empty_rows = list(csv.DictReader(input_file))

        self.assertEqual(empty_count, 0)
        self.assertEqual(empty_rows, [])

    # Verify discovery keeps all candidates but makes the output block untestable.
    def test_extracts_safe_counterfactual_candidates(self):
        observation = {
            "prompt_token_ids": list(range(12)),
            "prompt_token_count": 12,
            "server_metrics": {
                "cacheselect_partial_reuse_plan": {
                    "transition_id": "base-to-edit",
                    "block_size": 4,
                    "native_cached_tokens": 4,
                    "candidate_block_count": 2,
                    "candidate_token_count": 8,
                    "candidates": [
                        {
                            "source_block_index": 7,
                            "target_block_index": 2,
                            "source_resident": True,
                        },
                        {
                            "source_block_index": 7,
                            "target_block_index": 1,
                            "source_resident": True,
                        },
                    ],
                }
            },
        }

        discovery = extract_counterfactual_candidate_discovery(
            trace_id="rag-trace",
            transition_id="base-to-edit",
            observation=observation,
        )

        self.assertEqual(discovery.candidate_block_indices, (1, 2))
        self.assertEqual(discovery.testable_block_indices, (1,))
        self.assertEqual(discovery.excluded_output_block_index, 2)
        interventions = build_discovered_counterfactual_interventions(discovery)
        self.assertEqual([item.reused_block_index for item in interventions], [1])
        self.assertEqual(interventions[0].candidate_block_indices, (1, 2))
        self.assertEqual(interventions[0].repaired_block_indices, (2,))

    # Verify an inconsistent discovery cannot create misleading trial instructions.
    def test_rejects_inconsistent_discovered_interventions(self):
        discovery = CounterfactualCandidateDiscovery(
            trace_id="trace",
            transition_id="transition",
            block_size=4,
            candidate_block_indices=(1, 2),
            testable_block_indices=(1, 2),
            excluded_output_block_index=2,
        )

        with self.assertRaisesRegex(ValueError, "do not match"):
            build_discovered_counterfactual_interventions(discovery)

    # Verify discovered blocks run in order through fresh single-block trials.
    def test_runs_discovered_counterfactual_trials_sequentially(self):
        trace = build_rag_trace()
        source, edited = trace.requests[:2]
        discovery = CounterfactualCandidateDiscovery(
            trace_id=trace.trace_id,
            transition_id=trace.transitions[0].transition_id,
            block_size=4,
            candidate_block_indices=(1, 2, 3),
            testable_block_indices=(1, 2),
            excluded_output_block_index=3,
        )
        recorder = object()

        # Return the selected index so the test can inspect ordering directly.
        def fake_trial(**kwargs):
            return kwargs["intervention"].reused_block_index

        with patch(
            "benchmarks.counterfactual_trial.run_single_block_counterfactual_trial",
            side_effect=fake_trial,
        ) as run_trial:
            result = run_discovered_counterfactual_trials(
                discovery=discovery,
                source_request=source,
                edited_request=edited,
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                recorder=recorder,
                selected_block_indices=(2,),
            )

        self.assertEqual(result.trials, (2,))
        self.assertEqual(
            [call.kwargs["block_size"] for call in run_trial.call_args_list], [4]
        )
        self.assertTrue(
            all(
                call.kwargs["recorder"] is recorder for call in run_trial.call_args_list
            )
        )

    # Verify every trial reuses exactly one block and repairs all its peers.
    def test_builds_isolated_single_block_interventions(self):
        interventions = build_single_block_interventions(
            trace_id="trace",
            transition_id="base-to-edit",
            candidate_block_indices=[12, 10, 11],
        )
        self.assertEqual(
            [item.reused_block_index for item in interventions], [10, 11, 12]
        )
        self.assertEqual(interventions[1].repaired_block_indices, (10, 12))
        self.assertEqual(
            interventions[1].to_vllm_xargs(),
            {"cacheselect_counterfactual_reuse_block_index": "11"},
        )

    # Verify a task failure becomes a counterfactual repair label.
    def test_labels_quality_regression_as_repair(self):
        intervention = build_single_block_interventions(
            trace_id="trace",
            transition_id="transition",
            candidate_block_indices=[4],
        )[0]
        result = score_single_block_intervention(
            intervention,
            execution_evidence=_successful_execution(intervention),
            reference_output="SOUTH-913",
            intervention_output="NORTH-731",
            reference_quality_passed=True,
            intervention_quality_passed=False,
        )
        label = counterfactual_block_label(result)
        self.assertEqual(label.decision, RepairDecision.REPAIR)
        self.assertEqual(label.source, LabelSource.COUNTERFACTUAL_EXECUTION)
        reuse_result = score_single_block_intervention(
            intervention,
            execution_evidence=_successful_execution(intervention),
            reference_output="SOUTH-913",
            intervention_output="The answer remains SOUTH-913.",
            reference_quality_passed=True,
            intervention_quality_passed=True,
        )
        self.assertEqual(reuse_result.decision, RepairDecision.REUSE)

    # Defer every non-exact natural answer to blinded semantic review.
    def test_abstains_from_non_exact_automatic_reuse_label(self):
        intervention = build_single_block_interventions(
            trace_id="trace",
            transition_id="transition",
            candidate_block_indices=[4],
        )[0]

        result = score_single_block_intervention(
            intervention,
            execution_evidence=_successful_execution(intervention),
            reference_output="The limit is 31 days.",
            intervention_output="The limit is not 31 days.",
            reference_quality_passed=True,
            intervention_quality_passed=True,
            require_exact_output_match=True,
        )

        self.assertIsNone(result.decision)
        self.assertIn("manual review", result.reason)

    # Verify a bad full-compute answer cannot establish block ground truth.
    def test_rejects_invalid_reference(self):
        intervention = build_single_block_interventions(
            trace_id="trace",
            transition_id="transition",
            candidate_block_indices=[4],
        )[0]
        result = score_single_block_intervention(
            intervention,
            execution_evidence=_successful_execution(intervention),
            reference_output="wrong",
            intervention_output="wrong",
            reference_quality_passed=False,
            intervention_quality_passed=False,
        )
        self.assertFalse(result.valid_reference)
        with self.assertRaisesRegex(ValueError, "invalid experiment"):
            counterfactual_block_label(result)

    # Verify a fallback execution cannot silently become a reuse label.
    def test_rejects_unexecuted_intervention(self):
        intervention = build_single_block_interventions(
            trace_id="trace",
            transition_id="transition",
            candidate_block_indices=[4],
        )[0]
        evidence = _successful_execution(intervention, executed=False)
        wrong_target = _successful_execution(
            intervention,
            reported_target=5,
        )
        result = score_single_block_intervention(
            intervention,
            execution_evidence=evidence,
            reference_output="SOUTH-913",
            intervention_output="SOUTH-913",
            reference_quality_passed=True,
            intervention_quality_passed=True,
        )
        self.assertFalse(evidence.valid)
        self.assertFalse(wrong_target.valid)
        self.assertFalse(result.valid_execution)
        self.assertIsNone(result.decision)

    # Verify one trial isolates and orders its reference, donor, and intervention.
    def test_runs_one_isolated_counterfactual_trial(self):
        trace = build_rag_trace()
        source, edited = trace.requests[:2]
        intervention = SingleBlockIntervention(
            trace.trace_id, trace.transitions[0].transition_id, (4,), 4
        )
        fresh = {
            "output_text": edited.ground_truth.expected_answer,
            "finish_reason": "stop",
            "quality": {"mode": "requirements", "passed": True},
            "cached_tokens": 0,
            "runtime_policy": {"policy": "FULL_RECOMPUTE"},
            "server_metrics": {},
        }
        observations = [fresh, {**fresh}, {**fresh}]

        with (
            patch(
                "benchmarks.counterfactual_trial._observe_request",
                side_effect=observations,
            ) as observe,
            patch(
                "benchmarks.counterfactual_trial.validate_counterfactual_execution",
                return_value=_successful_execution(intervention),
            ),
        ):
            result = run_single_block_counterfactual_trial(
                source_request=source,
                edited_request=edited,
                intervention=intervention,
                block_size=4,
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                recorder=object(),
                required_reference_output=edited.ground_truth.expected_answer,
            )

        calls = observe.call_args_list
        roles = [call.args[0].request_id.rsplit(":", 1)[1] for call in calls]
        self.assertEqual(roles, ["reference", "donor", "intervention"])
        self.assertNotEqual(
            calls[0].kwargs["cache_salt"], calls[1].kwargs["cache_salt"]
        )
        self.assertEqual(calls[1].kwargs["cache_salt"], calls[2].kwargs["cache_salt"])
        active_xargs = calls[2].kwargs["vllm_xargs"]
        self.assertEqual(
            active_xargs["cacheselect_counterfactual_reuse_block_index"], "4"
        )
        self.assertEqual(
            active_xargs["cacheselect_source_request_id"], calls[1].args[0].request_id
        )
        self.assertEqual(result.label.decision, RepairDecision.REUSE)
        self.assertTrue(result.quality_comparison["passed"])

    # Preserve a long intervention but never turn its incomplete output into a label.
    def test_withholds_truncated_counterfactual_intervention(self):
        trace = build_rag_trace()
        source, edited = trace.requests[:2]
        intervention = SingleBlockIntervention(
            trace.trace_id, trace.transitions[0].transition_id, (4,), 4
        )
        fresh = {
            "request_id": "shared-reference",
            "output_text": edited.ground_truth.expected_answer,
            "finish_reason": "stop",
            "quality": {"mode": "requirements", "passed": True},
            "cached_tokens": 0,
            "runtime_policy": {"policy": "FULL_RECOMPUTE"},
            "server_metrics": {},
        }
        truncated = {
            **fresh,
            "output_text": "incomplete intervention",
            "finish_reason": "length",
        }

        with (
            patch(
                "benchmarks.counterfactual_trial._observe_request",
                side_effect=({**fresh}, truncated),
            ),
            patch(
                "benchmarks.counterfactual_trial.validate_counterfactual_execution",
                return_value=_successful_execution(intervention),
            ),
        ):
            result = run_single_block_counterfactual_trial(
                source_request=source,
                edited_request=edited,
                intervention=intervention,
                block_size=4,
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                recorder=object(),
                reference_observation=fresh,
                required_reference_output=edited.ground_truth.expected_answer,
                require_exact_output_match=True,
            )

        self.assertIsNone(result.label)
        self.assertIsNone(result.label_result.decision)
        self.assertIn("withheld", result.label_result.reason)
        self.assertEqual(result.intervention_observation, truncated)

    # Reuse one discovery answer and reject the case if the final answer drifts.
    def test_shares_reference_and_checks_final_stability(self):
        trace = build_rag_trace()
        source, edited = trace.requests[:2]
        discovery = CounterfactualCandidateDiscovery(
            trace.trace_id,
            trace.transitions[0].transition_id,
            4,
            (1, 2),
            (1, 2),
            None,
        )
        answer = edited.ground_truth.expected_answer
        fresh = {
            "request_id": "shared-reference",
            "output_text": answer,
            "finish_reason": "stop",
            "quality": {"mode": "requirements", "passed": True},
            "cached_tokens": 0,
            "runtime_policy": {"policy": "FULL_RECOMPUTE"},
            "server_metrics": {},
        }

        with (
            patch(
                "benchmarks.counterfactual_trial._observe_request",
                side_effect=({**fresh} for _ in range(5)),
            ) as observe,
            patch(
                "benchmarks.counterfactual_trial.validate_counterfactual_execution",
                side_effect=lambda intervention, **_: _successful_execution(
                    intervention
                ),
            ),
        ):
            result = run_discovered_counterfactual_trials(
                discovery=discovery,
                source_request=source,
                edited_request=edited,
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                recorder=object(),
                reference_observation=fresh,
                required_reference_output=answer,
            )

        self.assertEqual(observe.call_count, 5)
        self.assertTrue(
            all(trial.reference_observation is fresh for trial in result.trials)
        )
        self.assertTrue(
            observe.call_args_list[-1].args[0].request_id.endswith(
                ":stability_reference"
            )
        )
        self.assertEqual(
            observe.call_args_list[1].kwargs["policy_metadata"][
                "counterfactual_reference_request_id"
            ],
            "shared-reference",
        )

        drifted = {**fresh, "output_text": "different answer"}
        with (
            patch(
                "benchmarks.counterfactual_trial._observe_request",
                side_effect=({**fresh}, {**fresh}, drifted),
            ),
            patch(
                "benchmarks.counterfactual_trial.validate_counterfactual_execution",
                return_value=_successful_execution(
                    build_discovered_counterfactual_interventions(discovery)[0]
                ),
            ),
        ):
            unstable = run_discovered_counterfactual_trials(
                discovery=discovery,
                source_request=source,
                edited_request=edited,
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                recorder=object(),
                selected_block_indices=(1,),
                reference_observation=fresh,
                required_reference_output=answer,
            )

        self.assertFalse(unstable.reference_stable)

    # Stop after the reference when it differs from the manually audited text.
    def test_rejects_changed_approved_reference(self):
        trace = build_rag_trace()
        source, edited = trace.requests[:2]
        intervention = SingleBlockIntervention(
            trace.trace_id, trace.transitions[0].transition_id, (4,), 4
        )
        observation = {
            "output_text": "changed answer",
            "finish_reason": "stop",
            "quality": {"passed": True},
            "cached_tokens": 0,
            "runtime_policy": {"policy": "FULL_RECOMPUTE"},
            "server_metrics": {},
        }

        with patch(
            "benchmarks.counterfactual_trial._observe_request",
            return_value=observation,
        ) as observe:
            with self.assertRaisesRegex(RuntimeError, "approved reference"):
                run_single_block_counterfactual_trial(
                    source_request=source,
                    edited_request=edited,
                    intervention=intervention,
                    block_size=4,
                    url="http://vllm.test/v1/chat/completions",
                    model="test-model",
                    max_completion_tokens=8,
                    api_key=None,
                    timeout_seconds=2.0,
                    recorder=object(),
                    required_reference_output="audited answer",
                )

        self.assertEqual(observe.call_count, 1)

    # Record reference drift while preserving exact within-trial causal labels.
    def test_allows_recorded_reference_drift_for_natural_answers(self):
        trace = build_rag_trace()
        source, edited = trace.requests[:2]
        intervention = SingleBlockIntervention(
            trace.trace_id, trace.transitions[0].transition_id, (4,), 4
        )
        observation = {
            "output_text": "fresh full-compute wording",
            "finish_reason": "stop",
            "quality": {"mode": "requirements", "passed": True},
            "cached_tokens": 0,
            "runtime_policy": {"policy": "FULL_RECOMPUTE"},
            "server_metrics": {},
        }

        with (
            patch(
                "benchmarks.counterfactual_trial._observe_request",
                side_effect=(observation, observation, observation),
            ),
            patch(
                "benchmarks.counterfactual_trial.validate_counterfactual_execution",
                return_value=_successful_execution(intervention),
            ),
        ):
            result = run_single_block_counterfactual_trial(
                source_request=source,
                edited_request=edited,
                intervention=intervention,
                block_size=4,
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                recorder=object(),
                required_reference_output="previous discovery wording",
                require_reference_output_match=False,
                require_exact_output_match=True,
            )

        self.assertFalse(result.reference_output_exact_match)
        self.assertEqual(result.label.decision, RepairDecision.REUSE)
