from unittest import TestCase

from benchmarks.block_dataset import LabelSource, RepairDecision
from benchmarks.counterfactual_labels import (
    CounterfactualExecutionEvidence,
    SingleBlockIntervention,
    build_single_block_interventions,
    counterfactual_block_label,
    score_single_block_intervention,
    validate_counterfactual_execution,
)


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
                "cacheselect_counterfactual_reuse_block_index": selected_target,
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
