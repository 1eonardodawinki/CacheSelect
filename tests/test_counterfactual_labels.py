from unittest import TestCase

from benchmarks.block_dataset import LabelSource, RepairDecision
from benchmarks.counterfactual_labels import (
    build_single_block_interventions,
    counterfactual_block_label,
    score_single_block_intervention,
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
            reference_output="wrong",
            intervention_output="wrong",
            reference_quality_passed=False,
            intervention_quality_passed=False,
        )
        self.assertFalse(result.valid_reference)
        with self.assertRaisesRegex(ValueError, "invalid reference"):
            counterfactual_block_label(result)
