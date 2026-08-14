from dataclasses import replace
from unittest import TestCase

from benchmarks.block_dataset import (
    LabelSource,
    RepairDecision,
    label_candidate_block,
)
from cacheselect.block_features import CandidateBlockFeatures


BASE_FEATURES = CandidateBlockFeatures(
    previous_token_count=64,
    current_token_count=64,
    previous_changed_token_count=4,
    current_changed_token_count=4,
    block_size=16,
    candidate_block_index=2,
    candidate_position_ratio=0.5,
    relative_block_offset=1,
    nearest_changed_block_distance=1,
    source_displacement_blocks=0.0,
    candidate_share_of_native_recompute=0.5,
    same_position_match=True,
    requires_repacking=False,
    changed_candidate_token_overlap_ratio=0.25,
    introduced_candidate_token_overlap_ratio=1.0,
    removed_candidate_token_overlap_ratio=0.0,
    changed_candidate_token_jaccard=0.1,
)


class BlockDatasetTests(TestCase):
    # Verify labels retain provenance while leaving runtime features unchanged.
    def test_attaches_benchmark_label_to_candidate_features(self):
        features = replace(BASE_FEATURES, candidate_block_index=3)

        example = label_candidate_block(
            trace_id="pointer-trace",
            transition_id="pointer-01-to-02",
            features=features,
            decision=RepairDecision.REPAIR,
            source=LabelSource.SYNTHETIC_DEPENDENCY,
            reason="The changed pointer selects a fact stored in this block.",
        )

        self.assertIs(example.features, features)
        self.assertEqual(example.features.candidate_block_index, 3)
        self.assertEqual(example.label.decision, RepairDecision.REPAIR)
        self.assertEqual(example.label.source, LabelSource.SYNTHETIC_DEPENDENCY)

    # Verify every stored label includes enough context to audit it later.
    def test_rejects_missing_label_context(self):
        for field_name in ("trace_id", "transition_id", "reason"):
            arguments = {
                "trace_id": "trace",
                "transition_id": "transition",
                "features": BASE_FEATURES,
                "decision": RepairDecision.REUSE,
                "source": LabelSource.COUNTERFACTUAL_EXECUTION,
                "reason": "Reusing this block preserved the reference output.",
            }
            arguments[field_name] = ""

            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    label_candidate_block(**arguments)
