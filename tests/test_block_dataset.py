from dataclasses import replace
from unittest import TestCase

from benchmarks.block_dataset import (
    LabelSource,
    RepairDecision,
    build_synthetic_dependency_examples,
    label_candidate_block,
)
from benchmarks.schema import (
    PromptSegment,
    RequestGroundTruth,
    RequestSpec,
    RequestTransition,
    TransitionGroundTruth,
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

    # Verify annotated pointer dependencies become per-block repair labels.
    def test_builds_labels_from_rendered_segment_locations(self):
        words = [
            "prefix",
            "pointer-B",
            "stable-a",
            "stable-b",
            "version-A",
            "NORTH",
            "stable-c",
            "stable-d",
            "version-B",
            "SOUTH",
            "query",
            "which-code",
        ]
        rendered_prompt = " ".join(words)
        token_offsets = []
        cursor = 0
        for word in words:
            token_offsets.append((cursor, cursor + len(word)))
            cursor += len(word) + 1

        request = RequestSpec(
            request_id="pointer-edited",
            workload="quality_stress",
            sequence_index=1,
            messages=[{"role": "user", "content": rendered_prompt}],
            segments=[
                PromptSegment("pointer", "user", "pointer", 2, "pointer-B"),
                PromptSegment(
                    "version_a_fact", "user", "retrieved_fact", 1, "version-A NORTH"
                ),
                PromptSegment(
                    "version_b_fact", "user", "retrieved_fact", 1, "version-B SOUTH"
                ),
                PromptSegment("query", "user", "query", 1, "query which-code"),
            ],
            ground_truth=RequestGroundTruth("SOUTH", []),
        )
        transition = RequestTransition(
            transition_id="pointer-base-to-edit",
            previous_request_id="pointer-base",
            current_request_id=request.request_id,
            ground_truth=TransitionGroundTruth(
                change_type="pointer_edit",
                changed_segment_ids=["pointer"],
                expected_native_behavior="prefix_hit_until_pointer",
                dependent_segment_ids=[
                    "version_a_fact",
                    "version_b_fact",
                    "query",
                ],
            ),
        )
        features = tuple(
            replace(
                BASE_FEATURES,
                current_token_count=len(words),
                previous_token_count=len(words),
                block_size=2,
                candidate_block_index=block_index,
            )
            for block_index in range(1, 6)
        )

        examples = build_synthetic_dependency_examples(
            trace_id="pointer-trace",
            transition=transition,
            current_request=request,
            rendered_prompt=rendered_prompt,
            token_offsets=token_offsets,
            features=features,
        )

        self.assertEqual(
            [example.label.decision for example in examples],
            [
                RepairDecision.REUSE,
                RepairDecision.REPAIR,
                RepairDecision.REUSE,
                RepairDecision.REPAIR,
                RepairDecision.REPAIR,
            ],
        )
        self.assertIn("version_a_fact", examples[1].label.reason)
        self.assertEqual(
            {example.label.source for example in examples},
            {LabelSource.SYNTHETIC_DEPENDENCY},
        )

    # Verify unannotated scenarios cannot silently label every block as reusable.
    def test_rejects_transition_without_dependency_annotations(self):
        request = RequestSpec(
            request_id="edited",
            workload="quality_stress",
            sequence_index=1,
            messages=[],
            segments=[],
            ground_truth=RequestGroundTruth("answer", []),
        )
        transition = RequestTransition(
            transition_id="base-to-edit",
            previous_request_id="base",
            current_request_id="edited",
            ground_truth=TransitionGroundTruth("edit", ["fact"], "prefix_miss"),
        )

        with self.assertRaisesRegex(ValueError, "no annotated dependent segments"):
            build_synthetic_dependency_examples(
                trace_id="trace",
                transition=transition,
                current_request=request,
                rendered_prompt="answer",
                token_offsets=[(0, 6)],
                features=[
                    replace(
                        BASE_FEATURES,
                        current_token_count=1,
                        previous_token_count=1,
                        block_size=1,
                        candidate_block_index=0,
                    )
                ],
            )
