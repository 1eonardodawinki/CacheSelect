from unittest import TestCase

from benchmarks.evaluation import compare_response_quality, score_response
from benchmarks.schema import (
    AnswerRequirement,
    ReferenceSimilarityGate,
    RequestGroundTruth,
)


# Build one calibrated natural-answer key for focused scoring tests.
def _reference_ground_truth() -> RequestGroundTruth:
    return RequestGroundTruth(
        expected_answer="The destination is London by train.",
        requirements=[],
        reference_similarity_gate=ReferenceSimilarityGate(
            minimum_token_recall=0.6,
            minimum_rouge_l_f1=0.5,
            maximum_metric_drop=0.05,
            calibration_id="mtrag-qwen-v1",
        ),
    )


class ReferenceQualityEvaluationTests(TestCase):
    # Pass a natural answer that clears both calibrated reference thresholds.
    def test_reference_similarity_passes(self):
        result = score_response(
            "The destination is London by train.",
            _reference_ground_truth(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["mode"], "reference_similarity")
        self.assertEqual(result["metrics"]["token_recall"], 1.0)
        self.assertEqual(result["calibration_id"], "mtrag-qwen-v1")

    # Reject fluent text that does not recover the published answer.
    def test_reference_similarity_fails_irrelevant_answer(self):
        result = score_response("I cannot determine that.", _reference_ground_truth())

        self.assertFalse(result["passed"])
        self.assertLess(result["score"], 0.5)

    # Preserve exact-fact scoring for the existing synthetic workloads.
    def test_requirements_mode_is_unchanged(self):
        ground_truth = RequestGroundTruth(
            "NORTH-731",
            [AnswerRequirement("code", ["north-731"])],
        )

        result = score_response("The code is NORTH-731.", ground_truth)

        self.assertTrue(result["passed"])
        self.assertEqual(result["mode"], "requirements")

    # Never treat an unconfigured empty answer key as a successful response.
    def test_missing_quality_policy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no configured quality gate"):
            score_response("anything", RequestGroundTruth("answer", []))

    # Reject an intervention that clears the floor but regresses too far.
    def test_pairwise_comparison_rejects_excessive_drop(self):
        ground_truth = _reference_ground_truth()
        reference = score_response(ground_truth.expected_answer, ground_truth)
        intervention = score_response(
            "The destination is London.",
            ground_truth,
        )

        comparison = compare_response_quality(
            reference,
            intervention,
            ground_truth,
        )

        self.assertTrue(intervention["passed"])
        self.assertFalse(comparison["passed"])
        self.assertGreater(comparison["metric_drops"]["token_recall"], 0.05)

    # Accept an intervention whose metrics stay within the calibrated tolerance.
    def test_pairwise_comparison_accepts_small_drop(self):
        ground_truth = _reference_ground_truth()
        reference = score_response(ground_truth.expected_answer, ground_truth)
        comparison = compare_response_quality(
            reference,
            reference,
            ground_truth,
        )

        self.assertTrue(comparison["passed"])
        self.assertEqual(
            comparison["metric_drops"],
            {"token_recall": 0.0, "rouge_l_f1": 0.0},
        )
