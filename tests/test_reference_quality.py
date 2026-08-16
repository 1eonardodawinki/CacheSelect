from unittest import TestCase

from benchmarks.reference_quality import score_reference_answer


class ReferenceQualityTests(TestCase):
    # Treat punctuation, articles, case, and spacing like MTRAG token recall.
    def test_exact_content_normalizes_surface_form(self):
        scores = score_reference_answer(
            "DESTINATION is london!",
            "The destination is London.",
        )

        self.assertEqual(scores["token_recall"], 1.0)
        self.assertEqual(scores["rouge_l_f1"], 1.0)

    # Keep ordered overlap distinct from unordered reference-word coverage.
    def test_reordered_answer_has_full_recall_but_lower_rouge(self):
        scores = score_reference_answer(
            "London destination is",
            "The destination is London.",
        )

        self.assertEqual(scores["token_recall"], 1.0)
        self.assertLess(scores["rouge_l_f1"], 1.0)

    # Record an omitted answer as zero quality instead of passing by default.
    def test_missing_prediction_scores_zero(self):
        scores = score_reference_answer(None, "A documented answer")

        self.assertEqual(scores, {"token_recall": 0.0, "rouge_l_f1": 0.0})
