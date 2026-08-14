from pathlib import Path
from unittest import TestCase

from benchmarks.train_boosted_selector import train_boosted_selector


class BoostedSelectorTests(TestCase):
    # Verify the tree model trains without inspecting or reporting the test split.
    def test_trains_at_the_requested_validation_recall(self):
        model, report = train_boosted_selector(
            Path("results/block-dataset-v1/candidate-blocks.csv"),
            minimum_repair_recall=0.95,
        )

        self.assertEqual(report["model"], "hist_gradient_boosting")
        self.assertGreaterEqual(report["validation"]["repair_recall"], 0.95)
        self.assertEqual(report["validation"]["examples"], 745)
        self.assertNotIn("test", report)
        self.assertEqual(model.n_features_in_, 17)
