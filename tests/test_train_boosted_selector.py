from pathlib import Path
from unittest import TestCase

from benchmarks.compare_selectors import compare_validation_selectors
from benchmarks.train_boosted_selector import train_boosted_selector


class BoostedSelectorTests(TestCase):
    # Verify the tree model trains without inspecting or reporting the test split.
    def test_trains_at_the_requested_validation_recall(self):
        model, report = train_boosted_selector(
            Path("results/block-dataset-v1/candidate-blocks.csv"),
            minimum_repair_recall=0.95,
        )

        self.assertEqual(report["model"], "hist_gradient_boosting")
        self.assertEqual(report["feature_schema"], "block-v1")
        metrics = report["validation"]["hist_gradient_boosting"]
        self.assertGreaterEqual(metrics["repair_recall"], 0.95)
        self.assertEqual(metrics["examples"], 745)
        self.assertNotIn("test", report)
        self.assertEqual(model.n_features_in_, 17)

    # Verify the shared scoreboard compares models without opening test data.
    def test_compares_validation_operating_points(self):
        report = compare_validation_selectors(
            Path("results/block-dataset-v1/candidate-blocks.csv")
        )
        logistic = report["models"]["logistic_regression"]
        boosted = report["models"]["hist_gradient_boosting"]
        mlp = report["models"]["mlp"]

        self.assertFalse(report["test_split_evaluated"])
        self.assertEqual(report["feature_schema"], "block-v1")
        self.assertEqual(
            set(boosted["operating_points"]), {"0.90", "0.95", "0.99", "1.00"}
        )
        self.assertGreater(
            boosted["metrics"]["selected_reuse_rate"],
            logistic["metrics"]["selected_reuse_rate"],
        )
        self.assertGreater(
            mlp["metrics"]["selected_reuse_rate"],
            boosted["metrics"]["selected_reuse_rate"],
        )
