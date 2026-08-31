from pathlib import Path
from unittest import TestCase

import numpy as np

from benchmarks.compare_selectors import (
    _clustered_intervals,
    compare_validation_selectors,
)
from benchmarks.train_boosted_selector import train_boosted_selector


class BoostedSelectorTests(TestCase):
    # Resample whole conversations and keep the uncertainty calculation reproducible.
    def test_clustered_intervals_are_deterministic(self):
        labels = np.asarray([1, 0, 1, 0])
        predicted = np.asarray([True, False, False, False])
        groups = np.asarray(["a", "a", "b", "b"])

        first = _clustered_intervals(labels, predicted, groups, samples=100)
        second = _clustered_intervals(labels, predicted, groups, samples=100)

        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {"repair_recall", "safe_reuse_precision", "selected_reuse_rate"},
        )

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
