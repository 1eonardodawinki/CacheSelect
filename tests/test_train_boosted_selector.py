from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from benchmarks.compare_selectors import (
    _clustered_intervals,
    compare_cross_validated_selectors,
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
        self.assertEqual(
            set(boosted["reuse_budget_points"]),
            {"0.05", "0.10", "0.25", "0.50", "0.75", "1.00"},
        )
        self.assertGreater(
            boosted["metrics"]["selected_reuse_rate"],
            logistic["metrics"]["selected_reuse_rate"],
        )
        self.assertGreater(
            mlp["metrics"]["selected_reuse_rate"],
            boosted["metrics"]["selected_reuse_rate"],
        )

    def test_compares_grouped_out_of_fold_predictions(self):
        labels = np.tile([0, 1], 10)
        features = np.zeros((len(labels), 17))
        features[:, :2] = np.column_stack((labels, np.arange(len(labels))))
        groups = np.repeat(np.arange(10), 2)
        with (
            patch(
                "benchmarks.compare_selectors.load_selector_dataset",
                return_value={"train": (features, labels)},
            ),
            patch(
                "benchmarks.compare_selectors._training_groups",
                return_value=groups,
            ),
        ):
            report = compare_cross_validated_selectors(
                Path("unused.csv"), folds=2
            )

        self.assertEqual(report["conversation_groups"], 10)
        self.assertEqual(report["folds"], 2)
        self.assertFalse(report["test_split_evaluated"])
        for result in report["models"].values():
            self.assertEqual(result["metrics"]["examples"], 20)
            self.assertGreaterEqual(result["metrics"]["repair_recall"], 0.95)
            self.assertEqual(result["reuse_budget_points"]["1.00"]["actual_reuse_rate"], 1.0)
