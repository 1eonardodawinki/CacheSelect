from unittest import TestCase
from pathlib import Path

import numpy as np

from benchmarks.train_mlp_selector import (
    balanced_binary_training_rows,
    train_mlp_selector,
)


class MlpSelectorTests(TestCase):
    # Repeat the minority class without dropping either class's original rows.
    def test_balances_binary_training_rows(self):
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([0, 0, 0, 1])

        balanced_features, balanced_labels = balanced_binary_training_rows(
            features,
            labels,
        )

        self.assertEqual(balanced_features.shape, (6, 1))
        self.assertEqual(int(np.sum(balanced_labels == 0)), 3)
        self.assertEqual(int(np.sum(balanced_labels == 1)), 3)
        self.assertIn(3.0, balanced_features[:, 0])
        np.testing.assert_array_equal(
            balanced_features[:, 0],
            np.asarray([3.0, 2.0, 3.0, 3.0, 0.0, 1.0]),
        )

    # Produce the same row sequence whenever the experiment seed is unchanged.
    def test_balancing_is_reproducible(self):
        features = np.arange(10, dtype=float).reshape(5, 2)
        labels = np.asarray([0, 0, 0, 1, 1])

        first = balanced_binary_training_rows(features, labels, random_state=7)
        second = balanced_binary_training_rows(features, labels, random_state=7)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    # Train the neural selector without using the sealed test split.
    def test_trains_at_the_requested_validation_recall(self):
        model, report = train_mlp_selector(
            Path("results/block-dataset-v1/candidate-blocks.csv"),
            minimum_repair_recall=0.95,
        )

        metrics = report["validation"]["mlp"]
        self.assertGreaterEqual(metrics["repair_recall"], 0.95)
        self.assertGreaterEqual(metrics["selected_reuse_rate"], 0.60)
        self.assertFalse(report["test_split_evaluated"])
        self.assertTrue(report["training"]["converged"])
        self.assertEqual(model.n_features_in_, 17)
