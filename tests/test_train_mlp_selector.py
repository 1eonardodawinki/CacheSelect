import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from benchmarks.train_mlp_selector import (
    balanced_binary_training_rows,
    export_mlp_selector,
    sweep_mlp_architectures,
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
        self.assertEqual(report["feature_schema"], "block-v1")
        self.assertGreaterEqual(metrics["repair_recall"], 0.95)
        self.assertGreaterEqual(metrics["selected_reuse_rate"], 0.60)
        self.assertFalse(report["test_split_evaluated"])
        self.assertTrue(report["training"]["converged"])
        self.assertEqual(report["hyperparameters"]["hidden_layer_sizes"], [8])
        self.assertEqual(model.n_features_in_, 17)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "mlp.json"
            artifact = export_mlp_selector(model, report, output)
            self.assertEqual(json.loads(output.read_text()), artifact)
        self.assertEqual(len(artifact["standardizer"]["mean"]), 17)
        self.assertEqual(len(artifact["layers"]), 2)
        self.assertEqual(artifact["layers"][0]["activation"], "relu")
        self.assertEqual(artifact["layers"][1]["activation"], "logistic")
        self.assertEqual(
            set(artifact["reuse_budget_thresholds"]),
            {"0.05", "0.10", "0.25", "0.50", "0.75", "1.00"},
        )

    def test_configures_hidden_layers(self):
        model, report = train_mlp_selector(
            Path("results/block-dataset-v1/candidate-blocks.csv"),
            hidden_layer_sizes=(4, 2),
        )

        self.assertEqual(report["hyperparameters"]["hidden_layer_sizes"], [4, 2])
        self.assertEqual(model.named_steps["classifier"].hidden_layer_sizes, (4, 2))

    def test_sweeps_grouped_out_of_fold_predictions(self):
        labels = np.tile([0, 1], 8)
        features = np.column_stack((labels, np.arange(len(labels))))
        groups = np.repeat(np.arange(8), 2)
        with (
            patch(
                "benchmarks.train_mlp_selector.load_selector_dataset",
                return_value={"train": (features, labels)},
            ),
            patch(
                "benchmarks.train_mlp_selector._training_groups",
                return_value=groups,
            ),
        ):
            report = sweep_mlp_architectures(
                Path("unused.csv"), architectures=((2,),), folds=2
            )

        self.assertEqual(report["recommended_hidden_layer_sizes"], [2])
        self.assertEqual(report["conversation_groups"], 8)
