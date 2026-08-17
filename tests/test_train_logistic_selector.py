import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from cacheselect.selector_features import (
    CONTEXT_FEATURE_SCHEMA,
    FEATURE_NAMES,
)
from benchmarks.train_logistic_selector import (
    evaluate_selector,
    load_selector_dataset,
    select_repair_threshold,
)


class LogisticSelectorTests(TestCase):
    # Verify the committed dataset exposes only the declared model features.
    def test_loads_feature_matrix_and_all_splits(self):
        dataset = load_selector_dataset(
            Path("results/block-dataset-v1/candidate-blocks.csv")
        )

        self.assertEqual(set(dataset), {"train", "validation", "test"})
        self.assertEqual(dataset["train"][0].shape, (3020, len(FEATURE_NAMES)))
        self.assertEqual(int(np.sum(dataset["train"][1])), 577)

    # Read an expanded row in the exact order declared by the v2 schema.
    def test_loads_context_feature_schema(self):
        with TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "context.csv"
            fieldnames = ["split", "decision", *CONTEXT_FEATURE_SCHEMA.feature_names]
            expected = [float(index) for index in range(len(fieldnames) - 2)]
            expected[15:17] = [1.0, 0.0]
            with dataset_path.open("w", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                for split in ("train", "validation", "test"):
                    row = {
                        name: str(index)
                        for index, name in enumerate(
                            CONTEXT_FEATURE_SCHEMA.feature_names
                        )
                    }
                    row.update(
                        {
                            "split": split,
                            "decision": "reuse",
                            "same_position_match": "True",
                            "requires_repacking": "False",
                        }
                    )
                    writer.writerow(row)

            dataset = load_selector_dataset(
                dataset_path,
                feature_schema=CONTEXT_FEATURE_SCHEMA,
            )

        self.assertEqual(dataset["train"][0].shape, (1, 27))
        np.testing.assert_array_equal(dataset["train"][0][0], expected)

    # Verify threshold selection spends reuse only within the recall constraint.
    def test_selects_highest_threshold_meeting_repair_recall(self):
        labels = np.asarray([1, 1, 0, 0])
        probabilities = np.asarray([0.9, 0.6, 0.55, 0.1])

        threshold = select_repair_threshold(
            labels,
            probabilities,
            minimum_repair_recall=0.5,
        )

        self.assertEqual(threshold, 0.9)

    # Verify missed repairs reduce the safety-oriented reuse precision metric.
    def test_reports_safe_reuse_precision(self):
        metrics = evaluate_selector(
            np.asarray([1, 1, 0, 0]),
            np.asarray([True, False, False, False]),
        )

        self.assertEqual(metrics["missed_repair"], 1)
        self.assertEqual(metrics["safe_reuse"], 2)
        self.assertAlmostEqual(metrics["safe_reuse_precision"], 2 / 3)
