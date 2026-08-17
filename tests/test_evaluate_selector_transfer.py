import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.evaluate_selector_transfer import load_external_selector_rows
from cacheselect.selector_features import CONTEXT_FEATURE_SCHEMA


class ExternalSelectorRowsTests(TestCase):
    # Load a natural table even when every row belongs to the training corpus.
    def test_loads_single_split_external_rows(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "target.csv"
            fieldnames = ["split", "decision", *CONTEXT_FEATURE_SCHEMA.feature_names]
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                row = {name: "1" for name in CONTEXT_FEATURE_SCHEMA.feature_names}
                row.update(
                    split="train",
                    decision="repair",
                    same_position_match="True",
                    requires_repacking="False",
                )
                writer.writerow(row)

            features, labels = load_external_selector_rows(
                path,
                feature_schema=CONTEXT_FEATURE_SCHEMA,
            )

        self.assertEqual(features.shape, (1, 27))
        self.assertEqual(labels.tolist(), [1])
