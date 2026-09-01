import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.export_reuse_budget_models import export_reuse_budget_models


class ExportReuseBudgetModelsTests(TestCase):
    def test_changes_only_the_runtime_threshold_and_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "selector.json"
            model.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "selected_repair_threshold": 0.2,
                        "reuse_budget_thresholds": {
                            target: index / 10
                            for index, target in enumerate(
                                ("0.05", "0.10", "0.25", "0.50", "0.75", "1.00"),
                                1,
                            )
                        },
                        "weights": [1, 2, 3],
                    }
                ),
                encoding="utf-8",
            )

            outputs = export_reuse_budget_models(model, root / "variants")
            first = json.loads(outputs["0.05"].read_text(encoding="utf-8"))
            last = json.loads(outputs["1.00"].read_text(encoding="utf-8"))

        self.assertEqual(len(outputs), 6)
        self.assertEqual(first["selected_repair_threshold"], 0.1)
        self.assertEqual(last["selected_repair_threshold"], 0.6)
        self.assertEqual(first["weights"], [1, 2, 3])
        self.assertEqual(first["reuse_budget"]["target_reuse_rate"], 0.05)
