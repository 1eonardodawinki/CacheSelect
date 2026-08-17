import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.analyze_natural_selector_failures import (
    analyze_natural_selector_failures,
)
from cacheselect.selector_features import CONTEXT_FEATURE_SCHEMA


class NaturalSelectorFailureAnalysisTests(TestCase):
    # Preserve transition grouping when comparing repair and reuse features.
    def test_reports_global_and_within_transition_differences(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "natural.csv"
            fieldnames = [
                "transition_id",
                "decision",
                *CONTEXT_FEATURE_SCHEMA.feature_names,
            ]
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for transition, decision, candidate_index in (
                    ("transition-a", "repair", 4),
                    ("transition-a", "reuse", 2),
                    ("transition-b", "repair", 1),
                    ("transition-b", "reuse", 3),
                ):
                    row = {name: 0 for name in CONTEXT_FEATURE_SCHEMA.feature_names}
                    row.update(
                        transition_id=transition,
                        decision=decision,
                        candidate_block_index=candidate_index,
                        same_position_match="False",
                        requires_repacking="False",
                    )
                    writer.writerow(row)

            report = analyze_natural_selector_failures(path)

        self.assertEqual(report["examples"], 4)
        self.assertEqual(report["transitions"], 2)
        self.assertEqual(report["mixed_label_transitions"], 2)
        candidate = next(
            row
            for row in report["feature_summaries"]
            if row["feature"] == "candidate_block_index"
        )
        self.assertEqual(candidate["repair_median"], 2.5)
        self.assertEqual(candidate["reuse_median"], 2.5)
        self.assertEqual(candidate["mixed_transition_repair_higher"], 1)
        self.assertEqual(candidate["mixed_transition_repair_lower"], 1)
