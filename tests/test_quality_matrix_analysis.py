import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.analyze_cacheselect_quality import analyze_quality_matrix


# Write one complete synthetic condition in the production directory layout.
def _write_condition(
    root: Path,
    target_tokens: int,
    edit_radius: int,
    quality_scenarios: tuple[str, ...] = ("direct",),
) -> None:
    condition = root / f"tokens-{target_tokens}" / f"radius-{edit_radius}"
    analysis = condition / "analysis"
    analysis.mkdir(parents=True)
    (condition / "metadata.env").write_text(
        f"target_tokens={target_tokens}\n"
        f"edit_radius={edit_radius}\n"
        "answer_sensitive=1\n"
        f"quality_scenarios={' '.join(quality_scenarios)}\n",
        encoding="utf-8",
    )
    for quality_scenario in quality_scenarios:
        scenario_analysis = analysis
        if len(quality_scenarios) > 1:
            scenario_analysis /= quality_scenario
            scenario_analysis.mkdir()
        summaries = [
            {
                "target_tokens": target_tokens,
                "position": position,
                "repetitions": 3,
                "mean_reused_rows": 64.0,
                "mean_active_vs_native_ttft_ms": -10.0,
                "active_quality_pass_rate": 1.0,
                "active_shadow_exact_match_rate": 1.0,
                "mean_active_shadow_word_similarity": 1.0,
            }
            for position in ("early", "middle", "late")
        ]
        (scenario_analysis / "summary.json").write_text(
            json.dumps(summaries),
            encoding="utf-8",
        )


class QualityMatrixAnalysisTests(TestCase):
    # Verify nine complete conditions become 27 explicitly labelled cells.
    def test_complete_matrix_is_consolidated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for target_tokens in (256, 1024, 4096):
                for edit_radius in (0, 1, 2):
                    _write_condition(root, target_tokens, edit_radius)

            rows = analyze_quality_matrix(root)

        self.assertEqual(len(rows), 27)
        self.assertTrue(all(row["quality_scenario"] == "direct" for row in rows))
        self.assertEqual(
            (rows[0]["target_tokens"], rows[0]["edit_radius"], rows[0]["position"]),
            (256, 0, "early"),
        )
        self.assertEqual(
            (
                rows[-1]["target_tokens"],
                rows[-1]["edit_radius"],
                rows[-1]["position"],
            ),
            (4096, 2, "late"),
        )

    # Verify composed matrices retain their scenario label in every output row.
    def test_composed_matrix_is_labelled(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for target_tokens in (256, 1024, 4096):
                for edit_radius in (0, 1, 2):
                    _write_condition(
                        root,
                        target_tokens,
                        edit_radius,
                        quality_scenarios=("composed",),
                    )

            rows = analyze_quality_matrix(root, "composed")

        self.assertTrue(all(row["quality_scenario"] == "composed" for row in rows))

    # Verify a shared-server suite becomes one scenario-labelled result table.
    def test_multi_scenario_suite_is_consolidated(self):
        scenarios = ("direct", "conflict_5", "pointer", "rule")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for target_tokens in (256, 1024, 4096):
                for edit_radius in (0, 1, 2):
                    _write_condition(
                        root,
                        target_tokens,
                        edit_radius,
                        quality_scenarios=scenarios,
                    )

            rows = analyze_quality_matrix(root)

        self.assertEqual(len(rows), 108)
        self.assertEqual(
            {row["quality_scenario"] for row in rows},
            set(scenarios),
        )
