import unittest

from benchmarks.analyze_hybrid_gdn_shadow import analyze_hybrid_gdn_shadow


# Build one synthetic completed layer/block shadow record.
def _shadow_record(layer: str, block: int, output_error: float, state_error: float):
    return {
        "layer_name": layer,
        "target_block_index": block,
        "source_contextual_hash": "ab",
        "reason": "compared",
        "output_relative_l2": output_error,
        "output_max_absolute_error": output_error * 2,
        "final_state_relative_l2": state_error,
        "final_state_max_absolute_error": state_error * 2,
    }


# Build a passing two-scenario smoke summary for analysis tests.
def _summary() -> dict:
    observations = []
    for scenario, block, scale in (
        ("early_edit", 7, 1.0),
        ("middle_edit", 12, 2.0),
    ):
        results = [
            _shadow_record("layers.0.gdn", block, 0.1 * scale, 0.01 * scale),
            _shadow_record("layers.1.gdn", block, 0.2 * scale, 0.02 * scale),
        ]
        observations.append(
            {
                "scenario": scenario,
                "role": "target",
                "gdn_delta_reuse": {
                    "shadow_complete": True,
                    "shadow_expected_count": len(results),
                    "shadow_results": results,
                },
            }
        )
    return {
        "run_id": "hybrid-shadow-test",
        "model": "Qwen/Qwen3.5-9B",
        "gdn_delta_shadow": {"passed": True},
        "observations": observations,
    }


class HybridGDNShadowAnalysisTests(unittest.TestCase):
    # Aggregate completed comparisons by scenario, layer and logical block.
    def test_summarizes_shadow_divergence(self) -> None:
        analysis = analyze_hybrid_gdn_shadow(_summary())

        self.assertEqual(analysis["overall"]["comparison_count"], 4)
        self.assertEqual(analysis["overall"]["max_output_relative_l2"], 0.4)
        self.assertEqual(len(analysis["by_layer"]), 2)
        self.assertEqual(analysis["by_layer"][0]["comparison_count"], 2)
        self.assertEqual(len(analysis["by_block"]), 2)

    # Reject a run whose claimed expected comparisons were not all recorded.
    def test_rejects_incomplete_shadow_results(self) -> None:
        summary = _summary()
        summary["observations"][0]["gdn_delta_reuse"]["shadow_expected_count"] = 3

        with self.assertRaisesRegex(ValueError, "expected count"):
            analyze_hybrid_gdn_shadow(summary)


if __name__ == "__main__":
    unittest.main()
