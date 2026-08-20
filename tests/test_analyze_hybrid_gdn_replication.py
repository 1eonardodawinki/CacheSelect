import unittest

from benchmarks.analyze_hybrid_gdn_replication import (
    analyze_hybrid_gdn_replication,
)
from benchmarks.hybrid_gdn_breakeven import summarize_hybrid_gdn_breakeven


# Build a valid source artifact from compact per-condition speedup values.
def _summary(run_id: str, capacity: int, values: dict[int, list[float]]) -> dict:
    """Return one synthetic break-even summary for replication tests."""
    trials = [
        {
            "reused_block_count": count,
            "speedup": speedup,
            "ttft_speedup": speedup,
            "reuse_fraction": count / (count + 1),
            "prompt_token_count": (count + 1) * 64,
            "reused_prompt_tokens": count * 64,
            "exact_output_match": True,
        }
        for count, speedups in values.items()
        for speedup in speedups
    ]
    summary = summarize_hybrid_gdn_breakeven(trials)
    summary.update(
        {
            "run_id": run_id,
            "model": "Qwen/Qwen3.5-9B",
            "block_size": 64,
            "cache_capacity": capacity,
            "trials": trials,
        }
    )
    return summary


class AnalyzeHybridGDNReplicationTests(unittest.TestCase):
    # Treat conflicting and single-run near-parity timings as noise.
    def test_requires_replicated_interval_above_parity(self) -> None:
        first = _summary("first", 16, {16: [0.98, 0.99, 0.995]})
        second = _summary(
            "second",
            48,
            {
                16: [1.03, 1.01, 1.02, 0.95, 0.997],
                32: [1.03, 1.02, 1.01, 0.99, 0.98],
            },
        )

        analysis = analyze_hybrid_gdn_replication(
            [first, second], bootstrap_samples=2_000
        )

        self.assertEqual(analysis["decision"], "NO_RELIABLE_SPEEDUP")
        sixteen = next(
            cell for cell in analysis["cells"] if cell["reused_block_count"] == 16
        )
        self.assertEqual(sixteen["source_run_count"], 2)
        self.assertLess(sixteen["wall_speedup_95pct_interval"][0], 1.0)
        self.assertGreater(sixteen["wall_speedup_95pct_interval"][1], 1.0)
        self.assertEqual(sixteen["evidence"], "INCONCLUSIVE")

    # Accept a speedup only when independent runs both support it after pooling.
    def test_identifies_a_replicated_speedup(self) -> None:
        first = _summary("first", 16, {16: [1.04, 1.05, 1.06, 1.05, 1.04]})
        second = _summary("second", 48, {16: [1.02, 1.03, 1.04, 1.03, 1.02]})

        analysis = analyze_hybrid_gdn_replication(
            [first, second], bootstrap_samples=2_000
        )

        self.assertEqual(analysis["decision"], "REPLICATED_SPEEDUP")
        self.assertEqual(analysis["replicated_speedup_block_counts"], [16])
        self.assertEqual(analysis["cells"][0]["evidence"], "REPLICATED_FASTER")


if __name__ == "__main__":
    unittest.main()
