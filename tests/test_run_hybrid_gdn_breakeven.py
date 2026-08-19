import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.run_hybrid_gdn_breakeven import run_hybrid_gdn_breakeven


# Represent each repeated filler phrase as two predictable test tokens.
def _fake_tokenize(*, prompt: str, **_kwargs) -> tuple[int, ...]:
    """Return deterministic source/target tokens without a live server."""
    marker = 11 if "marker A" in prompt else 12
    filler_count = prompt.count("Stable context follows")
    return (90, marker, *(30 for _ in range(filler_count * 2)), 70, 71)


class RunHybridGDNBreakEvenTests(unittest.TestCase):
    # Run every role in order and preserve exact active-execution evidence.
    def test_runs_interleaved_trials_against_one_server(self) -> None:
        calls = []

        # Return role-specific timings and a complete active target proof.
        def fake_request(**kwargs):
            calls.append(kwargs)
            block_count = int(kwargs["scenario"].split("-")[1])
            prompt_tokens = len(_fake_tokenize(prompt=kwargs["prompt"]))
            role = kwargs["role"]
            row = {
                "finish_reason": "stop",
                "cached_tokens": 0,
                "prompt_tokens": prompt_tokens,
                "output_text": "GDN break-even complete.",
                "client_wall_seconds": 2.0 if role == "reference" else 1.0,
                "time_to_first_token_ms": 200.0 if role == "reference" else 100.0,
                "gdn_delta_reuse": None,
            }
            if role == "target":
                executed_layers = 2
                reused = block_count * 4 * executed_layers
                row["gdn_delta_reuse"] = {
                    "plan": {
                        "block_size": 4,
                        "candidates": [
                            {"target_block_index": index}
                            for index in range(1, block_count + 1)
                        ],
                    },
                    "preflight_eligible": True,
                    "candidate_block_count": block_count,
                    "resolved_layer_count": executed_layers,
                    "active_executed_layer_count": executed_layers,
                    "active_reused_layer_tokens": reused,
                    "active_recomputed_layer_tokens": (
                        prompt_tokens * executed_layers - reused
                    ),
                    "active_complete": True,
                }
            return row

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "benchmarks.run_hybrid_gdn_breakeven._tokenize_prompt",
                    side_effect=_fake_tokenize,
                ),
                patch(
                    "benchmarks.run_hybrid_gdn_breakeven._run_recorded_request",
                    side_effect=fake_request,
                ),
                patch(
                    "benchmarks.run_hybrid_gdn_breakeven.validate_ledger",
                    return_value=SimpleNamespace(is_complete=True, failed=0),
                ),
            ):
                result = run_hybrid_gdn_breakeven(
                    base_url="http://127.0.0.1:8000",
                    model="test-model",
                    run_id="break-even",
                    output=root / "summary.json",
                    request_log_dir=root / "logs",
                    reused_block_counts=(1, 2),
                    repetitions=2,
                    block_size=4,
                    cache_capacity=2,
                )

            self.assertEqual(result["trial_count"], 4)
            self.assertTrue(result["all_outputs_exact"])
            self.assertEqual(result["first_break_even_reused_block_count"], 1)
            self.assertTrue((root / "summary.json").is_file())
            self.assertEqual(len(calls), 12)
            self.assertEqual(
                [calls[index]["role"] for index in range(6)],
                ["reference", "source", "target"] * 2,
            )
            for start in range(0, len(calls), 3):
                reference, source, target = calls[start : start + 3]
                self.assertNotEqual(reference["cache_salt"], source["cache_salt"])
                self.assertEqual(source["cache_salt"], target["cache_salt"])
                self.assertEqual(
                    target["cacheselect_source_request_id"],
                    source["cacheselect_request_id"],
                )

    # Stop before any HTTP request if the server cannot retain the largest case.
    def test_rejects_insufficient_cache_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "benchmarks.run_hybrid_gdn_breakeven._run_recorded_request"
            ) as request:
                with self.assertRaisesRegex(ValueError, "largest reuse condition"):
                    run_hybrid_gdn_breakeven(
                        base_url="http://127.0.0.1:8000",
                        model="test-model",
                        run_id="too-small",
                        output=root / "summary.json",
                        request_log_dir=root / "logs",
                        reused_block_counts=(1, 4),
                        repetitions=1,
                        block_size=4,
                        cache_capacity=2,
                    )
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
