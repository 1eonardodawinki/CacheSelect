import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.run_hybrid_gdn_profile import run_hybrid_gdn_component_profile


# Represent each repeated filler phrase as two predictable test tokens.
def _fake_tokenize(*, prompt: str, **_kwargs) -> tuple[int, ...]:
    marker = 11 if "marker A" in prompt else 12
    filler_count = prompt.count("Stable context follows")
    return (90, marker, *(30 for _ in range(filler_count * 2)), 70, 71)


class RunHybridGDNProfileTests(unittest.TestCase):
    # Keep donor work outside the two isolated full and active profile windows.
    def test_profiles_reference_and_active_target_only(self) -> None:
        calls = []
        profiled_roles = []

        # Return complete execution evidence for the controlled three-request run.
        def fake_request(**kwargs):
            calls.append(kwargs)
            role = kwargs["role"]
            prompt_tokens = len(_fake_tokenize(prompt=kwargs["prompt"]))
            row = {
                "role": role,
                "finish_reason": "stop",
                "cached_tokens": 0,
                "prompt_tokens": prompt_tokens,
                "output_text": "GDN break-even complete.",
                "client_wall_seconds": 2.0 if role == "reference" else 1.0,
                "time_to_first_token_ms": 200.0 if role == "reference" else 100.0,
                "gdn_delta_reuse": None,
            }
            if role == "target":
                row["gdn_delta_reuse"] = {
                    "plan": {
                        "block_size": 4,
                        "candidates": [{"target_block_index": 1}],
                    },
                    "preflight_eligible": True,
                    "candidate_block_count": 1,
                    "resolved_layer_count": 2,
                    "active_executed_layer_count": 2,
                    "active_reused_layer_tokens": 8,
                    "active_recomputed_layer_tokens": prompt_tokens * 2 - 8,
                    "active_complete": True,
                }
            return row

        # Execute the callback while substituting a deterministic trace filename.
        def fake_capture(**kwargs):
            row = kwargs["run_request"]()
            profiled_roles.append(row["role"])
            return row, kwargs["profile_dir"] / f"{row['role']}.pt.trace.json.gz"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "benchmarks.run_hybrid_gdn_profile._tokenize_prompt",
                    side_effect=_fake_tokenize,
                ),
                patch(
                    "benchmarks.run_hybrid_gdn_profile._run_recorded_request",
                    side_effect=fake_request,
                ),
                patch(
                    "benchmarks.run_hybrid_gdn_profile.capture_profiled_request",
                    side_effect=fake_capture,
                ),
                patch(
                    "benchmarks.run_hybrid_gdn_profile.validate_ledger",
                    return_value=SimpleNamespace(is_complete=True, failed=0),
                ),
            ):
                result = run_hybrid_gdn_component_profile(
                    base_url="http://127.0.0.1:8000",
                    model="test-model",
                    run_id="profile",
                    output=root / "summary.json",
                    request_log_dir=root / "requests",
                    profile_dir=root / "profiles",
                    reused_block_count=1,
                    block_size=4,
                )
                summary_exists = (root / "summary.json").is_file()

        self.assertEqual(
            [call["role"] for call in calls],
            ["reference", "source", "target"],
        )
        self.assertEqual(profiled_roles, ["reference", "target"])
        self.assertEqual(
            result["profile_traces"],
            {
                "full_reference": str(root / "profiles/reference.pt.trace.json.gz"),
                "active_reuse": str(root / "profiles/target.pt.trace.json.gz"),
            },
        )
        self.assertTrue(result["all_outputs_exact"])
        self.assertTrue(summary_exists)
        self.assertNotEqual(calls[0]["cache_salt"], calls[1]["cache_salt"])
        self.assertEqual(calls[1]["cache_salt"], calls[2]["cache_salt"])


if __name__ == "__main__":
    unittest.main()
