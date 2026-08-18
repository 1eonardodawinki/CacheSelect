import subprocess
from pathlib import Path
from unittest import TestCase


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "run_hybrid_checkpoint_smoke.slurm"
)


class HybridCheckpointSlurmTests(TestCase):
    # Prevent an invalid shell script from consuming an A30 allocation.
    def test_has_valid_shell_syntax(self):
        subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
        )

    # Preserve the exact runtime and output-validation contract under test.
    def test_preserves_hybrid_checkpoint_contract(self):
        script = SCRIPT.read_text(encoding="utf-8")
        required_tokens = (
            "#SBATCH --partition=a30",
            "#SBATCH --gres=gpu:nvidia_a30:1",
            "vllm serve",
            "--enable-prefix-caching",
            "--mamba-cache-mode all",
            "--gdn-delta-cache-capacity",
            "--gdn-prefill-backend triton",
            "--no-enable-chunked-prefill",
            "--enforce-eager",
            "--enable-per-request-metrics",
            "VLLM_SERVER_DEV_MODE=1",
            "/server_info?config_format=json",
            'cache["mamba_cache_mode"] == "all"',
            'cache["gdn_delta_cache_capacity"] == int(sys.argv[2])',
            'additional["gdn_prefill_backend"] == "triton"',
            "gpu-after-load.csv",
            "python -m benchmarks.run_hybrid_apc_baseline",
            "--require-token-aligned-edits",
            "--require-edited-prefix-reuse",
            "--require-gdn-preflight-observability",
            "--require-gdn-shadow-execution",
            "--validate-against-reference",
            "python -m benchmarks.analyze_hybrid_gdn_shadow",
            'test -s "$SHADOW_ANALYSIS_PATH"',
            "--max-completion-tokens 64",
            'test -s "$SUMMARY_PATH"',
            'grep -q "GDN delta sidecar populated" "$SERVER_LOG"',
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, script)
