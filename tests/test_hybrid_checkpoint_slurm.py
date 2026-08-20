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
            '--gdn-delta-block-size "$DELTA_BLOCK_SIZE"',
            "CACHESELECT_GDN_DELTA_BLOCK_SIZE:-64",
            "CACHESELECT_GDN_DELTA_EXECUTION_MODE:-shadow",
            "CACHESELECT_RUN_ACTIVE_MATRIX:-0",
            "CACHESELECT_RUN_BREAK_EVEN_MATRIX:-0",
            "CACHESELECT_RUN_COMPONENT_PROFILE:-0",
            "CACHESELECT_GPU_MEMORY_UTILIZATION:-0.96",
            "CACHESELECT_MATRIX_RECORD_COUNTS:-64 112 160",
            "CACHESELECT_MATRIX_REPETITIONS:-3",
            "CACHESELECT_BREAK_EVEN_BLOCK_COUNTS:-1 2 4 8 16",
            "CACHESELECT_BREAK_EVEN_REPETITIONS:-3",
            "CACHESELECT_PROFILE_REUSED_BLOCK_COUNT:-32",
            '--gdn-delta-execution-mode "$EXECUTION_MODE"',
            '--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"',
            "--gdn-prefill-backend triton",
            "--no-enable-chunked-prefill",
            "--enforce-eager",
            "--enable-per-request-metrics",
            "VLLM_CUSTOM_SCOPES_FOR_PROFILING=1",
            "--profiler-config.profiler=torch",
            "--profiler-config.torch_profiler_dir=",
            "--profiler-config.ignore_frontend=true",
            "VLLM_SERVER_DEV_MODE=1",
            "VLLM_USE_V2_MODEL_RUNNER=1",
            'grep -q "Using V2 Model Runner" "$SERVER_LOG"',
            "/server_info?config_format=json",
            'cache["mamba_cache_mode"] == "all"',
            'cache["enable_cacheselect"] is False',
            'cache["cacheselect_execute_partial_reuse"] is False',
            "attention_reuse_policy=native_apc_only",
            'cache["gdn_delta_cache_capacity"] == int(sys.argv[2])',
            'cache["gdn_delta_block_size"] == int(sys.argv[3])',
            'cache["gdn_delta_execution_mode"] == sys.argv[4]',
            'additional["gdn_prefill_backend"] == "triton"',
            "gpu-after-load.csv",
            "python -m benchmarks.run_hybrid_apc_baseline",
            "--require-token-aligned-edits",
            "--require-gdn-preflight-observability",
            "--require-gdn-shadow-execution",
            "--require-gdn-active-execution",
            "--validate-against-reference",
            "python -m benchmarks.analyze_hybrid_gdn_shadow",
            "python -m benchmarks.analyze_hybrid_gdn_active",
            "python -m benchmarks.run_hybrid_gdn_matrix",
            "python -m benchmarks.run_hybrid_gdn_breakeven",
            "python -m benchmarks.run_hybrid_gdn_profile",
            'test -s "$SHADOW_ANALYSIS_PATH"',
            'test -s "$ACTIVE_ANALYSIS_PATH"',
            'test -s "$RESULT_DIR/matrix/matrix-summary.json"',
            'test -s "$RESULT_DIR/break-even/break-even-summary.json"',
            'test -s "$COMPONENT_PROFILE_PATH"',
            "--max-completion-tokens 64",
            'test -s "$SUMMARY_PATH"',
            'grep -q "GDN delta sidecar populated" "$SERVER_LOG"',
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, script)
        self.assertNotIn("--require-edited-prefix-reuse", script)
