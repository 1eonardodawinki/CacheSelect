import subprocess
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "run_runpod_hybrid_breakeven.sh"
)


class RunPodHybridBreakEvenTests(unittest.TestCase):
    # Prevent a shell error from wasting paid A40 time.
    def test_has_valid_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
        )

    # Preserve the GPU, source-checkout, configuration, and artifact contract.
    def test_preserves_runpod_provenance_contract(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        required_tokens = (
            "CACHESELECT_EXPECTED_GPU_NAME:-NVIDIA A40",
            "CACHESELECT_MINIMUM_GPU_MEMORY_MIB:-45000",
            "git diff --quiet",
            "git rev-parse origin/main",
            'export PYTHONPATH="$PROJECT_ROOT/vllm',
            "import vllm; print(vllm.__file__)",
            "CACHESELECT_GDN_DELTA_EXECUTION_MODE=active",
            "CACHESELECT_GDN_DELTA_CACHE_CAPACITY:-16",
            "CACHESELECT_RUN_ACTIVE_MATRIX=0",
            "CACHESELECT_RUN_BREAK_EVEN_MATRIX=1",
            "CACHESELECT_BREAK_EVEN_BLOCK_COUNTS:-1 2 4 8 16",
            "bash benchmarks/run_hybrid_checkpoint_smoke.slurm",
            'grep -Fx "project_commit=$PROJECT_COMMIT"',
            'grep -Fx "gdn_delta_execution_mode=active"',
            "python -m benchmarks.analyze_hybrid_gdn_breakeven",
            "break-even-analysis.json",
            "break-even-report.md",
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, script)


if __name__ == "__main__":
    unittest.main()
