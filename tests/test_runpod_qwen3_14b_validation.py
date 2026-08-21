import subprocess
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "run_runpod_qwen3_14b_validation.sh"
)


class RunPodQwen3ValidationTests(unittest.TestCase):
    # Prevent a shell error from consuming paid A40 time.
    def test_has_valid_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
        )

    # Preserve the exact model, precision, isolation, and evidence contract.
    def test_preserves_qwen3_14b_validation_contract(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        required_tokens = (
            "Qwen/Qwen3-14B",
            "CACHESELECT_EXPECTED_GPU_NAME:-NVIDIA A40",
            "CACHESELECT_MINIMUM_GPU_MEMORY_MIB:-45000",
            "git diff --quiet",
            "git rev-parse origin/main",
            'export PYTHONPATH="$PROJECT_ROOT/vllm',
            "import vllm; print(vllm.__file__)",
            'export HF_HUB_OFFLINE="${CACHESELECT_HF_OFFLINE:-0}"',
            "VLLM_SERVER_DEV_MODE=1",
            'setsid vllm serve "$MODEL"',
            "--dtype bfloat16",
            "--max-model-len 8192",
            "--max-num-seqs 1",
            "--max-num-batched-tokens 8192",
            "--block-size 16",
            "--enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--enforce-eager",
            "--enable-prompt-tokens-details",
            "--enable-per-request-metrics",
            "/server_info?config_format=json",
            "AutoConfig.from_pretrained(model_path, local_files_only=True)",
            "gpu-after-load.csv",
            "python -m benchmarks.run_full_attention_model_validation",
            'assert summary["passed"] is True',
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, script)
        self.assertNotIn("--enable-cacheselect", script)
        self.assertNotIn("--quantization", script)


if __name__ == "__main__":
    unittest.main()
