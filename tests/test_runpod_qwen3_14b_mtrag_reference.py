import subprocess
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "run_runpod_qwen3_14b_mtrag_reference.sh"
)


class RunPodQwen3MtragReferenceTests(unittest.TestCase):
    # Prevent shell errors from wasting paid A40 time.
    def test_has_valid_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
        )

    # Preserve the model, frozen data, split, and full-computation contract.
    def test_preserves_reference_contract(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        required_tokens = (
            "Qwen/Qwen3-14B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "CACHESELECT_EXPECTED_GPU_NAME:-NVIDIA A40",
            "CACHESELECT_MINIMUM_GPU_MEMORY_MIB:-45000",
            "5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873",
            "raw.githubusercontent.com/IBM/mt-rag-benchmark/cc5b1d481b391181b89f7ced860308482e785463",
            "git diff --quiet",
            "git rev-parse origin/main",
            'export PYTHONPATH="$PROJECT_ROOT/vllm',
            "import vllm; print(vllm.__file__)",
            "CACHESELECT_MODEL_DTYPE=bfloat16",
            "CACHESELECT_MTRAG_MAX_COMPLETION_TOKENS=768",
            "CACHESELECT_HF_OFFLINE",
            "CACHESELECT_MTRAG_EXPAND_REFERENCES",
            "benchmarks.analyze_mtrag_coverage",
            "benchmarks.freeze_mtrag_reference_expansion",
            "EXPECTED_TRAIN_COUNT=60",
            "EXPECTED_VALIDATION_COUNT=15",
            "--local-files-only",
            "bash benchmarks/run_mtrag_reference_splits.slurm",
            "train-reference-calibration.json",
            "validation-reference-calibration.json",
            'row["cached_tokens"] == 0',
            'row["finish_reason"] == "stop"',
            "benchmarks.prepare_mtrag_reference_review",
            "blinded-reference-review.json",
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, script)
        self.assertNotIn("test-manifest.json", script)


if __name__ == "__main__":
    unittest.main()
