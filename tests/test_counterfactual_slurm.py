import subprocess
from pathlib import Path
from unittest import TestCase


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "run_counterfactual_dataset.slurm"
)


class CounterfactualSlurmTests(TestCase):
    # Verify the submitted file remains valid Bash before consuming a GPU.
    def test_has_valid_shell_syntax(self):
        subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
        )

    # Protect the server and runner flags required for causal block labels.
    def test_preserves_counterfactual_execution_contract(self):
        script = SCRIPT.read_text(encoding="utf-8")
        required_tokens = (
            "#SBATCH --array=0-2%3",
            "rag-00-to-01 rag-01-to-02 rag-02-to-03",
            "--enable-prefix-caching",
            "--enable-cacheselect",
            "--cacheselect-repair-selector full_block",
            "--cacheselect-execute-partial-reuse",
            "--no-enable-chunked-prefill",
            "--enforce-eager",
            "--enable-prompt-tokens-details",
            "--enable-per-request-metrics",
            "python -m benchmarks.run_counterfactual_dataset",
            "--trace benchmarks/traces/rag.json",
            '--transition-id "$TRANSITION_ID"',
            '--split "$DATASET_SPLIT"',
            '--model "$MODEL"',
            '--base-url "http://127.0.0.1:$PORT"',
            "--max-completion-tokens 96",
            '--run-id "counterfactual-$EXPERIMENT_ID-$TASK_ID-$TRANSITION_ID"',
            '--request-log-dir "$REQUEST_LOG_DIR"',
            '--output "$DATASET_PATH"',
            '--summary-output "$SUMMARY_PATH"',
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, script)
