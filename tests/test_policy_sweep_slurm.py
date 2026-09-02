import subprocess
from pathlib import Path
from unittest import TestCase


class PolicySweepSlurmTests(TestCase):
    def test_runs_native_apc_and_all_frozen_logistic_budgets(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "run_imperial_qwen3_14b_policy_sweep.slurm"
        )
        subprocess.run(["bash", "-n", script], check=True)
        content = script.read_text(encoding="utf-8")

        self.assertIn("CACHESELECT_NATIVE_APC_EVALUATION=1", content)
        self.assertIn("CACHESELECT_POLICY_SPLIT=validation", content)
        self.assertIn("cacheselect-inputs/qwen3-mtrag-v3", content)
        self.assertIn("CACHESELECT_CORRECT_KV_POSITIONS=1", content)
        self.assertIn("CACHESELECT_MIN_REUSE_SPAN_BLOCKS", content)
        self.assertIn("for BUDGET in 005 010 025 050 075 100", content)
        self.assertIn("analyze_policy_reuse_sweep", content)
