from pathlib import Path
from unittest import TestCase

from benchmarks.mtrag_quality import load_mtrag_manual_quality_audit


class MtragQualityTests(TestCase):
    # Load the frozen audit and expose only manually approved training tasks.
    def test_loads_frozen_manual_quality_audit(self):
        root = Path("results/mtrag-quality-calibration-v1")

        audit = load_mtrag_manual_quality_audit(
            root / "manual-audit-v1.json",
            reference_artifact_path=(
                root / "run-274842" / "reference-calibration.json"
            ),
            expected_model="Qwen/Qwen2.5-1.5B-Instruct",
        )

        self.assertEqual(audit.reviewed_task_count, 20)
        self.assertEqual(len(audit.approved_task_ids), 8)
        self.assertEqual(
            set(audit.approved_reference_outputs),
            set(audit.approved_task_ids),
        )
        self.assertTrue(all(audit.approved_reference_outputs.values()))
        self.assertEqual(
            audit.source_dataset_revision,
            "cc5b1d481b391181b89f7ced860308482e785463",
        )
        self.assertEqual(len(audit.source_dataset_sha256), 64)
        self.assertEqual(audit.quality_gate.minimum_token_recall, 0.15)
        self.assertEqual(audit.quality_gate.minimum_rouge_l_f1, 0.10)
        self.assertEqual(audit.quality_gate.maximum_metric_drop, 0.02)
        self.assertIn(
            "ccdfb6b6f98c55047ae81b705104dbd6<::>2",
            audit.approved_task_ids,
        )
        self.assertNotIn(
            "fc3e765d7ffc28ffc7ee7ea38fd3c73a<::>3",
            audit.approved_task_ids,
        )

    # Reject an audit when the caller expects a different deployed model.
    def test_rejects_model_mismatch(self):
        root = Path("results/mtrag-quality-calibration-v1")

        with self.assertRaisesRegex(ValueError, "provenance"):
            load_mtrag_manual_quality_audit(
                root / "manual-audit-v1.json",
                reference_artifact_path=(
                    root / "run-274842" / "reference-calibration.json"
                ),
                expected_model="different-model",
            )
