import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmarks.run_mtrag_counterfactual_pilot import main


class RunMtragCounterfactualPilotTests(TestCase):
    # Keep the cluster launcher valid and configured for guarded partial reuse.
    def test_slurm_launcher_contract(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "run_mtrag_counterfactual_pilot.slurm"
        )

        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        content = script.read_text(encoding="utf-8")
        for token in (
            "vllm serve",
            "--enable-prefix-caching",
            "--enable-cacheselect",
            "--cacheselect-repair-selector full_block",
            "--cacheselect-execute-partial-reuse",
            "--no-enable-chunked-prefill",
            "--enforce-eager",
            "python -m benchmarks.run_mtrag_counterfactual_pilot",
            "--max-completion-tokens 384",
            "CACHESELECT_SERVER_START_TIMEOUT_SECONDS",
            "SERVER_START_TIMEOUT_SECONDS:-3600",
            "Still waiting for vLLM",
        ):
            with self.subTest(token=token):
                self.assertIn(token, content)

    # Wire validated source, audit, cases and one recorder into the pilot runner.
    def test_runs_and_saves_audited_pilot(self):
        source = b"mtrag source\n"
        source_sha256 = hashlib.sha256(source).hexdigest()
        audit = SimpleNamespace(
            quality_gate=SimpleNamespace(calibration_id="manual-v1"),
            approved_task_ids=frozenset({"task-2"}),
            approved_reference_outputs={"task-2": "answer"},
            source_dataset_revision="revision-1",
            source_dataset_sha256=source_sha256,
        )
        cases = (SimpleNamespace(name="first"), SimpleNamespace(name="second"))
        result = {
            "schema_version": 1,
            "experiment": "mtrag-counterfactual-pilot",
            "case_count": 1,
            "trial_count": 2,
            "valid_training_rows": 1,
            "invalid_trials": 0,
            "abstained_trials": 1,
            "repair_labels": 0,
            "reuse_labels": 1,
            "cases": [],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "RAG.jsonl"
            manifest_path = root / "manifest.json"
            audit_path = root / "audit.json"
            reference_path = root / "reference.json"
            summary_path = root / "summary.json"
            input_path.write_bytes(source)
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_sha256": source_sha256,
                        "source_revision": "revision-1",
                    }
                ),
                encoding="utf-8",
            )
            audit_path.write_text("{}", encoding="utf-8")
            reference_path.write_text("{}", encoding="utf-8")
            argv = [
                "run_mtrag_counterfactual_pilot",
                "--input",
                str(input_path),
                "--manifest",
                str(manifest_path),
                "--audit",
                str(audit_path),
                "--reference-artifact",
                str(reference_path),
                "--model",
                "test-model",
                "--output-dir",
                str(root / "output"),
                "--summary-output",
                str(summary_path),
                "--run-id",
                "test-run",
                "--start-case",
                "2",
            ]
            recorder = SimpleNamespace(path=root / "requests.jsonl")
            ledger = SimpleNamespace(
                is_complete=True,
                failed=0,
                started=8,
                path=recorder.path,
            )
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "benchmarks.run_mtrag_counterfactual_pilot."
                    "load_mtrag_manual_quality_audit",
                    return_value=audit,
                ),
                patch(
                    "benchmarks.run_mtrag_counterfactual_pilot.load_mtrag_tasks",
                    return_value=(SimpleNamespace(),),
                ),
                patch(
                    "benchmarks.run_mtrag_counterfactual_pilot."
                    "build_mtrag_counterfactual_cases",
                    return_value=cases,
                ) as build_cases,
                patch(
                    "benchmarks.run_mtrag_counterfactual_pilot.RequestRecorder",
                    return_value=recorder,
                ),
                patch(
                    "benchmarks.run_mtrag_counterfactual_pilot."
                    "run_mtrag_counterfactual_cases",
                    return_value=result,
                ) as run_cases,
                patch(
                    "benchmarks.run_mtrag_counterfactual_pilot.validate_ledger",
                    return_value=ledger,
                ),
            ):
                main()

            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(build_cases.call_args.kwargs["expected_model"], "test-model")
        self.assertEqual(run_cases.call_args.args[0], cases[1:])
        self.assertEqual(run_cases.call_args.kwargs["source_case_start"], 2)
        self.assertEqual(run_cases.call_args.kwargs["max_completion_tokens"], 384)
        self.assertIs(run_cases.call_args.kwargs["recorder"], recorder)
        self.assertEqual(summary["trial_count"], 2)
        self.assertEqual(summary["recorded_requests"], 8)

    # Reject a raw dataset that differs from the frozen pilot provenance.
    def test_rejects_source_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "RAG.jsonl"
            manifest_path = root / "manifest.json"
            audit_path = root / "audit.json"
            reference_path = root / "reference.json"
            input_path.write_text("different", encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"source_sha256": "a" * 64}), encoding="utf-8"
            )
            audit_path.write_text("{}", encoding="utf-8")
            reference_path.write_text("{}", encoding="utf-8")
            argv = [
                "run_mtrag_counterfactual_pilot",
                "--input",
                str(input_path),
                "--manifest",
                str(manifest_path),
                "--audit",
                str(audit_path),
                "--reference-artifact",
                str(reference_path),
                "--model",
                "test-model",
                "--output-dir",
                str(root / "output"),
            ]
            audit = SimpleNamespace(
                source_dataset_sha256="a" * 64,
                source_dataset_revision="revision-1",
            )
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "benchmarks.run_mtrag_counterfactual_pilot."
                    "load_mtrag_manual_quality_audit",
                    return_value=audit,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    main()
