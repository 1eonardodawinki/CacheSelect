import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmarks.run_mtrag_calibration import main


class RunMtragCalibrationCommandTests(TestCase):
    # Keep the split launcher valid and prevent it from opening the test manifest.
    def test_slurm_launcher_runs_only_train_and_validation(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "run_mtrag_reference_splits.slurm"
        )

        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        content = script.read_text(encoding="utf-8")
        for token in (
            "SPLITS=(train validation)",
            "vllm serve",
            "--max-model-len 8192",
            "--enable-cacheselect",
            "python -m benchmarks.run_mtrag_calibration",
            "train-reference-calibration.json",
            "validation-reference-calibration.json",
        ):
            with self.subTest(token=token):
                self.assertIn(token, content)
        self.assertNotIn("test-manifest.json", content)

    # Wire source verification, recording, execution, and artifact saving together.
    def test_saves_complete_calibration_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "RAG.jsonl"
            source.write_bytes(b"public-mtrag-rows\n")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                ),
                encoding="utf-8",
            )
            output = root / "calibration.json"
            recorder = SimpleNamespace(path=root / "requests.jsonl")
            cases = (object(), object())
            arguments = [
                "run_mtrag_calibration",
                "--input",
                str(source),
                "--manifest",
                str(manifest),
                "--model",
                "test-model",
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "benchmarks.run_mtrag_calibration.load_mtrag_tasks",
                    return_value=(object(),),
                ),
                patch(
                    "benchmarks.run_mtrag_calibration."
                    "build_mtrag_reference_calibration_cases",
                    return_value=cases,
                ),
                patch(
                    "benchmarks.run_mtrag_calibration.RequestRecorder",
                    return_value=recorder,
                ),
                patch(
                    "benchmarks.run_mtrag_calibration.run_mtrag_reference_calibration",
                    return_value={"request_count": 2, "rows": []},
                ) as run_calibration,
                patch(
                    "benchmarks.run_mtrag_calibration.validate_ledger",
                    return_value=SimpleNamespace(
                        is_complete=True,
                        failed=0,
                        completed=2,
                        path=recorder.path,
                    ),
                ),
            ):
                main()

            artifact = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(artifact["request_count"], 2)
        self.assertEqual(artifact["model"], "test-model")
        self.assertEqual(artifact["request_ledger"], str(recorder.path))
        self.assertIs(run_calibration.call_args.kwargs["recorder"], recorder)
