import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmarks.mtrag import (
    MTRAG_PROMPT_TEMPLATE_VERSION,
    MtragContext,
    MtragMessage,
    MtragTask,
    mtrag_conversation_split,
)
from benchmarks.prepare_reviewed_mtrag_counterfactual import (
    prepare_reviewed_mtrag_counterfactual,
)
from benchmarks.reviewed_mtrag_counterfactual import (
    REVIEW_STATUS,
    build_reviewed_mtrag_counterfactual_cases,
    load_reviewed_mtrag_reference_set,
)
from benchmarks.run_mtrag_counterfactual_pilot import main as run_main

MODEL = "Qwen/Qwen3-14B"
SOURCE_HASH = "a" * 64


def _conversation(prefix: str) -> str:
    for index in range(100):
        candidate = f"{prefix}-{index}"
        if mtrag_conversation_split(candidate).value in {"train", "validation"}:
            return candidate
    raise AssertionError("no usable conversation fixture")


def _pair(prefix: str, collection: str) -> tuple[MtragTask, MtragTask]:
    conversation = _conversation(prefix)
    context = (MtragContext("document-1", "Guide", "London is the answer."),)
    previous = MtragTask(
        f"{conversation}<::>1",
        conversation,
        1,
        collection,
        context,
        (MtragMessage("user", "Where should I travel?"),),
        "London.",
    )
    current = MtragTask(
        f"{conversation}<::>2",
        conversation,
        2,
        collection,
        context,
        (
            *previous.messages,
            MtragMessage("assistant", "I will check."),
            MtragMessage("user", "What does the guide say?"),
        ),
        "London.",
    )
    return previous, current


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _review_dir(root: Path, task: MtragTask) -> Path:
    split = mtrag_conversation_split(task.conversation_id).value
    output = f"Reviewed answer for {task.task_id}"
    reference = {
        "task_id": task.task_id,
        "split": split,
        "collection": task.collection,
        "prompt_token_count": 96,
        "cached_tokens": 0,
        "finish_reason": "stop",
        "output_text": output,
        "expected_answer": task.target_text,
    }
    hashes = {}
    for current_split in ("train", "validation", "test"):
        filename = f"{current_split}-reference-calibration.json"
        path = root / filename
        _write(
            path,
            {
                "analysis": "mtrag-reference-quality-calibration",
                "model": MODEL,
                "source_sha256": SOURCE_HASH,
                "rows": [reference] if split == current_split else [],
            },
        )
        hashes[f"/remote/{filename}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    review_id = f"review-{task.task_id}"
    _write(
        root / "blinded-reference-review.json",
        {
            "review": "mtrag-reference-correctness-blinded",
            "rows": [
                {
                    "review_id": review_id,
                    "model_answer": output,
                    "expected_answer": task.target_text,
                }
            ],
        },
    )
    _write(
        root / "assistant-verdicts-blinded.json",
        {
            "source_review": "blinded-reference-review.json",
            "rows": [
                {"review_id": review_id, "verdict": "pass", "reason": "correct"}
            ],
        },
    )
    _write(
        root / "blinded-reference-key.json",
        {
            "key": "mtrag-reference-correctness-identity-key",
            "source_artifact_sha256": hashes,
            "rows": [
                {
                    "review_id": review_id,
                    "task_id": task.task_id,
                    "split": split,
                    "collection": task.collection,
                    "source_artifact": f"/remote/{split}-reference-calibration.json",
                }
            ],
        },
    )
    return root


def _coverage_row(previous: MtragTask, current: MtragTask) -> dict:
    return {
        "conversation_id": current.conversation_id,
        "collection": current.collection,
        "previous_task_id": previous.task_id,
        "current_task_id": current.task_id,
        "shared_document_ids": ["document-1"],
        "reuse_opportunity": {
            "previous_token_count": 96,
            "current_token_count": 96,
            "native_cached_tokens": 16,
            "candidate_blocks": [
                {
                    "current_block_index": block,
                    "has_whole_source_block": True,
                    "previous_starts": [block * 16],
                }
                for block in (1, 2, 3, 4)
            ],
        },
    }


class ReviewedMtragPipelineTests(TestCase):
    def test_prepares_references_and_builds_cases(self):
        pairs = (_pair("one", "Cloud"), _pair("two", "Finance"))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            review_dirs = tuple(
                _review_dir(root / f"review-{index}", pair[1])
                for index, pair in enumerate(pairs)
            )
            coverage_path = root / "coverage.json"
            _write(
                coverage_path,
                {
                    "analysis": "mtrag-natural-block-coverage",
                    "model": MODEL,
                    "prompt_template_version": MTRAG_PROMPT_TEMPLATE_VERSION,
                    "source_sha256": SOURCE_HASH,
                    "block_size": 16,
                    "transitions": [_coverage_row(*pair) for pair in pairs],
                },
            )
            references, plan = prepare_reviewed_mtrag_counterfactual(
                review_dirs, coverage_path, include_repacking=True
            )
            references_path = root / "references.json"
            references_path.write_text(
                json.dumps(references, indent=2) + "\n", encoding="utf-8"
            )
            approved = load_reviewed_mtrag_reference_set(
                references_path, expected_model=MODEL
            )
            cases = build_reviewed_mtrag_counterfactual_cases(
                (*pairs[0], *pairs[1]), plan, approved, expected_model=MODEL
            )

        self.assertEqual(references["accepted_task_count"], 2)
        self.assertIsNone(plan["max_target_blocks"])
        self.assertTrue(plan["repacking_enabled"])
        self.assertEqual(plan["total_target_blocks"], 8)
        self.assertEqual(len(cases), 2)
        self.assertTrue(
            all(case.expected_candidate_block_indices == (1, 2, 3, 4) for case in cases)
        )
        self.assertTrue(
            all(case.target_block_indices == (1, 2, 3, 4) for case in cases)
        )

    def test_reviewed_runner_requires_the_approved_output(self):
        source = b"raw MTRAG source\n"
        source_hash = hashlib.sha256(source).hexdigest()
        references = SimpleNamespace(
            status=REVIEW_STATUS,
            source_dataset_sha256=source_hash,
            reference_outputs={"task-2": "approved"},
            manifest_sha256="b" * 64,
        )
        result = {"case_count": 1, "trial_count": 2, "invalid_trials": 0}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "RAG.jsonl"
            source_path.write_bytes(source)
            plan_path, references_path = root / "plan.json", root / "references.json"
            plan_path.write_text("{}", encoding="utf-8")
            references_path.write_text("{}", encoding="utf-8")
            summary_path = root / "summary.json"
            argv = [
                "runner", "--input", str(source_path), "--manifest", str(plan_path),
                "--references", str(references_path), "--model", MODEL,
                "--output-dir", str(root / "output"), "--summary-output",
                str(summary_path), "--run-id", "test-run",
            ]
            recorder = SimpleNamespace(path=root / "requests.jsonl")
            ledger = SimpleNamespace(is_complete=True, failed=0, started=8, path=recorder.path)
            module = "benchmarks.run_mtrag_counterfactual_pilot"
            with (
                patch.object(sys, "argv", argv),
                patch(f"{module}.load_reviewed_mtrag_reference_set", return_value=references),
                patch(f"{module}.load_mtrag_tasks", return_value=(SimpleNamespace(),)),
                patch(f"{module}.build_reviewed_mtrag_counterfactual_cases", return_value=(SimpleNamespace(),)),
                patch(f"{module}.RequestRecorder", return_value=recorder),
                patch(f"{module}.run_mtrag_counterfactual_cases", return_value=result) as run,
                patch(f"{module}.validate_ledger", return_value=ledger),
            ):
                run_main()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertFalse(run.call_args.kwargs["require_reference_output_match"])
        self.assertEqual(summary["recorded_requests"], 8)

    def test_runner_accepts_pending_live_references(self):
        source = b"raw MTRAG source\n"
        source_hash = hashlib.sha256(source).hexdigest()
        result = {"case_count": 1, "trial_count": 2, "invalid_trials": 0}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "RAG.jsonl"
            source_path.write_bytes(source)
            plan_path = root / "plan.json"
            _write(
                plan_path,
                {
                    "reference_status": "generated_in_trial_pending_review",
                    "source_dataset_sha256": source_hash,
                    "transitions": [{"current_task_id": "task-2"}],
                },
            )
            summary_path = root / "summary.json"
            argv = [
                "runner", "--input", str(source_path), "--manifest", str(plan_path),
                "--live-references", "--model", MODEL, "--output-dir",
                str(root / "output"), "--summary-output", str(summary_path),
            ]
            recorder = SimpleNamespace(path=root / "requests.jsonl")
            ledger = SimpleNamespace(
                is_complete=True, failed=0, started=8, path=recorder.path
            )
            module = "benchmarks.run_mtrag_counterfactual_pilot"
            with (
                patch.object(sys, "argv", argv),
                patch(f"{module}.load_mtrag_tasks", return_value=(SimpleNamespace(),)),
                patch(
                    f"{module}.build_mtrag_counterfactual_cases",
                    return_value=(SimpleNamespace(),),
                ),
                patch(f"{module}.RequestRecorder", return_value=recorder),
                patch(
                    f"{module}.run_mtrag_counterfactual_cases", return_value=result
                ) as run,
                patch(f"{module}.validate_ledger", return_value=ledger),
            ):
                run_main()

        self.assertIsNone(run.call_args.kwargs["reference_outputs"])
        self.assertFalse(run.call_args.kwargs["require_reference_quality"])

    def test_runpod_launcher_contract(self):
        script = Path(__file__).resolve().parents[1] / "benchmarks" / "run_runpod_qwen3_14b_counterfactual.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        content = script.read_text(encoding="utf-8")
        self.assertIn('[[ "$CHATRAG_EVALUATION" == 1 ]] || test -s "$PLAN"', content)
        for token in (
            "Qwen/Qwen3-14B",
            '== *"A40"*',
            'plan["total_target_blocks"] == plan["total_testable_blocks"]',
            'row["target_block_indices"] == row["testable_block_indices"]',
            'summary["trial_count"] + summary["skipped_target_blocks"]',
            "CACHESELECT_COUNTERFACTUAL_START_CASE",
            "CACHESELECT_COUNTERFACTUAL_MAX_CASES",
            "CACHESELECT_MLP_SMOKE",
            "CACHESELECT_MLP_EVALUATION",
            "CACHESELECT_CHATRAG_EVALUATION",
            "CACHESELECT_CHATRAG_INPUT",
            "CACHESELECT_CHATRAG_MANIFEST",
            "CACHESELECT_NATIVE_APC_EVALUATION",
            "qwen3-chatrag-native-apc",
            "CACHESELECT_MLP_MODEL",
            "CACHESELECT_CORRECT_KV_POSITIONS",
            "CACHESELECT_MIN_REUSE_SPAN_BLOCKS",
            "CACHESELECT_MIN_REPACKED_REUSE_SPAN_BLOCKS",
            "CACHESELECT_FULL_MTRAG_COUNTERFACTUAL",
            "CACHESELECT_COUNTERFACTUAL_START_BATCH",
            "CACHESELECT_COUNTERFACTUAL_BLOCKS_PER_BATCH",
            "CACHESELECT_COUNTERFACTUAL_EXCLUDE_COMPLETED",
            "mtrag_completed_transitions.json",
            '--start-case "$START_CASE"',
            '--max-cases "$MAX_CASES"',
            "RunPod checkout must be clean",
            "--enable-cacheselect",
            "--cacheselect-repair-selector full_block",
            "--cacheselect-repair-selector mlp",
            "--cacheselect-mlp-model",
            "--cacheselect-execute-partial-reuse",
            "--cacheselect-repack-partial-reuse",
            "--cacheselect-correct-kv-positions",
            "--cacheselect-min-reuse-span-blocks",
            "--cacheselect-min-repacked-reuse-span-blocks",
            "python -m benchmarks.run_mtrag_counterfactual_pilot",
            "--live-references",
            "live-reference-calibration.json",
            "--policy-evaluation",
            "--native-apc-policy",
            "--input-format chatrag",
            "--policy-split test",
            "--max-completion-tokens 2048",
            "prepare_mtrag_counterfactual_review",
            "artifacts.tar.gz",
        ):
            with self.subTest(token=token):
                self.assertIn(token, content)

    def test_imperial_rope_training_launcher_contract(self):
        script = Path(__file__).resolve().parents[1] / "benchmarks" / "run_imperial_qwen3_14b_rope_training_200.slurm"
        subprocess.run(["bash", "-n", str(script)], check=True)
        content = script.read_text(encoding="utf-8")
        for token in (
            "#SBATCH --partition=a40",
            "#SBATCH --gres=gpu:nvidia_a40:1",
            "CACHESELECT_COUNTERFACTUAL_MAX_CASES=200",
            "CACHESELECT_COUNTERFACTUAL_EXCLUDE_COMPLETED=0",
            "CACHESELECT_CORRECT_KV_POSITIONS=1",
            "qwen3-mtrag-rope-training-200-v1",
            "run_imperial_qwen3_14b_counterfactual.slurm",
        ):
            with self.subTest(token=token):
                self.assertIn(token, content)

    def test_imperial_launcher_contract(self):
        script = Path(__file__).resolve().parents[1] / "benchmarks" / "run_imperial_qwen3_14b_counterfactual.slurm"
        subprocess.run(["bash", "-n", str(script)], check=True)
        content = script.read_text(encoding="utf-8")
        for token in (
            "#SBATCH --partition=a40",
            "#SBATCH --gres=gpu:nvidia_a40:1",
            "#SBATCH --time=1-12:00:00",
            "qwen3-mtrag-v3",
            'CACHESELECT_COUNTERFACTUAL_START_CASE:-1',
            "CACHESELECT_SERVER_PORT",
            'CACHESELECT_EXECUTION_PLATFORM="imperial-a40"',
            "run_runpod_qwen3_14b_counterfactual.sh",
        ):
            with self.subTest(token=token):
                self.assertIn(token, content)
