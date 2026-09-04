import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmarks.block_dataset import DatasetSplit
from benchmarks.chatrag import chatrag_conversation_split, chatrag_tasks
from benchmarks.chatrag_policy import (
    build_chatrag_policy_cases,
    freeze_chatrag_policy_manifest,
)
from benchmarks.mtrag_coverage import analyze_mtrag_coverage
from benchmarks.run_mtrag_counterfactual_pilot import main as run_main


def _rows():
    context = [{"title": "Shared", "text": "A repeated passage."}]
    return [
        {
            "document": "doc",
            "messages": [{"role": "user", "content": "First?"}],
            "answers": ["First answer"],
            "ctxs": context,
        },
        {
            "document": "doc",
            "messages": [
                {"role": "user", "content": "First?"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second?"},
            ],
            "answers": ["Second answer"],
            "ctxs": context,
        },
    ]


class ChatRagPolicyTests(TestCase):
    def test_freezes_and_resolves_complete_conversations(self):
        tasks = chatrag_tasks(_rows(), subset="doc2dial")
        with patch(
            "benchmarks.mtrag_coverage.rendered_chat_token_ids",
            side_effect=[list(range(12)), [0, 1, 99, 98, *range(4, 12)]],
        ):
            coverage = analyze_mtrag_coverage(tasks, tokenizer=object(), block_size=4)
        coverage.update(
            analysis="chatrag-natural-block-coverage",
            model="test-model",
            subset="doc2dial",
            max_contexts=5,
        )
        coverage = json.loads(json.dumps(coverage))

        manifest = freeze_chatrag_policy_manifest(
            coverage,
            source_sha256="a" * 64,
        )
        split = chatrag_conversation_split(tasks[0].conversation_id)
        cases = build_chatrag_policy_cases(
            tasks,
            manifest,
            split=split,
            expected_model="test-model",
        )

        self.assertEqual(manifest["transition_count"], 1)
        self.assertEqual(cases[0].trace.workload, "chatrag_rag")
        self.assertTrue(cases[0].trace.trace_id.startswith("chatrag:"))

    def test_existing_runner_accepts_chatrag_policy_inputs(self):
        source = b"parquet"
        source_hash = hashlib.sha256(source).hexdigest()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "doc2dial.parquet"
            source_path.write_bytes(source)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_dataset_sha256": source_hash,
                        "max_contexts": 5,
                    }
                ),
                encoding="utf-8",
            )
            summary_path = root / "summary.json"
            argv = [
                "runner",
                "--input", str(source_path),
                "--input-format", "chatrag",
                "--manifest", str(manifest_path),
                "--live-references",
                "--policy-evaluation",
                "--policy-split", "train",
                "--model", "test-model",
                "--output-dir", str(root / "output"),
                "--summary-output", str(summary_path),
            ]
            case = SimpleNamespace(split=DatasetSplit.TRAIN)
            recorder = SimpleNamespace(path=root / "requests.jsonl")
            ledger = SimpleNamespace(
                is_complete=True,
                failed=0,
                started=3,
                path=recorder.path,
            )
            result = {"case_count": 1}
            module = "benchmarks.run_mtrag_counterfactual_pilot"
            with (
                patch.object(sys, "argv", argv),
                patch(f"{module}.load_chatrag_tasks", return_value=(object(),)),
                patch(f"{module}.build_chatrag_policy_cases", return_value=(case,)),
                patch(f"{module}.RequestRecorder", return_value=recorder),
                patch(f"{module}.run_mtrag_policy_cases", return_value=result) as run,
                patch(f"{module}.validate_ledger", return_value=ledger),
            ):
                run_main()
            summary = json.loads(summary_path.read_text())

        self.assertIsNone(run.call_args.kwargs["reference_outputs"])
        self.assertEqual(summary["policy_split"], "train")
