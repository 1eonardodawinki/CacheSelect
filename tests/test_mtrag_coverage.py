from unittest import TestCase
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from benchmarks.analyze_mtrag_coverage import main
from benchmarks.mtrag import MtragContext, MtragMessage, MtragTask
from benchmarks.mtrag_coverage import analyze_mtrag_coverage


# Build one typed task without involving JSONL parsing in coverage tests.
def _task(task_id: str, turn: int, document_id: str) -> MtragTask:
    return MtragTask(
        task_id=task_id,
        conversation_id="conversation-1",
        turn=turn,
        collection="Cloud",
        contexts=(MtragContext(document_id, "Title", f"Text for {document_id}"),),
        messages=(MtragMessage("user", f"Question {turn}?"),),
        target_text=f"Answer {turn}.",
    )


class MtragCoverageTests(TestCase):
    # Verify summary counts distinguish passage overlap and executable blocks.
    def test_analyzes_natural_transition_coverage(self):
        tasks = (_task("task-1", 1, "shared"), _task("task-2", 2, "shared"))
        previous_tokens = list(range(1, 13))
        current_tokens = [1, 2, 99, 98, *range(5, 13)]

        with patch(
            "benchmarks.mtrag_coverage.rendered_chat_token_ids",
            side_effect=[previous_tokens, current_tokens],
        ):
            result = analyze_mtrag_coverage(tasks, tokenizer=object(), block_size=4)

        self.assertEqual(result["task_count"], 2)
        self.assertEqual(result["transition_count"], 1)
        self.assertEqual(result["shared_document_transition_count"], 1)
        self.assertEqual(result["candidate_transition_count"], 1)
        self.assertEqual(result["whole_source_block_count"], 2)
        self.assertEqual(result["candidate_tokens"], 8)
        row = result["transitions"][0]
        self.assertEqual(row["shared_document_ids"], ["shared"])
        self.assertEqual(row["reuse_opportunity"]["native_cached_tokens"], 0)
        self.assertEqual(row["reuse_opportunity"]["prefix_alignment_loss_tokens"], 2)

    # Reject invalid geometry before running tokenizer-dependent analysis.
    def test_rejects_invalid_block_size(self):
        with self.assertRaisesRegex(ValueError, "block_size"):
            analyze_mtrag_coverage(
                (_task("task-1", 1, "document"),),
                tokenizer=object(),
                block_size=0,
            )

    # Verify the CLI records tokenizer provenance beside the coverage result.
    def test_coverage_command_saves_auditable_artifact(self):
        tokenizer = MagicMock()
        result = {
            "transition_count": 3,
            "whole_source_candidate_transition_count": 2,
        }
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "coverage.json"
            arguments = [
                "analyze_mtrag_coverage",
                "--input",
                "rag.jsonl",
                "--output",
                str(output),
                "--model",
                "test-model",
            ]
            with (
                patch("sys.argv", arguments),
                patch(
                    "benchmarks.analyze_mtrag_coverage.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ) as load_tokenizer,
                patch(
                    "benchmarks.analyze_mtrag_coverage.load_mtrag_tasks",
                    return_value=(object(),),
                ),
                patch(
                    "benchmarks.analyze_mtrag_coverage.analyze_mtrag_coverage",
                    return_value=result,
                ),
            ):
                main()

            artifact = json.loads(output.read_text())

        load_tokenizer.assert_called_once_with(
            "test-model",
            local_files_only=False,
        )
        self.assertEqual(artifact["input"], "rag.jsonl")
        self.assertEqual(artifact["model"], "test-model")
        self.assertEqual(artifact["transition_count"], 3)
