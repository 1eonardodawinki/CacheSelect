import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.mtrag import (
    MTRAG_SYSTEM_PROMPT,
    adjacent_mtrag_transitions,
    load_mtrag_tasks,
    mtrag_source_sha256,
    render_mtrag_messages,
)


# Build one minimal public-schema MTRAG row for focused importer tests.
def _task(conversation: str, turn: int, document_text: str = "Shared text."):
    history = [{"speaker": "user", "text": "First question?"}]
    if turn == 2:
        history.extend(
            [
                {"speaker": "agent", "text": "First answer."},
                {"speaker": "user", "text": "Follow-up question?"},
            ]
        )
    return {
        "task_id": f"{conversation}-{turn}",
        "conversation_id": conversation,
        "turn": str(turn),
        "task_type": "rag",
        "Collection": "Cloud",
        "contexts": [
            {
                "document_id": "shared-document",
                "title": "Shared title",
                "text": document_text,
            }
        ],
        "input": history,
        "targets": [{"speaker": "agent", "text": f"Answer {turn}."}],
    }


class MtragTests(TestCase):
    # Produce the same source identity during selection and later GPU execution.
    def test_hashes_raw_source_bytes(self):
        content = b"public-mtrag-rows\n"
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rag.jsonl"
            path.write_bytes(content)

            digest = mtrag_source_sha256(path)

        self.assertEqual(digest, hashlib.sha256(content).hexdigest())

    # Verify public JSONL fields become the typed subset CacheSelect needs.
    def test_loads_typed_mtrag_tasks(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rag.jsonl"
            path.write_text(json.dumps(_task("alpha", 2)) + "\n")
            tasks = load_mtrag_tasks(path)

        self.assertEqual(tasks[0].task_id, "alpha-2")
        self.assertEqual(tasks[0].turn, 2)
        self.assertEqual(tasks[0].messages[1].role, "assistant")
        self.assertEqual(tasks[0].contexts[0].title, "Shared title")
        self.assertEqual(tasks[0].target_text, "Answer 2.")

    # Verify turn sorting creates pairs within, but never across, conversations.
    def test_pairs_consecutive_conversation_turns(self):
        rows = [_task("beta", 1), _task("alpha", 2), _task("alpha", 1)]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rag.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            tasks = load_mtrag_tasks(path)

        transitions = adjacent_mtrag_transitions(tasks)

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0][0].task_id, "alpha-1")
        self.assertEqual(transitions[0][1].task_id, "alpha-2")

    # Verify rendering retains history and augments only the current question.
    def test_renders_deterministic_rag_chat_messages(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rag.jsonl"
            path.write_text(json.dumps(_task("alpha", 2)) + "\n")
            task = load_mtrag_tasks(path)[0]

        messages = render_mtrag_messages(task)

        self.assertEqual(
            messages[0], {"role": "system", "content": MTRAG_SYSTEM_PROMPT}
        )
        self.assertEqual(messages[1]["content"], "First question?")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertIn("document_id=shared-document", messages[3]["content"])
        self.assertIn("Shared text.", messages[3]["content"])
        self.assertTrue(messages[3]["content"].endswith("Follow-up question?"))

    # Reject a document identity that changes text across requests.
    def test_rejects_inconsistent_repeated_document_text(self):
        rows = [_task("alpha", 1), _task("alpha", 2, "Changed text.")]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rag.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            with self.assertRaisesRegex(ValueError, "inconsistent text"):
                load_mtrag_tasks(path)
