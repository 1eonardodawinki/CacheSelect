from unittest import TestCase
from unittest.mock import patch

from benchmarks.chatrag import chatrag_tasks
from benchmarks.mtrag_coverage import analyze_mtrag_coverage


def _row(messages, answer, *, document="doc"):
    return {
        "document": document,
        "messages": messages,
        "answers": [answer],
        "ctxs": [
            {"title": "Shared", "text": "The same passage."},
            {"title": "Extra", "text": "Another passage."},
        ],
    }


class ChatRagTests(TestCase):
    def test_adapts_growing_histories_to_adjacent_tasks(self):
        first = _row([{"role": "user", "content": "First?"}], "First answer")
        second = _row(
            [
                {"role": "user", "content": "First?"},
                {"role": "assistant", "content": "First answer"},
                {"role": "assistant", "content": "A follow-up prompt"},
                {"role": "user", "content": "Second?"},
            ],
            "Second answer",
        )

        tasks = chatrag_tasks([first, second], subset="doc2dial", max_contexts=1)

        self.assertEqual([task.turn for task in tasks], [1, 2])
        self.assertEqual(tasks[0].conversation_id, tasks[1].conversation_id)
        self.assertEqual(tasks[0].contexts, tasks[1].contexts)
        self.assertEqual(len(tasks[0].contexts), 1)

    def test_reuses_existing_block_coverage_analysis(self):
        rows = [
            _row([{"role": "user", "content": "First?"}], "First answer"),
            _row(
                [
                    {"role": "user", "content": "First?"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second?"},
                ],
                "Second answer",
            ),
        ]
        tasks = chatrag_tasks(rows, subset="doc2dial")
        with patch(
            "benchmarks.mtrag_coverage.rendered_chat_token_ids",
            side_effect=[list(range(12)), [0, 1, 99, 98, *range(4, 12)]],
        ):
            result = analyze_mtrag_coverage(tasks, tokenizer=object(), block_size=4)

        self.assertEqual(result["conversation_count"], 1)
        self.assertEqual(result["transition_count"], 1)
        self.assertEqual(result["candidate_block_count"], 2)
