from unittest import TestCase

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import mtrag_conversation_split
from benchmarks.mtrag_reference_expansion import (
    select_mtrag_reference_expansion,
)


# Find stable conversation identities for one requested split.
def _conversation(split: DatasetSplit, index: int) -> str:
    for candidate_index in range(index * 100, (index + 1) * 100):
        candidate = f"expansion-{split.value}-{candidate_index}"
        if mtrag_conversation_split(candidate) is split:
            return candidate
    raise AssertionError("could not find split fixture")


# Build one executable transition row for selection tests.
def _row(conversation: str, collection: str, turn: int) -> dict:
    return {
        "conversation_id": conversation,
        "collection": collection,
        "current_task_id": f"{conversation}<::>{turn}",
        "shared_document_ids": ["document"],
        "reuse_opportunity": {
            "current_token_count": 128,
            "candidate_blocks": [
                {"current_block_index": 2, "has_whole_source_block": True}
            ],
        },
    }


def _coverage(rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "analysis": "mtrag-natural-block-coverage",
        "prompt_template_version": 1,
        "model": "Qwen/Qwen3-14B",
        "tokenizer_class": "Qwen2Tokenizer",
        "block_size": 16,
        "transitions": rows,
    }


class MtragReferenceExpansionTests(TestCase):
    # Balance domains and never repeat a previously completed task.
    def test_selects_balanced_unused_tasks(self):
        rows = []
        excluded = None
        for collection_index, collection in enumerate(("A", "B", "C", "D")):
            conversation = _conversation(DatasetSplit.TRAIN, collection_index)
            rows.extend(
                (
                    _row(conversation, collection, 2),
                    _row(conversation, collection, 3),
                )
            )
            if collection == "A":
                excluded = rows[-2]["current_task_id"]

        result = select_mtrag_reference_expansion(
            _coverage(rows),
            split=DatasetSplit.TRAIN,
            excluded_task_ids=frozenset({excluded}),
            prior_conversation_counts={},
            task_count=4,
        )

        self.assertEqual(result["task_count"], 4)
        self.assertEqual(set(result["collection_task_counts"].values()), {1})
        self.assertNotIn(excluded, {row["task_id"] for row in result["tasks"]})

    # Prefer distinct conversations before selecting another turn from one.
    def test_maximizes_conversation_diversity(self):
        first = _conversation(DatasetSplit.VALIDATION, 0)
        second = _conversation(DatasetSplit.VALIDATION, 1)
        rows = [_row(first, "A", 2), _row(first, "A", 3), _row(second, "A", 2)]

        result = select_mtrag_reference_expansion(
            _coverage(rows),
            split=DatasetSplit.VALIDATION,
            excluded_task_ids=frozenset(),
            prior_conversation_counts={first: 1},
            task_count=2,
        )

        self.assertEqual(
            {row["conversation_id"] for row in result["tasks"]},
            {first, second},
        )
