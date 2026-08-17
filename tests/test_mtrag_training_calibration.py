from unittest import TestCase

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import mtrag_conversation_split
from benchmarks.mtrag_calibration import select_mtrag_training_calibration


# Find a stable conversation identity assigned to the requested split.
def _conversation(prefix: str, split: DatasetSplit) -> str:
    for index in range(1000):
        candidate = f"{prefix}-{index}"
        if mtrag_conversation_split(candidate) is split:
            return candidate
    raise AssertionError("could not find a conversation for the requested split")


# Build one compact natural-coverage row with an executable candidate block.
def _coverage_row(collection: str, conversation: str, turn: int) -> dict:
    return {
        "conversation_id": conversation,
        "collection": collection,
        "previous_task_id": f"{conversation}<::>{turn - 1}",
        "current_task_id": f"{conversation}<::>{turn}",
        "shared_document_ids": [f"document-{conversation}"],
        "reuse_opportunity": {
            "current_token_count": 160,
            "native_cached_tokens": 16,
            "candidate_blocks": [
                {"current_block_index": 2, "has_whole_source_block": True}
            ],
        },
    }


class MtragTrainingCalibrationTests(TestCase):
    # Select one unseen training conversation per collection without leakage.
    def test_selects_balanced_unseen_conversations(self):
        excluded = _conversation("excluded", DatasetSplit.TRAIN)
        repeated = _conversation("repeated", DatasetSplit.TRAIN)
        rows = [
            _coverage_row("Cloud", excluded, 2),
            _coverage_row("Cloud", repeated, 2),
            _coverage_row("Cloud", repeated, 3),
            _coverage_row("Cloud", _conversation("cloud", DatasetSplit.TRAIN), 2),
            _coverage_row("Finance", _conversation("finance", DatasetSplit.TRAIN), 2),
            _coverage_row(
                "Finance", _conversation("validation", DatasetSplit.VALIDATION), 2
            ),
        ]
        # A changing document set may still leave reusable history or template blocks.
        rows[4]["shared_document_ids"] = []
        coverage = {
            "schema_version": 1,
            "analysis": "mtrag-natural-block-coverage",
            "prompt_template_version": 1,
            "model": "test-model",
            "tokenizer_class": "test-tokenizer",
            "block_size": 16,
            "transitions": rows,
        }

        result = select_mtrag_training_calibration(
            coverage,
            excluded_conversation_ids=frozenset({excluded}),
            per_collection=1,
        )

        self.assertEqual(result["task_count"], 2)
        self.assertEqual(result["collection_count"], 2)
        self.assertEqual({row["collection"] for row in result["tasks"]}, {
            "Cloud",
            "Finance",
        })
        self.assertEqual(len({row["conversation_id"] for row in result["tasks"]}), 2)
        self.assertTrue(all(row["split"] == "train" for row in result["tasks"]))
        self.assertNotIn(excluded, {row["conversation_id"] for row in result["tasks"]})
