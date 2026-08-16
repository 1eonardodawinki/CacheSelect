from unittest import TestCase

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import mtrag_conversation_split
from benchmarks.mtrag_pilot import select_audited_mtrag_counterfactual_pilot


# Find a deterministic training conversation for selection fixtures.
def _training_conversation_id(prefix: str) -> str:
    for index in range(1000):
        candidate = f"{prefix}-{index}"
        if mtrag_conversation_split(candidate) is DatasetSplit.TRAIN:
            return candidate
    raise AssertionError("could not find a training conversation")


# Build one coverage row with the requested aligned candidate blocks.
def _row(collection: str, suffix: str, blocks: tuple[int, ...]) -> dict:
    conversation = _training_conversation_id(suffix)
    return {
        "conversation_id": conversation,
        "collection": collection,
        "previous_task_id": f"{conversation}-1",
        "current_task_id": f"{conversation}-2",
        "shared_document_ids": [f"document-{suffix}"],
        "reuse_opportunity": {
            "current_token_count": 160,
            "native_cached_tokens": 16,
            "candidate_blocks": [
                {
                    "current_block_index": index,
                    "has_whole_source_block": True,
                }
                for index in blocks
            ],
        },
    }


class MtragPilotTests(TestCase):
    # Keep full candidate sets while selecting only two isolated GPU targets.
    def test_selects_audited_target_subsets(self):
        cloud = _row("Cloud", "a", (1, 2, 3, 4))
        finance = _row("Finance", "b", (1, 2, 4))
        coverage = {
            "schema_version": 1,
            "analysis": "mtrag-natural-block-coverage",
            "block_size": 16,
            "transitions": [cloud, finance],
        }
        approved = frozenset(
            {cloud["current_task_id"], finance["current_task_id"]}
        )

        result = select_audited_mtrag_counterfactual_pilot(
            coverage,
            approved_task_ids=approved,
            quality_calibration_id="manual-v1",
        )

        self.assertEqual(result["transition_count"], 2)
        self.assertEqual(result["total_testable_blocks"], 7)
        self.assertEqual(result["total_target_blocks"], 4)
        for row in result["transitions"]:
            self.assertLessEqual(len(row["target_block_indices"]), 2)
            self.assertTrue(
                set(row["target_block_indices"]).issubset(
                    row["testable_block_indices"]
                )
            )

    # Reject a candidate whose final full block would be used for output logits.
    def test_excludes_output_producing_block(self):
        row = _row("Cloud", "output", (1, 9))
        coverage = {
            "schema_version": 1,
            "analysis": "mtrag-natural-block-coverage",
            "block_size": 16,
            "transitions": [row],
        }

        result = select_audited_mtrag_counterfactual_pilot(
            coverage,
            approved_task_ids=frozenset({row["current_task_id"]}),
            quality_calibration_id="manual-v1",
        )

        selected = result["transitions"][0]
        self.assertEqual(selected["aligned_candidate_block_indices"], [1, 9])
        self.assertEqual(selected["testable_block_indices"], [1])
        self.assertEqual(selected["excluded_output_block_index"], 9)
