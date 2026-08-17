import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.block_dataset import DatasetSplit
from benchmarks.freeze_mtrag_data_splits import freeze_mtrag_data_splits
from benchmarks.mtrag import mtrag_conversation_split


# Find one conversation assigned to a requested frozen split.
def _conversation(split: DatasetSplit) -> str:
    for index in range(1000):
        candidate = f"conversation-{split.value}-{index}"
        if mtrag_conversation_split(candidate) is split:
            return candidate
    raise AssertionError("could not find split fixture")


# Build one executable coverage row for a compact split fixture.
def _row(split: DatasetSplit) -> dict:
    conversation = _conversation(split)
    return {
        "conversation_id": conversation,
        "collection": "RAG",
        "current_task_id": f"{conversation}<::>2",
        "shared_document_ids": [],
        "reuse_opportunity": {
            "current_token_count": 64,
            "candidate_blocks": [
                {"current_block_index": 1, "has_whole_source_block": True}
            ],
        },
    }


class FreezeMtragDataSplitsTests(TestCase):
    # Freeze disjoint manifests and leave the test set explicitly unopened.
    def test_freezes_all_conversation_splits(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "RAG.jsonl"
            source.write_text("fixture\n", encoding="utf-8")
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "analysis": "mtrag-natural-block-coverage",
                        "block_size": 16,
                        "transitions": [_row(split) for split in DatasetSplit],
                    }
                ),
                encoding="utf-8",
            )
            exclusion = root / "previous.json"
            exclusion.write_text(
                json.dumps(
                    {
                        "source": "MTRAG",
                        "source_revision": "revision",
                        "source_sha256": source_sha256,
                        "tasks": [{"conversation_id": "already-used"}],
                    }
                ),
                encoding="utf-8",
            )

            plan = freeze_mtrag_data_splits(
                coverage_path=coverage,
                source_dataset_path=source,
                exclusion_manifest_path=exclusion,
                output_dir=root / "output",
                task_counts={split: 1 for split in DatasetSplit},
                existing_training_transitions=0,
            )

            manifests = [
                json.loads(Path(path).read_text(encoding="utf-8"))
                for path in plan["manifests"].values()
            ]
        self.assertEqual(plan["test_status"], "frozen_unopened")
        self.assertEqual({manifest["split"] for manifest in manifests}, {
            "train",
            "validation",
            "test",
        })
        conversation_ids = {
            manifest["tasks"][0]["conversation_id"] for manifest in manifests
        }
        self.assertEqual(len(conversation_ids), 3)
