import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.block_dataset import DatasetSplit
from benchmarks.freeze_mtrag_reference_expansion import (
    freeze_mtrag_reference_expansion,
)
from benchmarks.mtrag import mtrag_conversation_split


def _conversation(split: DatasetSplit, index: int) -> str:
    for candidate_index in range(index * 100, (index + 1) * 100):
        candidate = f"freeze-expansion-{split.value}-{candidate_index}"
        if mtrag_conversation_split(candidate) is split:
            return candidate
    raise AssertionError("could not find split fixture")


def _row(split: DatasetSplit, collection: str, index: int) -> dict:
    conversation = _conversation(split, index)
    return {
        "conversation_id": conversation,
        "collection": collection,
        "current_task_id": f"{conversation}<::>2",
        "shared_document_ids": ["document"],
        "reuse_opportunity": {
            "current_token_count": 128,
            "candidate_blocks": [
                {"current_block_index": 2, "has_whole_source_block": True}
            ],
        },
    }


class FreezeMtragReferenceExpansionTests(TestCase):
    # Freeze new train/validation manifests while excluding completed tasks.
    def test_freezes_additional_reference_manifests(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "RAG.jsonl"
            source.write_text("fixture\n", encoding="utf-8")
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            rows = [
                _row(DatasetSplit.TRAIN, "A", 0),
                _row(DatasetSplit.TRAIN, "B", 1),
                _row(DatasetSplit.VALIDATION, "A", 0),
                _row(DatasetSplit.VALIDATION, "B", 1),
            ]
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "analysis": "mtrag-natural-block-coverage",
                        "source_sha256": source_sha256,
                        "prompt_template_version": 1,
                        "model": "Qwen/Qwen3-14B",
                        "tokenizer_class": "Qwen2Tokenizer",
                        "block_size": 16,
                        "transitions": rows,
                    }
                ),
                encoding="utf-8",
            )
            existing = root / "existing.json"
            excluded = rows[0]
            existing.write_text(
                json.dumps(
                    {
                        "source": "IBM/MTRAG",
                        "source_revision": "revision",
                        "source_sha256": source_sha256,
                        "split": "train",
                        "tasks": [
                            {
                                "task_id": excluded["current_task_id"],
                                "conversation_id": excluded["conversation_id"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            plan = freeze_mtrag_reference_expansion(
                coverage_path=coverage,
                source_dataset_path=source,
                existing_manifest_paths=[existing],
                output_dir=output,
                train_count=1,
                validation_count=2,
            )
            train = json.loads((output / "train-manifest.json").read_text())
            validation = json.loads(
                (output / "validation-manifest.json").read_text()
            )

        self.assertEqual(plan["new_reference_tasks"], 3)
        self.assertEqual(plan["final_reference_tasks"], 4)
        self.assertEqual(train["source_model"], "Qwen/Qwen3-14B")
        self.assertEqual(train["task_count"], 1)
        self.assertEqual(validation["task_count"], 2)
        self.assertNotEqual(train["tasks"][0]["task_id"], excluded["current_task_id"])
        self.assertEqual(plan["test_status"], "sealed_unopened")
