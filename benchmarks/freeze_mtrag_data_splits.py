"""Freeze conversation-separated MTRAG train, validation, and test manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import mtrag_source_sha256
from benchmarks.mtrag_calibration import select_mtrag_reference_calibration


# Read a previous task or transition manifest and return every used conversation.
def _excluded_conversations(manifest: Mapping[str, object]) -> frozenset[str]:
    rows = manifest.get("transitions")
    if rows is None:
        rows = manifest.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("exclusion manifest has no tasks or transitions")
    conversation_ids = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("exclusion transition must be an object")
        conversation_id = row.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("exclusion transition has no conversation ID")
        conversation_ids.append(conversation_id)
    return frozenset(conversation_ids)


# Select and save all three manifests before any new output is observed.
def freeze_mtrag_data_splits(
    *,
    coverage_path: Path,
    source_dataset_path: Path,
    exclusion_manifest_path: Path,
    output_dir: Path,
    task_counts: Mapping[DatasetSplit, int],
    existing_training_transitions: int,
) -> dict[str, object]:
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    exclusion = json.loads(exclusion_manifest_path.read_text(encoding="utf-8"))
    source_sha256 = mtrag_source_sha256(source_dataset_path)
    if exclusion.get("source_sha256") != source_sha256:
        raise ValueError("raw MTRAG source does not match the previous experiment")
    excluded = _excluded_conversations(exclusion)
    if (
        isinstance(existing_training_transitions, bool)
        or not isinstance(existing_training_transitions, int)
        or existing_training_transitions < 0
    ):
        raise ValueError("existing training transition count must be non-negative")
    coverage_sha256 = hashlib.sha256(coverage_path.read_bytes()).hexdigest()
    exclusion_sha256 = hashlib.sha256(exclusion_manifest_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST):
        count = task_counts.get(split)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"missing positive task count for {split.value}")
        selected = select_mtrag_reference_calibration(
            coverage,
            split=split,
            excluded_conversation_ids=excluded,
            task_count=count,
        )
        manifest = {
            **selected,
            "source": exclusion.get("source"),
            "source_revision": exclusion.get("source_revision"),
            "source_sha256": source_sha256,
            "source_coverage_sha256": coverage_sha256,
            "excluded_manifest_sha256": exclusion_sha256,
            "excluded_conversation_count": len(excluded),
        }
        path = output_dir / f"{split.value}-manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        paths[split.value] = str(path)

    total = sum(task_counts[split] for split in DatasetSplit)
    final_total = total + existing_training_transitions
    plan = {
        "schema_version": 1,
        "analysis": "mtrag-natural-selector-data-splits",
        "existing_training_transitions": existing_training_transitions,
        "new_training_transitions": task_counts[DatasetSplit.TRAIN],
        "validation_transitions": task_counts[DatasetSplit.VALIDATION],
        "test_transitions": task_counts[DatasetSplit.TEST],
        "final_transition_count": final_total,
        "final_split_percentages": {
            "train": (
                task_counts[DatasetSplit.TRAIN] + existing_training_transitions
            )
            / final_total,
            "validation": task_counts[DatasetSplit.VALIDATION] / final_total,
            "test": task_counts[DatasetSplit.TEST] / final_total,
        },
        "test_status": "frozen_unopened",
        "manifests": paths,
    }
    (output_dir / "split-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    return plan


# Parse the fixed 70/15/15 collection plan and freeze its files.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = freeze_mtrag_data_splits(
        coverage_path=args.coverage,
        source_dataset_path=args.source_dataset,
        exclusion_manifest_path=args.exclude_manifest,
        output_dir=args.output_dir,
        task_counts={
            DatasetSplit.TRAIN: 21,
            DatasetSplit.VALIDATION: 6,
            DatasetSplit.TEST: 6,
        },
        existing_training_transitions=7,
    )
    print(
        f"Frozen {plan['final_transition_count']} total transitions at "
        f"{plan['final_split_percentages']}"
    )


if __name__ == "__main__":
    main()
