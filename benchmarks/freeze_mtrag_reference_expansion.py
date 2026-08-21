"""Freeze 60/15 additional MTRAG train and validation reference tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import mtrag_source_sha256
from benchmarks.mtrag_reference_expansion import (
    select_mtrag_reference_expansion,
)


# Read completed task identities without inspecting their model outputs.
def _existing_usage(
    manifests: Sequence[Mapping[str, object]],
) -> tuple[frozenset[str], Counter[str], Counter[str]]:
    task_ids = set()
    conversations: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for manifest in manifests:
        rows = manifest.get("tasks")
        if not isinstance(rows, list):
            raise ValueError("existing manifest has no task list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("existing manifest task must be an object")
            task_id = row.get("task_id")
            conversation_id = row.get("conversation_id")
            split = row.get("split", manifest.get("split"))
            if (
                not isinstance(task_id, str)
                or not task_id
                or not isinstance(conversation_id, str)
                or not conversation_id
                or split not in ("train", "validation")
            ):
                raise ValueError("existing manifest task identity is invalid")
            if task_id in task_ids:
                raise ValueError("existing manifests repeat a task")
            task_ids.add(task_id)
            conversations[conversation_id] += 1
            splits[split] += 1
    return frozenset(task_ids), conversations, splits


# Create both manifests before any expanded model outputs are observed.
def freeze_mtrag_reference_expansion(
    *,
    coverage_path: Path,
    source_dataset_path: Path,
    existing_manifest_paths: Sequence[Path],
    output_dir: Path,
    train_count: int = 60,
    validation_count: int = 15,
) -> dict[str, object]:
    if len(existing_manifest_paths) < 1:
        raise ValueError("at least one existing manifest is required")
    counts = {
        DatasetSplit.TRAIN: train_count,
        DatasetSplit.VALIDATION: validation_count,
    }
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in counts.values()
    ):
        raise ValueError("expansion counts must be positive integers")

    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in existing_manifest_paths
    ]
    source_sha256 = mtrag_source_sha256(source_dataset_path)
    if coverage.get("source_sha256") != source_sha256:
        raise ValueError("Qwen3 coverage does not match the raw MTRAG source")
    for manifest in manifests:
        if manifest.get("source_sha256") != source_sha256:
            raise ValueError("existing manifest does not match the MTRAG source")
    excluded_tasks, prior_conversations, existing_splits = _existing_usage(manifests)

    coverage_sha256 = hashlib.sha256(coverage_path.read_bytes()).hexdigest()
    manifest_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in existing_manifest_paths
    }
    provenance = manifests[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    selected_task_ids = set()
    for split, count in counts.items():
        selected = select_mtrag_reference_expansion(
            coverage,
            split=split,
            excluded_task_ids=excluded_tasks,
            prior_conversation_counts=prior_conversations,
            task_count=count,
        )
        current_ids = {row["task_id"] for row in selected["tasks"]}
        if selected_task_ids & current_ids:
            raise ValueError("expansion manifests overlap")
        selected_task_ids.update(current_ids)
        artifact = {
            **selected,
            "source": provenance.get("source"),
            "source_revision": provenance.get("source_revision"),
            "source_sha256": source_sha256,
            "source_coverage_sha256": coverage_sha256,
            "excluded_manifest_sha256": manifest_hashes,
            "excluded_task_count": len(excluded_tasks),
        }
        path = output_dir / f"{split.value}-manifest.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        paths[split.value] = str(path)

    new_total = train_count + validation_count
    final_train = existing_splits["train"] + train_count
    final_validation = existing_splits["validation"] + validation_count
    final_total = final_train + final_validation
    plan = {
        "schema_version": 1,
        "analysis": "mtrag-reference-expansion-plan",
        "selection_model": coverage.get("model"),
        "existing_reference_tasks": len(excluded_tasks),
        "new_train_tasks": train_count,
        "new_validation_tasks": validation_count,
        "new_reference_tasks": new_total,
        "final_reference_tasks": final_total,
        "final_train_fraction": final_train / final_total,
        "final_validation_fraction": final_validation / final_total,
        "test_status": "sealed_unopened",
        "manifests": paths,
    }
    (output_dir / "expansion-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument(
        "--existing-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=60)
    parser.add_argument("--validation-count", type=int, default=15)
    args = parser.parse_args()
    plan = freeze_mtrag_reference_expansion(
        coverage_path=args.coverage,
        source_dataset_path=args.source_dataset,
        existing_manifest_paths=args.existing_manifest,
        output_dir=args.output_dir,
        train_count=args.train_count,
        validation_count=args.validation_count,
    )
    print(
        f"Frozen {plan['new_reference_tasks']} additional references; "
        f"planned total={plan['final_reference_tasks']}"
    )


if __name__ == "__main__":
    main()
