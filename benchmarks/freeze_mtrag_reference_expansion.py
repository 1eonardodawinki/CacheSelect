"""Freeze 60/15 additional MTRAG train and validation reference tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import (
    MTRAG_SPLIT_SEED,
    mtrag_conversation_split,
    mtrag_source_sha256,
)
from benchmarks.mtrag_reference_expansion import (
    select_mtrag_reference_expansion,
)
from benchmarks.mtrag_pilot import mtrag_testable_blocks
from benchmarks.reviewed_mtrag_counterfactual import (
    QWEN3_REVIEWED_REFERENCE_GATE,
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


# Freeze every repacking-eligible transition while preserving conversation splits.
def freeze_mtrag_full_reference_splits(
    *,
    coverage_path: Path,
    source_dataset_path: Path,
    output_dir: Path,
    target_blocks_per_batch: int = 900,
    excluded_task_ids: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if isinstance(target_blocks_per_batch, bool) or target_blocks_per_batch < 1:
        raise ValueError("target blocks per batch must be positive")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    source_sha256 = mtrag_source_sha256(source_dataset_path)
    rows = coverage.get("transitions")
    if (
        coverage.get("analysis") != "mtrag-natural-block-coverage"
        or coverage.get("source_sha256") != source_sha256
        or not isinstance(rows, list)
    ):
        raise ValueError("coverage does not match the raw MTRAG source")

    raw_counts: Counter[DatasetSplit] = Counter()
    executable_counts: Counter[DatasetSplit] = Counter()
    excluded_counts: Counter[DatasetSplit] = Counter()
    max_prompt_tokens = 0
    max_testable_blocks = 0
    transitions = []
    block_size = coverage.get("block_size")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError("coverage block size is invalid")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("coverage transition must be an object")
        opportunity = row.get("reuse_opportunity")
        conversation_id = row.get("conversation_id")
        if not isinstance(opportunity, Mapping) or not isinstance(
            conversation_id, str
        ):
            raise ValueError("coverage transition metadata is invalid")
        candidate_blocks = opportunity.get("candidate_block_count")
        prompt_tokens = opportunity.get("current_token_count")
        if not isinstance(candidate_blocks, int) or not isinstance(prompt_tokens, int):
            raise ValueError("coverage transition counts are invalid")
        if candidate_blocks:
            split = mtrag_conversation_split(conversation_id)
            raw_counts[split] += 1
            candidates, testable, excluded = mtrag_testable_blocks(
                row,
                block_size=block_size,
                include_repacking=True,
            )
            if not testable:
                continue
            executable_counts[split] += 1
            if row["current_task_id"] in excluded_task_ids:
                excluded_counts[split] += 1
                continue
            max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
            max_testable_blocks = max(max_testable_blocks, len(testable))
            transitions.append(
                {
                    "split": split.value,
                    "conversation_id": conversation_id,
                    "collection": row["collection"],
                    "previous_task_id": row["previous_task_id"],
                    "current_task_id": row["current_task_id"],
                    "shared_document_ids": row["shared_document_ids"],
                    "candidate_block_indices": list(candidates),
                    "testable_block_indices": list(testable),
                    "target_block_indices": list(testable),
                    "excluded_output_block_index": excluded,
                }
            )

    if sum(raw_counts.values()) != coverage.get("candidate_transition_count"):
        raise ValueError("coverage candidate transition count is inconsistent")
    coverage_sha256 = hashlib.sha256(coverage_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in DatasetSplit:
        selected = select_mtrag_reference_expansion(
            coverage,
            split=split,
            excluded_task_ids=excluded_task_ids,
            prior_conversation_counts={},
            task_count=executable_counts[split] - excluded_counts[split],
            max_prompt_tokens=max_prompt_tokens,
            max_testable_blocks=max_testable_blocks,
            include_repacking=True,
        )
        artifact = {
            **selected,
            "source_sha256": source_sha256,
            "source_coverage_sha256": coverage_sha256,
        }
        path = output_dir / f"{split.value}-manifest.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        paths[split.value] = str(path)

    split_order = {split.value: index for index, split in enumerate(DatasetSplit)}
    transitions.sort(
        key=lambda row: (split_order[row["split"]], row["current_task_id"])
    )
    batches = []
    start = blocks = count = 0
    current_split = None
    for index, row in enumerate(transitions):
        row_blocks = len(row["target_block_indices"])
        if count and (
            blocks + row_blocks > target_blocks_per_batch
            or row["split"] != current_split
        ):
            batches.append(
                {
                    "batch": len(batches) + 1,
                    "split": current_split,
                    "start_case": start + 1,
                    "case_count": count,
                    "target_block_count": blocks,
                }
            )
            start, blocks, count = index, 0, 0
        current_split = row["split"]
        blocks += row_blocks
        count += 1
    batches.append(
        {
            "batch": len(batches) + 1,
            "split": current_split,
            "start_case": start + 1,
            "case_count": count,
            "target_block_count": blocks,
        }
    )
    counterfactual_plan = {
        "schema_version": 1,
        "selection": "mtrag-audited-counterfactual-pilot",
        "reference_status": "generated_in_trial_pending_review",
        "quality_calibration_id": QWEN3_REVIEWED_REFERENCE_GATE.calibration_id,
        "source_model": coverage.get("model"),
        "source_prompt_template_version": coverage.get("prompt_template_version"),
        "source_dataset_sha256": source_sha256,
        "source_coverage_sha256": coverage_sha256,
        "split_seed": MTRAG_SPLIT_SEED,
        "block_size": block_size,
        "max_target_blocks": None,
        "repacking_enabled": True,
        "transition_count": len(transitions),
        "total_testable_blocks": sum(
            len(row["testable_block_indices"]) for row in transitions
        ),
        "total_target_blocks": sum(
            len(row["target_block_indices"]) for row in transitions
        ),
        "target_blocks_per_batch": target_blocks_per_batch,
        "batch_count": len(batches),
        "batches": batches,
        "transitions": transitions,
    }
    (output_dir / "counterfactual-plan.json").write_text(
        json.dumps(counterfactual_plan, indent=2) + "\n", encoding="utf-8"
    )

    plan = {
        "schema_version": 1,
        "analysis": "mtrag-full-repacking-reference-plan",
        "selection_model": coverage.get("model"),
        "candidate_transitions": sum(raw_counts.values()),
        "candidate_transitions_by_split": {
            split.value: raw_counts[split] for split in DatasetSplit
        },
        "executable_transitions": sum(executable_counts.values()),
        "executable_transitions_by_split": {
            split.value: executable_counts[split] for split in DatasetSplit
        },
        "previously_completed_transitions": sum(excluded_counts.values()),
        "scheduled_transitions": len(transitions),
        "scheduled_transitions_by_split": {
            split.value: executable_counts[split] - excluded_counts[split]
            for split in DatasetSplit
        },
        "structurally_excluded_transitions": sum(raw_counts.values())
        - sum(executable_counts.values()),
        "max_prompt_tokens": max_prompt_tokens,
        "max_testable_blocks": max_testable_blocks,
        "selector_test_status": "frozen_unscored",
        "counterfactual_plan": str(output_dir / "counterfactual-plan.json"),
        "counterfactual_batches": len(batches),
        "manifests": paths,
    }
    (output_dir / "full-reference-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument(
        "--existing-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=60)
    parser.add_argument("--validation-count", type=int, default=15)
    parser.add_argument("--full-repacking-coverage", action="store_true")
    parser.add_argument("--target-blocks-per-batch", type=int, default=900)
    parser.add_argument("--exclude-plan", type=Path)
    args = parser.parse_args()
    if args.full_repacking_coverage:
        if args.existing_manifest:
            parser.error("full coverage does not accept existing manifests")
        excluded_task_ids = frozenset()
        if args.exclude_plan:
            prior = json.loads(args.exclude_plan.read_text(encoding="utf-8"))
            excluded_task_ids = frozenset(
                row["current_task_id"] for row in prior["transitions"]
            )
        plan = freeze_mtrag_full_reference_splits(
            coverage_path=args.coverage,
            source_dataset_path=args.source_dataset,
            output_dir=args.output_dir,
            target_blocks_per_batch=args.target_blocks_per_batch,
            excluded_task_ids=excluded_task_ids,
        )
        print(
            f"Scheduled {plan['scheduled_transitions']} new reference tasks; "
            f"excluded {plan['previously_completed_transitions']} completed tasks"
        )
        return
    if not args.existing_manifest:
        parser.error("reference expansion requires an existing manifest")
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
