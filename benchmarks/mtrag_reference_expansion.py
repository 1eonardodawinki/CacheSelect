"""Select additional MTRAG references without repeating completed tasks."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import MTRAG_SPLIT_SEED, mtrag_conversation_split
from benchmarks.mtrag_pilot import mtrag_testable_blocks


# Rank tasks before model outputs are generated or manually reviewed.
def _expansion_rank(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:expansion:{task_id}".encode()).hexdigest()


# Select a balanced expansion while preferring less-used conversations.
def select_mtrag_reference_expansion(
    coverage: Mapping[str, Any],
    *,
    split: DatasetSplit,
    excluded_task_ids: frozenset[str],
    prior_conversation_counts: Mapping[str, int],
    task_count: int,
    max_prompt_tokens: int = 7000,
    max_testable_blocks: int = 64,
    split_seed: str = MTRAG_SPLIT_SEED,
) -> dict[str, Any]:
    if (
        coverage.get("schema_version") != 1
        or coverage.get("analysis") != "mtrag-natural-block-coverage"
    ):
        raise ValueError("input is not an MTRAG coverage artifact")
    bounds = (task_count, max_prompt_tokens, max_testable_blocks)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in bounds
    ):
        raise ValueError("expansion selection bounds must be positive integers")
    if any(not task_id for task_id in excluded_task_ids):
        raise ValueError("excluded task IDs must not be empty")
    if any(
        not conversation_id
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for conversation_id, count in prior_conversation_counts.items()
    ):
        raise ValueError("prior conversation counts are invalid")

    block_size = coverage.get("block_size")
    rows = coverage.get("transitions")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
        or not isinstance(rows, list)
    ):
        raise ValueError("coverage block metadata is invalid")

    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_task_ids = set()
    collections = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("coverage transition must be an object")
        identity = tuple(
            raw.get(field)
            for field in ("conversation_id", "collection", "current_task_id")
        )
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError("coverage transition has invalid identity fields")
        conversation_id, collection, task_id = identity
        if task_id in seen_task_ids:
            raise ValueError("coverage contains a duplicate current task")
        seen_task_ids.add(task_id)
        collections.add(collection)
        opportunity = raw.get("reuse_opportunity")
        if not isinstance(opportunity, Mapping):
            raise ValueError("coverage transition has invalid reuse metadata")
        _, testable, _ = mtrag_testable_blocks(raw, block_size=block_size)
        prompt_tokens = opportunity.get("current_token_count")
        if (
            task_id in excluded_task_ids
            or mtrag_conversation_split(conversation_id, seed=split_seed) is not split
            or not testable
            or len(testable) > max_testable_blocks
            or not isinstance(prompt_tokens, int)
            or prompt_tokens > max_prompt_tokens
        ):
            continue
        candidates[collection].append(
            {
                "task_id": task_id,
                "conversation_id": conversation_id,
                "collection": collection,
                "split": split.value,
            }
        )

    if sum(map(len, candidates.values())) < task_count:
        raise ValueError("coverage has too few eligible unused tasks")
    usage = Counter(prior_conversation_counts)
    selected = []
    while len(selected) < task_count:
        made_progress = False
        for collection in sorted(collections):
            available = candidates[collection]
            if not available:
                continue
            chosen = min(
                available,
                key=lambda row: (
                    usage[row["conversation_id"]],
                    _expansion_rank(split_seed, row["task_id"]),
                ),
            )
            available.remove(chosen)
            selected.append(chosen)
            usage[chosen["conversation_id"]] += 1
            made_progress = True
            if len(selected) == task_count:
                break
        if not made_progress:
            raise ValueError("collections cannot satisfy the requested task count")

    selected_conversations = {row["conversation_id"] for row in selected}
    return {
        "schema_version": 1,
        "selection": "mtrag-reference-quality-calibration",
        "source_prompt_template_version": coverage.get("prompt_template_version"),
        "source_model": coverage.get("model"),
        "source_tokenizer_class": coverage.get("tokenizer_class"),
        "split_seed": split_seed,
        "split": split.value,
        "requested_task_count": task_count,
        "max_prompt_tokens": max_prompt_tokens,
        "max_testable_blocks": max_testable_blocks,
        "task_count": len(selected),
        "collection_count": len(collections),
        "collection_task_counts": {
            collection: sum(row["collection"] == collection for row in selected)
            for collection in sorted(collections)
        },
        "selected_conversation_count": len(selected_conversations),
        "previously_unused_conversation_count": sum(
            prior_conversation_counts.get(conversation_id, 0) == 0
            for conversation_id in selected_conversations
        ),
        "tasks": selected,
    }
