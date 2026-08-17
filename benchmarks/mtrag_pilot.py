"""Select an audited causal-label pilot from MTRAG coverage."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import MTRAG_SPLIT_SEED, mtrag_conversation_split


# Extract aligned candidates while protecting the final output-producing block.
def _testable_blocks(
    row: Mapping[str, Any],
    *,
    block_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
    opportunity = row.get("reuse_opportunity")
    if not isinstance(opportunity, Mapping):
        raise ValueError("coverage transition has no reuse opportunity")
    candidates = opportunity.get("candidate_blocks")
    prompt_tokens = opportunity.get("current_token_count")
    if (
        not isinstance(candidates, list)
        or isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 1
    ):
        raise ValueError("coverage transition has invalid candidate metadata")

    aligned = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("coverage candidate must be an object")
        index = candidate.get("current_block_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("coverage candidate has an invalid block index")
        if (index + 1) * block_size > prompt_tokens:
            raise ValueError("coverage candidate is outside the full prompt blocks")
        if candidate.get("has_whole_source_block") is True:
            aligned.append(index)
    aligned_blocks = tuple(sorted(aligned))
    if len(set(aligned_blocks)) != len(aligned_blocks):
        raise ValueError("coverage transition has duplicate aligned candidates")
    output_block = (prompt_tokens - 1) // block_size
    excluded = output_block if output_block in aligned_blocks else None
    testable = tuple(index for index in aligned_blocks if index != output_block)
    return aligned_blocks, testable, excluded


# Rank a transition or block before observing any counterfactual output.
def _rank(seed: str, *values: object) -> str:
    serialized = ":".join((seed, *(str(value) for value in values)))
    return hashlib.sha256(serialized.encode()).hexdigest()


# Choose approved training transitions and a bounded block subset per collection.
def select_audited_mtrag_counterfactual_pilot(
    coverage: Mapping[str, Any],
    *,
    approved_task_ids: frozenset[str],
    quality_calibration_id: str,
    per_collection: int = 1,
    max_prompt_tokens: int = 4096,
    max_testable_blocks: int = 64,
    max_target_blocks: int = 2,
    split_seed: str = MTRAG_SPLIT_SEED,
) -> dict[str, Any]:
    if (
        coverage.get("schema_version") != 1
        or coverage.get("analysis") != "mtrag-natural-block-coverage"
    ):
        raise ValueError("input is not an MTRAG coverage artifact")
    if not approved_task_ids or not quality_calibration_id:
        raise ValueError("audited pilot requires an approved quality calibration")
    bounds = (
        per_collection,
        max_prompt_tokens,
        max_testable_blocks,
        max_target_blocks,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in bounds
    ):
        raise ValueError("pilot selection bounds must be positive integers")
    block_size = coverage.get("block_size")
    rows = coverage.get("transitions")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("coverage block size must be positive")
    if not isinstance(rows, list):
        raise ValueError("coverage transitions must be a list")

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    available_collections = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("coverage transition must be an object")
        identity = tuple(
            raw.get(field)
            for field in (
                "conversation_id",
                "collection",
                "previous_task_id",
                "current_task_id",
            )
        )
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError("coverage transition has invalid identity fields")
        conversation_id, collection, previous_task_id, current_task_id = identity
        shared_documents = raw.get("shared_document_ids")
        opportunity = raw.get("reuse_opportunity")
        if not isinstance(shared_documents, list) or not isinstance(
            opportunity, Mapping
        ):
            raise ValueError("coverage transition has invalid overlap fields")
        aligned, testable, excluded = _testable_blocks(raw, block_size=block_size)
        prompt_tokens = opportunity.get("current_token_count")
        if (
            mtrag_conversation_split(conversation_id, seed=split_seed)
            is not DatasetSplit.TRAIN
            or not shared_documents
            or not testable
            or len(testable) > max_testable_blocks
            or not isinstance(prompt_tokens, int)
            or prompt_tokens > max_prompt_tokens
        ):
            continue
        available_collections.add(collection)
        if current_task_id not in approved_task_ids:
            continue
        ranked_targets = sorted(
            testable,
            key=lambda index: _rank(split_seed, current_task_id, index),
        )
        eligible[collection].append(
            {
                "split": DatasetSplit.TRAIN.value,
                "conversation_id": conversation_id,
                "collection": collection,
                "previous_task_id": previous_task_id,
                "current_task_id": current_task_id,
                "shared_document_ids": shared_documents,
                "current_prompt_tokens": prompt_tokens,
                "estimated_native_cached_tokens": opportunity.get(
                    "native_cached_tokens"
                ),
                "aligned_candidate_block_indices": list(aligned),
                "testable_block_indices": list(testable),
                "target_block_indices": sorted(ranked_targets[:max_target_blocks]),
                "excluded_output_block_index": excluded,
            }
        )
    if set(eligible) != available_collections or not available_collections:
        raise ValueError("manual audit approved no eligible task for every collection")

    selected = []
    for collection in sorted(eligible):
        ranked = sorted(
            eligible[collection],
            key=lambda row: _rank(split_seed, row["current_task_id"]),
        )
        selected.extend(ranked[:per_collection])
    return {
        "schema_version": 1,
        "selection": "mtrag-audited-counterfactual-pilot",
        "source_analysis": coverage["analysis"],
        "source_prompt_template_version": coverage.get("prompt_template_version"),
        "source_model": coverage.get("model"),
        "source_tokenizer_class": coverage.get("tokenizer_class"),
        "split_seed": split_seed,
        "block_size": block_size,
        "quality_calibration_id": quality_calibration_id,
        "per_collection": per_collection,
        "max_prompt_tokens": max_prompt_tokens,
        "max_testable_blocks": max_testable_blocks,
        "max_target_blocks": max_target_blocks,
        "transition_count": len(selected),
        "collection_count": len({row["collection"] for row in selected}),
        "total_testable_blocks": sum(
            len(row["testable_block_indices"]) for row in selected
        ),
        "total_target_blocks": sum(
            len(row["target_block_indices"]) for row in selected
        ),
        "transitions": selected,
    }
