"""Collect uncached MTRAG reference outputs before freezing quality limits."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from benchmarks.block_dataset import DatasetSplit
from benchmarks.counterfactual_trial import _require_fresh_full_recompute
from benchmarks.mtrag import (
    MTRAG_PROMPT_TEMPLATE_VERSION,
    MTRAG_SPLIT_SEED,
    MtragTask,
    mtrag_conversation_split,
)
from benchmarks.mtrag_trace import build_mtrag_request_spec
from benchmarks.mtrag_pilot import mtrag_testable_blocks
from benchmarks.run_vllm_baseline import _observe_request
from benchmarks.schema import ReferenceSimilarityGate, RequestSpec
from observability.request_recorder import RequestRecorder


MTRAG_UNCALIBRATED_GATE_ID = "mtrag-reference-calibration-unfrozen"


@dataclass(frozen=True)
class MtragReferenceCalibrationCase:
    """One frozen natural request used only to calibrate answer quality."""

    request: RequestSpec
    split: DatasetSplit
    collection: str


# Rank an unseen task deterministically before any model output is observed.
def _calibration_rank(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


# Select diverse unseen conversations from exactly one frozen dataset split.
def select_mtrag_reference_calibration(
    coverage: Mapping[str, Any],
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
    excluded_conversation_ids: frozenset[str] = frozenset(),
    task_count: int = 28,
    max_prompt_tokens: int = 4096,
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
        raise ValueError("calibration selection bounds must be positive integers")
    if any(not conversation_id for conversation_id in excluded_conversation_ids):
        raise ValueError("excluded conversation IDs must not be empty")
    block_size = coverage.get("block_size")
    rows = coverage.get("transitions")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
        or not isinstance(rows, list)
    ):
        raise ValueError("coverage block metadata is invalid")

    # Keep at most one transition from each conversation to maximize diversity.
    eligible: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
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
        conversation_id, collection, current_task_id = identity
        collections.add(collection)
        opportunity = raw.get("reuse_opportunity")
        shared_documents = raw.get("shared_document_ids")
        if not isinstance(opportunity, Mapping) or not isinstance(shared_documents, list):
            raise ValueError("coverage transition has invalid reuse metadata")
        _, testable, _ = mtrag_testable_blocks(raw, block_size=block_size)
        prompt_tokens = opportunity.get("current_token_count")
        if (
            conversation_id in excluded_conversation_ids
            or mtrag_conversation_split(conversation_id, seed=split_seed)
            is not split
            or not testable
            or len(testable) > max_testable_blocks
            or not isinstance(prompt_tokens, int)
            or prompt_tokens > max_prompt_tokens
        ):
            continue
        candidate = {
            "task_id": current_task_id,
            "conversation_id": conversation_id,
            "collection": collection,
            "split": split.value,
        }
        previous = eligible[collection].get(conversation_id)
        if previous is None or _calibration_rank(
            split_seed, current_task_id
        ) < _calibration_rank(split_seed, previous["task_id"]):
            eligible[collection][conversation_id] = candidate

    ranked_by_collection = {}
    for collection in sorted(collections):
        ranked_by_collection[collection] = sorted(
            eligible[collection].values(),
            key=lambda row: _calibration_rank(split_seed, row["task_id"]),
        )
    if sum(map(len, ranked_by_collection.values())) < task_count:
        raise ValueError("coverage has too few eligible tasks for the requested split")

    # Draw one task per collection per round so smaller domains stay represented.
    selected = []
    round_index = 0
    while len(selected) < task_count:
        for collection in sorted(ranked_by_collection):
            ranked = ranked_by_collection[collection]
            if round_index < len(ranked):
                selected.append(ranked[round_index])
                if len(selected) == task_count:
                    break
        round_index += 1
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
        "tasks": selected,
    }


# Resolve a compact task manifest without involving candidate block metadata.
def build_mtrag_reference_calibration_cases(
    tasks: Sequence[MtragTask],
    manifest: Mapping[str, Any],
    *,
    quality_gate: ReferenceSimilarityGate,
    expected_model: str,
) -> tuple[MtragReferenceCalibrationCase, ...]:
    if manifest.get("schema_version") != 1 or manifest.get("selection") != (
        "mtrag-reference-quality-calibration"
    ):
        raise ValueError("unsupported MTRAG reference calibration manifest")
    if (
        manifest.get("source_prompt_template_version") != MTRAG_PROMPT_TEMPLATE_VERSION
        or manifest.get("source_model") != expected_model
        or manifest.get("split_seed") != MTRAG_SPLIT_SEED
    ):
        raise ValueError("MTRAG calibration provenance is incompatible")
    rows = manifest.get("tasks")
    if not isinstance(rows, list) or manifest.get("task_count") != len(rows):
        raise ValueError("MTRAG calibration task count is inconsistent")
    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("MTRAG input contains duplicate task IDs")

    cases = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("MTRAG calibration task must be an object")
        try:
            task = tasks_by_id[row["task_id"]]
        except (KeyError, TypeError) as error:
            raise ValueError("MTRAG calibration references a missing task") from error
        split = mtrag_conversation_split(task.conversation_id)
        if (
            row.get("conversation_id") != task.conversation_id
            or row.get("collection") != task.collection
            or row.get("split") != split.value
        ):
            raise ValueError("MTRAG calibration task identity is inconsistent")
        cases.append(
            MtragReferenceCalibrationCase(
                build_mtrag_request_spec(task, quality_gate=quality_gate),
                split,
                task.collection,
            )
        )
    return tuple(cases)


# Run each selected current request once with an isolated, empty cache namespace.
def run_mtrag_reference_calibration(
    cases: tuple[MtragReferenceCalibrationCase, ...],
    *,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("MTRAG reference calibration requires at least one case")
    rows = []
    seen_task_ids = set()
    for case in cases:
        original = case.request
        if original.request_id in seen_task_ids:
            raise ValueError("MTRAG calibration contains a duplicate current task")
        seen_task_ids.add(original.request_id)
        gate = original.ground_truth.reference_similarity_gate
        if gate is None or gate.calibration_id != MTRAG_UNCALIBRATED_GATE_ID:
            raise ValueError("MTRAG calibration request has a frozen or missing gate")

        run_id = uuid4().hex
        request = replace(original, request_id=f"{run_id}:mtrag-calibration")
        observation = _observe_request(
            request,
            url=url,
            model=model,
            max_completion_tokens=max_completion_tokens,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            recorder=recorder,
            policy_metadata={
                "mtrag_calibration_task_id": original.request_id,
                "mtrag_collection": case.collection,
            },
            require_cacheselect_metrics=True,
            vllm_xargs={"cacheselect_request_id": request.request_id},
            cache_salt=f"{run_id}:uncached-reference",
        )
        _require_fresh_full_recompute(observation, "MTRAG calibration reference")
        quality = observation.get("quality") or {}
        if quality.get("mode") != "reference_similarity":
            raise RuntimeError("MTRAG calibration did not record reference metrics")
        rows.append(
            {
                "task_id": original.request_id,
                "split": case.split.value,
                "collection": case.collection,
                "prompt_token_count": observation.get("prompt_token_count"),
                "output_text": observation.get("output_text"),
                "expected_answer": original.ground_truth.expected_answer,
                "quality": quality,
            }
        )
    return {
        "schema_version": 1,
        "analysis": "mtrag-reference-quality-calibration",
        "gate_status": "unfrozen",
        "model": model,
        "max_completion_tokens": max_completion_tokens,
        "request_count": len(rows),
        "rows": rows,
    }
