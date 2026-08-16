"""Collect uncached MTRAG reference outputs before freezing quality limits."""

from __future__ import annotations

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
