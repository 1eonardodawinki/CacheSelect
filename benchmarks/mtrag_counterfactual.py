"""Run frozen MTRAG cases through the existing counterfactual workflow."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmarks.counterfactual_workflow import run_counterfactual_dataset_workflow
from benchmarks.counterfactual_trial import (
    CounterfactualReferenceQualityError,
    _require_fresh_full_recompute,
)
from benchmarks.evaluation import compare_response_quality
from benchmarks.mtrag_trace import MtragCounterfactualCase
from benchmarks.run_vllm_baseline import _observe_request
from benchmarks.schema import save_trace
from observability.request_recorder import RequestRecorder


COUNT_FIELDS = (
    "trial_count",
    "valid_training_rows",
    "invalid_trials",
    "abstained_trials",
    "reference_drift_trials",
    "repair_labels",
    "reuse_labels",
)


# Compare full computation with one natural MLP-selected reuse run per case.
def run_mtrag_policy_cases(
    cases: Sequence[MtragCounterfactualCase],
    *,
    reference_outputs: Mapping[str, str] | None,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("MTRAG policy evaluation contains no cases")
    rows = []
    for case in cases:
        if len(case.trace.requests) != 2 or len(case.trace.transitions) != 1:
            raise ValueError("each MTRAG policy case must contain one request pair")
        transition = case.trace.transitions[0]
        requests = {request.request_id: request for request in case.trace.requests}
        source = requests[transition.previous_request_id]
        edited = requests[transition.current_request_id]
        approved_output = reference_outputs[edited.request_id]
        evaluation_id = uuid4().hex
        reference = replace(edited, request_id=f"{evaluation_id}:reference")
        donor = replace(source, request_id=f"{evaluation_id}:donor")
        policy = replace(edited, request_id=f"{evaluation_id}:policy")

        def observe(request, role, salt, xargs, completion_tokens):
            return _observe_request(
                request,
                url=url,
                model=model,
                max_completion_tokens=completion_tokens,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                recorder=recorder,
                policy_metadata={
                    "mtrag_policy_evaluation_id": evaluation_id,
                    "mtrag_policy_role": role,
                },
                require_cacheselect_metrics=True,
                vllm_xargs=xargs,
                cache_salt=salt,
            )

        reference_observation = observe(
            reference,
            "reference",
            f"{evaluation_id}:reference",
            {"cacheselect_request_id": reference.request_id},
            max_completion_tokens,
        )
        _require_fresh_full_recompute(reference_observation, "policy reference")
        donor_observation = observe(
            donor,
            "donor",
            f"{evaluation_id}:reuse",
            {"cacheselect_request_id": donor.request_id},
            1,
        )
        _require_fresh_full_recompute(donor_observation, "policy donor")
        policy_observation = observe(
            policy,
            "policy",
            f"{evaluation_id}:reuse",
            {
                "cacheselect_request_id": policy.request_id,
                "cacheselect_source_request_id": donor.request_id,
                "cacheselect_transition_id": transition.transition_id,
            },
            max_completion_tokens,
        )
        metrics = policy_observation.get("server_metrics") or {}
        candidate_tokens = int(metrics.get("cacheselect_candidate_tokens") or 0)
        repair_tokens = int(metrics.get("cacheselect_repair_tokens") or 0)
        selected_reuse_tokens = int(
            metrics.get("cacheselect_skipped_repair_tokens") or 0
        )
        if candidate_tokens != repair_tokens + selected_reuse_tokens:
            raise RuntimeError("MLP repair accounting is inconsistent")
        if candidate_tokens and metrics.get("cacheselect_repair_selector") != "mlp":
            raise RuntimeError("MTRAG policy evaluation did not use the MLP")
        comparison = compare_response_quality(
            reference_observation["quality"],
            policy_observation["quality"],
            edited.ground_truth,
        )
        exact = reference_observation.get("output_text") == policy_observation.get(
            "output_text"
        )
        reference_ttft = float(
            (reference_observation.get("server_metrics") or {})[
                "time_to_first_token_ms"
            ]
        )
        policy_ttft = float(metrics["time_to_first_token_ms"])
        valid_reference = (
            reference_observation.get("finish_reason") == "stop"
            and comparison["valid_reference"]
        )
        quality_passed = (
            policy_observation.get("finish_reason") == "stop" and comparison["passed"]
        )
        rows.append(
            {
                "trace_id": case.trace.trace_id,
                "transition_id": transition.transition_id,
                "current_task_id": edited.request_id,
                "split": case.split.value,
                "collection": case.collection,
                "approved_reference_exact_match": (
                    reference_observation.get("output_text") == approved_output
                ),
                "reference_output": reference_observation.get("output_text"),
                "policy_output": policy_observation.get("output_text"),
                "reference_finish_reason": reference_observation.get("finish_reason"),
                "policy_finish_reason": policy_observation.get("finish_reason"),
                "valid_reference": valid_reference,
                "exact_output_match": exact,
                "quality_passed": quality_passed,
                "requires_manual_review": valid_reference and not exact,
                "quality_comparison": comparison,
                "candidate_tokens": candidate_tokens,
                "repair_tokens": repair_tokens,
                "selected_reuse_tokens": selected_reuse_tokens,
                "executed_cached_tokens": int(
                    policy_observation.get("cached_tokens") or 0
                ),
                "reuse_executed": bool(
                    metrics.get("cacheselect_compacted_batch_executed")
                ),
                "execution_reason": metrics.get("cacheselect_execution_reason"),
                "reference_ttft_ms": reference_ttft,
                "policy_ttft_ms": policy_ttft,
                "ttft_speedup": reference_ttft / policy_ttft,
                "reference_wall_seconds": reference_observation["client_wall_seconds"],
                "policy_wall_seconds": policy_observation["client_wall_seconds"],
            }
        )

    valid_rows = [row for row in rows if row["valid_reference"]]
    candidate_tokens = sum(row["candidate_tokens"] for row in rows)
    selected_tokens = sum(row["selected_reuse_tokens"] for row in rows)
    reference_ttft = sum(row["reference_ttft_ms"] for row in valid_rows)
    policy_ttft = sum(row["policy_ttft_ms"] for row in valid_rows)
    return {
        "schema_version": 1,
        "experiment": "mtrag-natural-policy-evaluation",
        "case_count": len(rows),
        "valid_reference_count": len(valid_rows),
        "exact_output_matches": sum(row["exact_output_match"] for row in valid_rows),
        "quality_passes": sum(row["quality_passed"] for row in valid_rows),
        "manual_review_cases": sum(row["requires_manual_review"] for row in rows),
        "candidate_tokens": candidate_tokens,
        "selected_reuse_tokens": selected_tokens,
        "executed_cached_tokens": sum(row["executed_cached_tokens"] for row in rows),
        "selected_reuse_share": (
            selected_tokens / candidate_tokens if candidate_tokens else 0.0
        ),
        "aggregate_ttft_speedup": (
            reference_ttft / policy_ttft if policy_ttft else None
        ),
        "cases": rows,
    }


# Execute every frozen case sequentially against one already-running vLLM server.
def run_mtrag_counterfactual_cases(
    cases: Sequence[MtragCounterfactualCase],
    *,
    reference_outputs: Mapping[str, str],
    output_dir: Path,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder,
    require_reference_output_match: bool = False,
    require_reference_quality: bool = True,
    source_case_start: int = 1,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("MTRAG pilot contains no cases")
    if source_case_start < 1:
        raise ValueError("source case start must be positive")
    summaries = []
    totals: Counter[str] = Counter()
    planned_target_blocks = sum(len(case.target_block_indices) for case in cases)
    for index, case in enumerate(cases, start=1):
        source_case_index = source_case_start + index - 1
        if len(case.trace.transitions) != 1:
            raise ValueError("each MTRAG case must contain exactly one transition")
        transition = case.trace.transitions[0]
        approved_output = None
        if reference_outputs is not None:
            try:
                approved_output = reference_outputs[transition.current_request_id]
            except KeyError as error:
                raise ValueError("MTRAG case has no approved reference output") from error

        case_dir = output_dir / f"case-{index:02d}"
        trace_path = case_dir / "trace.json"
        dataset_path = case_dir / "counterfactual-blocks.csv"
        summary_path = case_dir / "summary.json"
        case_dir.mkdir(parents=True, exist_ok=True)
        save_trace(case.trace, trace_path)
        common = {
            "source_case_index": source_case_index,
            "collection": case.collection,
            "current_task_id": transition.current_request_id,
            "trace_path": str(trace_path),
        }
        try:
            result = run_counterfactual_dataset_workflow(
                trace=case.trace,
                transition_id=transition.transition_id,
                split=case.split,
                output_path=dataset_path,
                url=url,
                model=model,
                max_completion_tokens=max_completion_tokens,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                recorder=recorder,
                expected_block_size=case.block_size,
                expected_candidate_block_indices=case.expected_candidate_block_indices,
                expected_testable_block_indices=case.expected_testable_block_indices,
                selected_block_indices=case.target_block_indices,
                required_reference_output=approved_output,
                require_reference_output_match=require_reference_output_match,
                require_exact_output_match=True,
                require_reference_quality=require_reference_quality,
            )
        except CounterfactualReferenceQualityError as error:
            result = {
                "status": "skipped_reference_quality",
                "reason": str(error),
                **{field: 0 for field in COUNT_FIELDS},
                "skipped_target_blocks": len(case.target_block_indices),
            }
            totals["skipped_reference_cases"] += 1
            totals["skipped_target_blocks"] += len(case.target_block_indices)
        else:
            result = {"status": "completed", **result, "skipped_target_blocks": 0}
            totals["completed_cases"] += 1
            for field in COUNT_FIELDS:
                totals[field] += result[field]
        summary = {**common, **result}
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)

    return {
        "schema_version": 1,
        "experiment": "mtrag-counterfactual-pilot",
        "case_count": len(summaries),
        "source_case_start": source_case_start,
        "source_case_end": source_case_start + len(summaries) - 1,
        "completed_case_count": totals["completed_cases"],
        "skipped_reference_case_count": totals["skipped_reference_cases"],
        "planned_target_blocks": planned_target_blocks,
        "skipped_target_blocks": totals["skipped_target_blocks"],
        **{field: totals[field] for field in COUNT_FIELDS},
        "cases": summaries,
    }
