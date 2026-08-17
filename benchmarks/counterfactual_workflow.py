"""Orchestrate one complete counterfactual block-labelling workflow."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.block_dataset import DatasetSplit
from benchmarks.counterfactual_dataset import save_counterfactual_training_dataset
from benchmarks.counterfactual_trial import (
    run_counterfactual_candidate_discovery,
    run_discovered_counterfactual_trials,
)
from benchmarks.schema import RequestSpec, RequestTransition, WorkloadTrace
from observability.request_recorder import RequestRecorder, validate_ledger


# Resolve one declared transition and its two requests without guessing by order.
def _resolve_transition(
    trace: WorkloadTrace,
    transition_id: str,
) -> tuple[RequestTransition, RequestSpec, RequestSpec]:
    matches = tuple(
        transition
        for transition in trace.transitions
        if transition.transition_id == transition_id
    )
    if len(matches) != 1:
        raise ValueError("transition ID must identify exactly one trace transition")
    requests_by_id = {request.request_id: request for request in trace.requests}
    if len(requests_by_id) != len(trace.requests):
        raise ValueError("trace contains duplicate request IDs")
    transition = matches[0]
    try:
        source = requests_by_id[transition.previous_request_id]
        edited = requests_by_id[transition.current_request_id]
    except KeyError as error:
        raise ValueError("transition references a missing request") from error
    return transition, source, edited


# Run discovery, isolated block trials, CSV export, and final ledger validation.
def run_counterfactual_dataset_workflow(
    *,
    trace: WorkloadTrace,
    transition_id: str,
    split: DatasetSplit,
    output_path: Path,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder,
    expected_block_size: int | None = None,
    expected_candidate_block_indices: tuple[int, ...] | None = None,
    expected_testable_block_indices: tuple[int, ...] | None = None,
    selected_block_indices: tuple[int, ...] | None = None,
    required_reference_output: str | None = None,
    require_reference_output_match: bool = True,
    require_exact_output_match: bool = False,
) -> dict[str, Any]:
    transition, source, edited = _resolve_transition(trace, transition_id)
    discovery_run = run_counterfactual_candidate_discovery(
        trace_id=trace.trace_id,
        transition=transition,
        source_request=source,
        edited_request=edited,
        url=url,
        model=model,
        max_completion_tokens=max_completion_tokens,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        recorder=recorder,
        required_edited_output=required_reference_output,
    )
    discovery = discovery_run.discovery
    if expected_block_size is not None and discovery.block_size != expected_block_size:
        raise RuntimeError("live vLLM block size differs from the frozen pilot")
    if (
        expected_candidate_block_indices is not None
        and discovery.candidate_block_indices != expected_candidate_block_indices
    ):
        raise RuntimeError("live vLLM candidates differ from the frozen pilot")
    if (
        expected_testable_block_indices is not None
        and discovery.testable_block_indices != expected_testable_block_indices
    ):
        raise RuntimeError("live vLLM candidates differ from the frozen pilot")
    if not discovery.testable_block_indices:
        raise RuntimeError("transition has no full candidate blocks safe to test")
    fresh_reference_output = discovery_run.edited_observation.get("output_text")
    if not isinstance(fresh_reference_output, str) or not fresh_reference_output:
        raise RuntimeError("discovery produced no full-compute reference output")
    selected = (
        discovery.testable_block_indices
        if selected_block_indices is None
        else selected_block_indices
    )
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or not set(selected).issubset(discovery.testable_block_indices)
    ):
        raise RuntimeError("planned blocks are not a valid live candidate subset")
    batch = run_discovered_counterfactual_trials(
        discovery=discovery,
        source_request=source,
        edited_request=edited,
        url=url,
        model=model,
        max_completion_tokens=max_completion_tokens,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        recorder=recorder,
        selected_block_indices=selected,
        # Bind every trial to the full-compute answer produced by this server run.
        required_reference_output=fresh_reference_output,
        require_reference_output_match=require_reference_output_match,
        require_exact_output_match=require_exact_output_match,
    )
    training_rows = save_counterfactual_training_dataset(
        batch,
        split=split,
        path=output_path,
    )

    ledger = validate_ledger(recorder.path)
    if not ledger.is_complete or ledger.failed:
        raise RuntimeError("counterfactual request ledger is incomplete or failed")
    decisions = Counter(
        trial.label.decision.value for trial in batch.trials if trial.label is not None
    )
    return {
        "trace_id": trace.trace_id,
        "transition_id": transition.transition_id,
        "split": split.value,
        "discovery_id": discovery_run.discovery_id,
        "approved_reference_exact_match": (
            discovery_run.approved_output_exact_match
        ),
        "candidate_blocks": len(discovery.candidate_block_indices),
        "testable_blocks": len(discovery.testable_block_indices),
        "planned_blocks": len(selected),
        "excluded_output_block_index": discovery.excluded_output_block_index,
        "trial_count": len(batch.trials),
        "valid_training_rows": training_rows,
        "invalid_trials": sum(
            not trial.label_result.valid_reference
            or not trial.label_result.valid_execution
            for trial in batch.trials
        ),
        "abstained_trials": sum(
            trial.label_result.valid_reference
            and trial.label_result.valid_execution
            and trial.label_result.decision is None
            for trial in batch.trials
        ),
        "reference_drift_trials": sum(
            trial.reference_output_exact_match is False for trial in batch.trials
        ),
        "repair_labels": decisions["repair"],
        "reuse_labels": decisions["reuse"],
        "dataset_path": str(output_path),
        "request_ledger": str(ledger.path),
        "recorded_requests": ledger.started,
    }
