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
    )
    discovery = discovery_run.discovery
    if not discovery.testable_block_indices:
        raise RuntimeError("transition has no full candidate blocks safe to test")
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
        "candidate_blocks": len(discovery.candidate_block_indices),
        "testable_blocks": len(discovery.testable_block_indices),
        "excluded_output_block_index": discovery.excluded_output_block_index,
        "trial_count": len(batch.trials),
        "valid_training_rows": training_rows,
        "invalid_trials": sum(trial.label is None for trial in batch.trials),
        "repair_labels": decisions["repair"],
        "reuse_labels": decisions["reuse"],
        "dataset_path": str(output_path),
        "request_ledger": str(ledger.path),
        "recorded_requests": ledger.started,
    }
