"""Save executed counterfactual trials as model-ready block examples."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from benchmarks.block_dataset import (
    DatasetSplit,
    SplitCandidateBlock,
    label_candidate_block,
    save_block_dataset_csv,
)
from benchmarks.counterfactual_trial import (
    CounterfactualTrialBatchResult,
    CounterfactualTrialResult,
)
from cacheselect.block_features import (
    CandidateBlockFeatures,
    extract_candidate_block_features,
)
from cacheselect.reuse_opportunity import analyze_reuse_opportunity


# Save a schema-correct empty table when every valid trial abstained.
def _save_empty_counterfactual_csv(path: Path) -> None:
    fieldnames = [
        "trace_id",
        "transition_id",
        "split",
        "decision",
        "label_source",
        "label_reason",
        *(field.name for field in fields(CandidateBlockFeatures)),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        csv.DictWriter(output, fieldnames=fieldnames).writeheader()


# Read and validate one prompt-token sequence recorded by the standard observer.
def _recorded_token_ids(
    observation: Mapping[str, Any],
    *,
    role: str,
) -> tuple[int, ...]:
    raw_token_ids = observation.get("prompt_token_ids")
    if not isinstance(raw_token_ids, (list, tuple)) or not raw_token_ids:
        raise ValueError(f"{role} observation has no prompt token IDs")
    if any(
        not isinstance(token, int) or isinstance(token, bool) for token in raw_token_ids
    ):
        raise ValueError(f"{role} prompt token IDs must be integers")
    return tuple(raw_token_ids)


# Reconstruct the cheap runtime features for the block tested by one trial.
def extract_counterfactual_trial_feature(
    trial: CounterfactualTrialResult,
    *,
    block_size: int,
) -> CandidateBlockFeatures:
    previous_tokens = _recorded_token_ids(trial.donor_observation, role="donor")
    current_tokens = _recorded_token_ids(
        trial.intervention_observation,
        role="intervention",
    )
    metrics = trial.intervention_observation.get("server_metrics")
    raw_plan = (
        metrics.get("cacheselect_partial_reuse_plan")
        if isinstance(metrics, Mapping)
        else None
    )
    if not isinstance(raw_plan, Mapping) or raw_plan.get("block_size") != block_size:
        raise ValueError("intervention has no matching partial-reuse plan")
    native_cached_tokens = raw_plan.get("native_cached_tokens")
    if not isinstance(native_cached_tokens, int) or isinstance(
        native_cached_tokens, bool
    ):
        raise ValueError("intervention native cached tokens must be an integer")

    opportunity = analyze_reuse_opportunity(
        previous_tokens,
        current_tokens,
        native_cached_tokens=native_cached_tokens,
        block_size=block_size,
    )
    matches = tuple(
        feature
        for feature in extract_candidate_block_features(
            previous_tokens,
            current_tokens,
            opportunity,
        )
        if feature.candidate_block_index == trial.intervention.reused_block_index
    )
    if len(matches) != 1:
        raise ValueError("tested block has no unique feature row")
    return matches[0]


# Save verified causal labels in the selector's existing training-table format.
def save_counterfactual_training_dataset(
    batch: CounterfactualTrialBatchResult,
    *,
    split: DatasetSplit,
    path: Path,
) -> int:
    discovery = batch.discovery
    selected_blocks = tuple(
        trial.intervention.reused_block_index for trial in batch.trials
    )
    if (
        not selected_blocks
        or selected_blocks != batch.target_block_indices
        or selected_blocks != tuple(sorted(set(selected_blocks)))
        or not set(selected_blocks).issubset(discovery.testable_block_indices)
    ):
        raise ValueError("trial results are not a valid discovered block subset")

    rows: list[SplitCandidateBlock] = []
    for trial in batch.trials:
        intervention = trial.intervention
        if (
            intervention.trace_id != discovery.trace_id
            or intervention.transition_id != discovery.transition_id
            or intervention.candidate_block_indices != discovery.candidate_block_indices
        ):
            raise ValueError("trial result does not belong to this discovery")
        if trial.label is None:
            continue
        if (
            not trial.execution_evidence.valid
            or trial.label_result.decision != trial.label.decision
        ):
            raise ValueError("label is inconsistent with its execution evidence")
        features = extract_counterfactual_trial_feature(
            trial,
            block_size=discovery.block_size,
        )
        rows.append(
            SplitCandidateBlock(
                example=label_candidate_block(
                    trace_id=discovery.trace_id,
                    transition_id=discovery.transition_id,
                    features=features,
                    decision=trial.label.decision,
                    source=trial.label.source,
                    reason=trial.label.reason,
                ),
                split=split,
            )
        )
    if rows:
        save_block_dataset_csv(rows, path)
    else:
        _save_empty_counterfactual_csv(path)
    return len(rows)
