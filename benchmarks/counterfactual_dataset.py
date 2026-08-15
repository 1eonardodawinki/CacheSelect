"""Save executed counterfactual trials as model-ready block examples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.counterfactual_trial import CounterfactualTrialResult
from cacheselect.block_features import (
    CandidateBlockFeatures,
    extract_candidate_block_features,
)
from cacheselect.reuse_opportunity import analyze_reuse_opportunity


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
    if len(matches) != 1 or matches[0].requires_repacking:
        raise ValueError("tested block has no unique aligned feature row")
    return matches[0]
