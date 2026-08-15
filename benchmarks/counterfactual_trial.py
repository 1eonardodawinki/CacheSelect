"""Run one isolated single-block KV-reuse counterfactual trial."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from benchmarks.block_dataset import BlockRepairLabel
from benchmarks.counterfactual_labels import (
    CounterfactualExecutionEvidence,
    CounterfactualLabelResult,
    SingleBlockIntervention,
    counterfactual_block_label,
    score_single_block_intervention,
    validate_counterfactual_execution,
)
from benchmarks.run_vllm_baseline import _observe_request
from benchmarks.schema import RequestSpec
from observability.request_recorder import RequestRecorder


@dataclass(frozen=True)
class CounterfactualTrialResult:
    """All observations and the optional training label from one trial."""

    trial_id: str
    intervention: SingleBlockIntervention
    reference_observation: dict[str, Any]
    donor_observation: dict[str, Any]
    intervention_observation: dict[str, Any]
    execution_evidence: CounterfactualExecutionEvidence
    label_result: CounterfactualLabelResult
    label: BlockRepairLabel | None


@dataclass(frozen=True)
class CounterfactualCandidateDiscovery:
    """Validated block candidates found in one vLLM discovery response."""

    trace_id: str
    transition_id: str
    block_size: int
    candidate_block_indices: tuple[int, ...]
    testable_block_indices: tuple[int, ...]
    excluded_output_block_index: int | None


# Convert vLLM's raw plan metrics into blocks suitable for isolated trials.
def extract_counterfactual_candidate_discovery(
    *,
    trace_id: str,
    transition_id: str,
    observation: Mapping[str, Any],
) -> CounterfactualCandidateDiscovery:
    if not trace_id or not transition_id:
        raise ValueError("trace and transition IDs must not be empty")
    token_ids = observation.get("prompt_token_ids")
    prompt_token_count = observation.get("prompt_token_count")
    if not isinstance(token_ids, (list, tuple)) or not token_ids:
        raise ValueError("discovery requires prompt token IDs")
    if (
        not isinstance(prompt_token_count, int)
        or isinstance(prompt_token_count, bool)
        or prompt_token_count != len(token_ids)
    ):
        raise ValueError("prompt token count does not match the recorded token IDs")

    metrics = observation.get("server_metrics")
    raw_plan = (
        metrics.get("cacheselect_partial_reuse_plan")
        if isinstance(metrics, Mapping)
        else None
    )
    if not isinstance(raw_plan, Mapping):
        raise ValueError("discovery response has no partial-reuse plan")
    if raw_plan.get("transition_id") != transition_id:
        raise ValueError("discovery plan has the wrong transition ID")

    block_size = raw_plan.get("block_size")
    native_cached_tokens = raw_plan.get("native_cached_tokens")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError("discovery block size must be a positive integer")
    if (
        not isinstance(native_cached_tokens, int)
        or isinstance(native_cached_tokens, bool)
        or native_cached_tokens < 0
        or native_cached_tokens > prompt_token_count
    ):
        raise ValueError("native cached tokens must be a non-negative integer")

    raw_candidates = raw_plan.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("discovery plan candidates must be a list")
    if raw_plan.get("candidate_block_count") != len(raw_candidates):
        raise ValueError("discovery candidate-block count is inconsistent")
    if raw_plan.get("candidate_token_count") != len(raw_candidates) * block_size:
        raise ValueError("discovery candidate-token count is inconsistent")

    first_uncached_block = (native_cached_tokens + block_size - 1) // block_size
    full_prompt_blocks = prompt_token_count // block_size
    targets: list[int] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("each discovery candidate must be an object")
        target = candidate.get("target_block_index")
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError("candidate target block must be an integer")
        if target < first_uncached_block or target >= full_prompt_blocks:
            raise ValueError("candidate target block is outside the reusable suffix")
        if candidate.get("source_resident") is not True:
            raise ValueError("every discovery source block must be resident")
        targets.append(target)
    if len(set(targets)) != len(targets):
        raise ValueError("candidate target blocks must be unique")

    ordered_targets = tuple(sorted(targets))
    output_block = (prompt_token_count - 1) // block_size
    excluded = output_block if output_block in ordered_targets else None
    # The final prompt row must still run because it produces the first output logits.
    testable = tuple(target for target in ordered_targets if target != output_block)
    return CounterfactualCandidateDiscovery(
        trace_id,
        transition_id,
        block_size,
        ordered_targets,
        testable,
        excluded,
    )


# Reject a reference or donor that unexpectedly used cached model computation.
def _require_fresh_full_recompute(
    observation: dict[str, Any],
    role: str,
) -> None:
    policy = (observation.get("runtime_policy") or {}).get("policy")
    metrics = observation.get("server_metrics") or {}
    if (
        observation.get("cached_tokens") != 0
        or policy != "FULL_RECOMPUTE"
        or metrics.get("cacheselect_compacted_batch_executed") is True
    ):
        raise RuntimeError(f"counterfactual {role} was not a fresh full recompute")


# Run an uncached reference, its donor, and one isolated reuse intervention.
def run_single_block_counterfactual_trial(
    *,
    source_request: RequestSpec,
    edited_request: RequestSpec,
    intervention: SingleBlockIntervention,
    block_size: int,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder,
) -> CounterfactualTrialResult:
    if source_request.request_id == edited_request.request_id:
        raise ValueError("source and edited requests must be different")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if intervention.reused_block_index not in intervention.candidate_block_indices:
        raise ValueError("reused block must belong to the intervention candidates")

    trial_id = uuid4().hex
    reference_salt = f"{trial_id}:reference"
    trial_salt = f"{trial_id}:reuse"
    reference_request = replace(edited_request, request_id=f"{trial_id}:reference")
    donor_request = replace(source_request, request_id=f"{trial_id}:donor")
    active_request = replace(edited_request, request_id=f"{trial_id}:intervention")

    # Delegate every role to the standard observer so full I/O is recorded.
    def observe(
        request: RequestSpec,
        role: str,
        cache_salt: str,
        vllm_xargs: dict[str, str],
    ) -> dict[str, Any]:
        return _observe_request(
            request,
            url=url,
            model=model,
            max_completion_tokens=max_completion_tokens,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            recorder=recorder,
            policy_metadata={
                "counterfactual_trial_id": trial_id,
                "counterfactual_role": role,
                "counterfactual_reuse_block_index": intervention.reused_block_index,
            },
            require_cacheselect_metrics=True,
            vllm_xargs=vllm_xargs,
            cache_salt=cache_salt,
        )

    reference = observe(
        reference_request,
        "reference",
        reference_salt,
        {"cacheselect_request_id": reference_request.request_id},
    )
    _require_fresh_full_recompute(reference, "reference")
    donor = observe(
        donor_request,
        "donor",
        trial_salt,
        {"cacheselect_request_id": donor_request.request_id},
    )
    _require_fresh_full_recompute(donor, "donor")
    active_xargs = {
        "cacheselect_request_id": active_request.request_id,
        "cacheselect_source_request_id": donor_request.request_id,
        "cacheselect_transition_id": intervention.transition_id,
        **intervention.to_vllm_xargs(),
    }
    active = observe(active_request, "intervention", trial_salt, active_xargs)
    evidence = validate_counterfactual_execution(
        intervention,
        block_size=block_size,
        server_metrics=active.get("server_metrics"),
    )
    result = score_single_block_intervention(
        intervention,
        execution_evidence=evidence,
        reference_output=reference.get("output_text"),
        intervention_output=active.get("output_text"),
        reference_quality_passed=bool((reference.get("quality") or {}).get("passed")),
        intervention_quality_passed=bool((active.get("quality") or {}).get("passed")),
    )
    label = counterfactual_block_label(result) if result.decision is not None else None
    return CounterfactualTrialResult(
        trial_id, intervention, reference, donor, active, evidence, result, label
    )
