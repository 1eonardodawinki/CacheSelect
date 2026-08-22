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
from benchmarks.evaluation import compare_response_quality
from benchmarks.run_vllm_baseline import _observe_request
from benchmarks.schema import RequestSpec, RequestTransition
from observability.request_recorder import RequestRecorder


class CounterfactualReferenceQualityError(RuntimeError):
    """The uncached edited answer is unsuitable as counterfactual ground truth."""


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
    quality_comparison: dict[str, Any] | None = None
    reference_output_exact_match: bool | None = None


@dataclass(frozen=True)
class CounterfactualCandidateDiscovery:
    """Validated block candidates found in one vLLM discovery response."""

    trace_id: str
    transition_id: str
    block_size: int
    candidate_block_indices: tuple[int, ...]
    testable_block_indices: tuple[int, ...]
    excluded_output_block_index: int | None


@dataclass(frozen=True)
class CounterfactualDiscoveryRunResult:
    """Recorded source/edit observations and their validated candidate set."""

    discovery_id: str
    source_observation: dict[str, Any]
    edited_observation: dict[str, Any]
    discovery: CounterfactualCandidateDiscovery
    approved_output_exact_match: bool | None = None


@dataclass(frozen=True)
class CounterfactualTrialBatchResult:
    """Ordered results from every safe block in one discovery."""

    discovery: CounterfactualCandidateDiscovery
    trials: tuple[CounterfactualTrialResult, ...]
    target_block_indices: tuple[int, ...]


# Create one isolated trial instruction for each safe discovered target block.
def build_discovered_counterfactual_interventions(
    discovery: CounterfactualCandidateDiscovery,
    *,
    selected_block_indices: tuple[int, ...] | None = None,
) -> tuple[SingleBlockIntervention, ...]:
    candidates = discovery.candidate_block_indices
    testable = discovery.testable_block_indices
    for name, indices in (("candidate", candidates), ("testable", testable)):
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in indices
        ):
            raise ValueError(f"{name} block indices must be non-negative integers")
        if indices != tuple(sorted(set(indices))):
            raise ValueError(f"{name} block indices must be sorted and unique")
    if not discovery.trace_id or not discovery.transition_id:
        raise ValueError("trace and transition IDs must not be empty")
    if (
        not isinstance(discovery.block_size, int)
        or isinstance(discovery.block_size, bool)
        or discovery.block_size < 1
    ):
        raise ValueError("discovery block size must be positive")

    excluded = discovery.excluded_output_block_index
    if excluded is not None and excluded not in candidates:
        raise ValueError("excluded output block must belong to the candidates")
    expected_testable = tuple(index for index in candidates if index != excluded)
    if testable != expected_testable:
        raise ValueError("testable blocks do not match the discovery candidates")
    selected = testable if selected_block_indices is None else selected_block_indices
    if (
        any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in selected
        )
        or selected != tuple(sorted(set(selected)))
        or not set(selected).issubset(testable)
    ):
        raise ValueError("selected blocks must be a sorted subset of testable blocks")

    # Every instruction retains all peers so vLLM repairs every non-selected block.
    return tuple(
        SingleBlockIntervention(
            trace_id=discovery.trace_id,
            transition_id=discovery.transition_id,
            candidate_block_indices=candidates,
            reused_block_index=reused_block_index,
        )
        for reused_block_index in selected
    )


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


# Run a conservative source/edit pair that only discovers candidate KV blocks.
def run_counterfactual_candidate_discovery(
    *,
    trace_id: str,
    transition: RequestTransition,
    source_request: RequestSpec,
    edited_request: RequestSpec,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder,
    required_edited_output: str | None = None,
) -> CounterfactualDiscoveryRunResult:
    if (
        transition.previous_request_id != source_request.request_id
        or transition.current_request_id != edited_request.request_id
    ):
        raise ValueError("discovery requests do not match the transition")
    discovery_id = uuid4().hex
    cache_salt = f"{discovery_id}:discovery"
    source = replace(source_request, request_id=f"{discovery_id}:source")
    edited = replace(edited_request, request_id=f"{discovery_id}:edited")

    # Use the same observer so both discovery requests enter the request ledger.
    def observe(
        request: RequestSpec,
        role: str,
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
                "counterfactual_discovery_id": discovery_id,
                "counterfactual_role": role,
            },
            require_cacheselect_metrics=True,
            vllm_xargs=vllm_xargs,
            cache_salt=cache_salt,
        )

    source_observation = observe(
        source,
        "discovery_source",
        {"cacheselect_request_id": source.request_id},
    )
    _require_fresh_full_recompute(source_observation, "discovery source")
    edited_observation = observe(
        edited,
        "discovery_edit",
        {
            "cacheselect_request_id": edited.request_id,
            "cacheselect_source_request_id": source.request_id,
            "cacheselect_transition_id": transition.transition_id,
        },
    )
    if (
        required_edited_output is not None
        and edited_observation.get("finish_reason") != "stop"
    ):
        raise RuntimeError("discovery edit was truncated")
    approved_output_exact_match = (
        None
        if required_edited_output is None
        else edited_observation.get("output_text") == required_edited_output
    )
    discovery = extract_counterfactual_candidate_discovery(
        trace_id=trace_id,
        transition_id=transition.transition_id,
        observation=edited_observation,
    )

    metrics = edited_observation.get("server_metrics") or {}
    expected_tokens = len(discovery.candidate_block_indices) * discovery.block_size
    if discovery.candidate_block_indices and (
        metrics.get("cacheselect_repair_selector") != "full_block"
        or metrics.get("cacheselect_candidate_tokens") != expected_tokens
        or metrics.get("cacheselect_repair_tokens") != expected_tokens
        or metrics.get("cacheselect_skipped_repair_tokens") != 0
        or metrics.get("cacheselect_compacted_batch_executed") is True
    ):
        raise RuntimeError("discovery did not conservatively repair every candidate")
    # The source only donates KV; only the edited answer is experiment ground truth.
    if not bool((edited_observation.get("quality") or {}).get("passed")):
        raise CounterfactualReferenceQualityError(
            "counterfactual discovery edit failed its quality gate"
        )
    return CounterfactualDiscoveryRunResult(
        discovery_id,
        source_observation,
        edited_observation,
        discovery,
        approved_output_exact_match,
    )


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
    reference_observation: dict[str, Any] | None = None,
    required_reference_output: str | None = None,
    require_reference_output_match: bool = True,
    require_exact_output_match: bool = False,
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

    reference = reference_observation
    if reference is None:
        reference = observe(
            reference_request,
            "reference",
            reference_salt,
            {"cacheselect_request_id": reference_request.request_id},
        )
        _require_fresh_full_recompute(reference, "reference")
    if (
        required_reference_output is not None
        and reference.get("finish_reason") != "stop"
    ):
        raise RuntimeError("full-compute reference was truncated")
    reference_output_exact_match = (
        None
        if required_reference_output is None
        else reference.get("output_text") == required_reference_output
    )
    if require_reference_output_match and reference_output_exact_match is False:
        raise RuntimeError("full-compute output differs from its approved reference")
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
    if (
        required_reference_output is not None
        and active.get("finish_reason") != "stop"
    ):
        raise RuntimeError("counterfactual intervention output was truncated")
    evidence = validate_counterfactual_execution(
        intervention,
        block_size=block_size,
        server_metrics=active.get("server_metrics"),
    )
    quality_comparison = compare_response_quality(
        reference.get("quality") or {},
        active.get("quality") or {},
        edited_request.ground_truth,
    )
    result = score_single_block_intervention(
        intervention,
        execution_evidence=evidence,
        reference_output=reference.get("output_text"),
        intervention_output=active.get("output_text"),
        reference_quality_passed=quality_comparison["valid_reference"],
        intervention_quality_passed=quality_comparison["passed"],
        require_exact_output_match=require_exact_output_match,
    )
    label = counterfactual_block_label(result) if result.decision is not None else None
    return CounterfactualTrialResult(
        trial_id,
        intervention,
        reference,
        donor,
        active,
        evidence,
        result,
        label,
        quality_comparison,
        reference_output_exact_match,
    )


# Run every discovered block trial sequentially to keep its cache state isolated.
def run_discovered_counterfactual_trials(
    *,
    discovery: CounterfactualCandidateDiscovery,
    source_request: RequestSpec,
    edited_request: RequestSpec,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder,
    selected_block_indices: tuple[int, ...] | None = None,
    reference_observation: dict[str, Any] | None = None,
    required_reference_output: str | None = None,
    require_reference_output_match: bool = True,
    require_exact_output_match: bool = False,
) -> CounterfactualTrialBatchResult:
    interventions = build_discovered_counterfactual_interventions(
        discovery,
        selected_block_indices=selected_block_indices,
    )
    trials: list[CounterfactualTrialResult] = []
    for intervention in interventions:
        # Each call creates a new salt namespace and donor for this block only.
        trials.append(
            run_single_block_counterfactual_trial(
                source_request=source_request,
                edited_request=edited_request,
                intervention=intervention,
                block_size=discovery.block_size,
                url=url,
                model=model,
                max_completion_tokens=max_completion_tokens,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                recorder=recorder,
                reference_observation=reference_observation,
                required_reference_output=required_reference_output,
                require_reference_output_match=require_reference_output_match,
                require_exact_output_match=require_exact_output_match,
            )
        )
    if reference_observation is not None:
        check_id = uuid4().hex
        check_request = replace(
            edited_request,
            request_id=f"{check_id}:stability_reference",
        )
        final_reference = _observe_request(
            check_request,
            url=url,
            model=model,
            max_completion_tokens=max_completion_tokens,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            recorder=recorder,
            policy_metadata={
                "counterfactual_reference_check_id": check_id,
                "counterfactual_role": "stability_reference",
            },
            require_cacheselect_metrics=True,
            vllm_xargs={"cacheselect_request_id": check_request.request_id},
            cache_salt=f"{check_id}:stability_reference",
        )
        _require_fresh_full_recompute(final_reference, "stability reference")
        if final_reference.get("finish_reason") != "stop":
            raise CounterfactualReferenceQualityError(
                "final full-compute reference was truncated"
            )
        if final_reference.get("output_text") != reference_observation.get(
            "output_text"
        ):
            raise CounterfactualReferenceQualityError(
                "full-compute reference was unstable"
            )
    targets = tuple(
        intervention.reused_block_index for intervention in interventions
    )
    return CounterfactualTrialBatchResult(discovery, tuple(trials), targets)
