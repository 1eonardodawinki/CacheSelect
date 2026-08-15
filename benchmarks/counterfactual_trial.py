"""Run one isolated single-block KV-reuse counterfactual trial."""

from __future__ import annotations

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
