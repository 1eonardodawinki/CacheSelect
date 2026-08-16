"""Plan and score single-block KV-reuse counterfactual experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from benchmarks.block_dataset import BlockRepairLabel, LabelSource, RepairDecision

COUNTERFACTUAL_BLOCK_KEY = "cacheselect_counterfactual_reuse_block_index"
# vLLM strips the xargs namespace when it serializes the internal plan.
COUNTERFACTUAL_PLAN_BLOCK_KEY = "counterfactual_reuse_block_index"
COUNTERFACTUAL_SELECTOR = "counterfactual_single_block"


@dataclass(frozen=True)
class SingleBlockIntervention:
    """One trial that reuses one candidate and repairs all remaining candidates."""

    trace_id: str
    transition_id: str
    candidate_block_indices: tuple[int, ...]
    reused_block_index: int

    # List the candidates that the intervention must recompute normally.
    @property
    def repaired_block_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in self.candidate_block_indices
            if index != self.reused_block_index
        )

    # Encode the one experimental choice as request-scoped vLLM metadata.
    def to_vllm_xargs(self) -> dict[str, str]:
        return {COUNTERFACTUAL_BLOCK_KEY: str(self.reused_block_index)}


@dataclass(frozen=True)
class CounterfactualExecutionEvidence:
    """Observed proof that vLLM executed one isolated block intervention."""

    valid: bool
    reason: str
    reused_batch_rows: int | None


@dataclass(frozen=True)
class CounterfactualLabelResult:
    """Auditable evidence and optional decision from one intervention."""

    intervention: SingleBlockIntervention
    valid_reference: bool
    valid_execution: bool
    decision: RepairDecision | None
    exact_output_match: bool
    word_similarity: float
    reason: str


# Check server metrics prove that exactly the selected full block was reused.
def validate_counterfactual_execution(
    intervention: SingleBlockIntervention,
    *,
    block_size: int,
    server_metrics: Mapping[str, Any] | None,
) -> CounterfactualExecutionEvidence:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    metrics = server_metrics or {}
    candidate_tokens = metrics.get("cacheselect_candidate_tokens")
    repair_tokens = metrics.get("cacheselect_repair_tokens")
    skipped_tokens = metrics.get("cacheselect_skipped_repair_tokens")
    copied_blocks = metrics.get("cacheselect_copied_blocks")
    copied_tokens = metrics.get("cacheselect_copied_tokens")
    reused_rows = metrics.get("cacheselect_reused_batch_rows")
    eligible = metrics.get("cacheselect_execution_eligible")
    compacted_built = metrics.get("cacheselect_compacted_batch_built")
    metadata_built = metrics.get("cacheselect_span_metadata_built")
    executed = metrics.get("cacheselect_compacted_batch_executed")
    selector = metrics.get("cacheselect_repair_selector")
    raw_plan = metrics.get("cacheselect_partial_reuse_plan")
    plan = raw_plan if isinstance(raw_plan, Mapping) else {}
    raw_candidates = plan.get("candidates") or ()
    plan_candidates = tuple(
        candidate for candidate in raw_candidates if isinstance(candidate, Mapping)
    )
    reported_target_values = [
        candidate.get("target_block_index") for candidate in plan_candidates
    ]
    reported_targets = tuple(
        sorted(target for target in reported_target_values if isinstance(target, int))
    )
    target_candidates = tuple(
        candidate
        for candidate in plan_candidates
        if candidate.get("target_block_index") == intervention.reused_block_index
    )
    expected_candidates = len(intervention.candidate_block_indices)
    expected_candidate_tokens = expected_candidates * block_size
    checks = (
        (selector == COUNTERFACTUAL_SELECTOR, "counterfactual selector was not used"),
        (plan.get("block_size") == block_size, "server block size did not match"),
        (
            plan.get("transition_id") == intervention.transition_id,
            "server transition did not match the intervention",
        ),
        (
            plan.get(COUNTERFACTUAL_PLAN_BLOCK_KEY)
            == intervention.reused_block_index,
            "server selected a different counterfactual block",
        ),
        (
            reported_targets == intervention.candidate_block_indices,
            "server candidate blocks did not match the intervention",
        ),
        (
            len(target_candidates) == 1
            and target_candidates[0].get("source_resident") is True,
            "selected source block was not uniquely resident",
        ),
        (
            candidate_tokens == expected_candidate_tokens,
            "resolved candidate-token count did not match the intervention",
        ),
        (
            repair_tokens == (expected_candidates - 1) * block_size,
            "vLLM did not repair every other candidate block",
        ),
        (skipped_tokens == block_size, "vLLM did not skip exactly one block"),
        (copied_blocks == expected_candidates, "not every candidate block was copied"),
        (
            copied_tokens == expected_candidate_tokens,
            "copied-token count did not match the intervention",
        ),
        (eligible is True, "partial execution did not pass its safety gate"),
        (
            metrics.get("cacheselect_execution_reason") == "eligible",
            "partial execution reported an unexpected reason",
        ),
        (reused_rows == block_size, "exactly one full block was not reused"),
        (compacted_built is True, "the compact partial batch was not built"),
        (metadata_built is True, "partial span metadata was not built"),
        (executed is True, "the compact partial forward did not execute"),
    )
    failure = next((reason for passed, reason in checks if not passed), None)
    return CounterfactualExecutionEvidence(
        valid=failure is None,
        reason=failure or "vLLM executed exactly one full-block intervention.",
        reused_batch_rows=reused_rows,
    )


# Create one isolated intervention for every unique candidate block.
def build_single_block_interventions(
    *,
    trace_id: str,
    transition_id: str,
    candidate_block_indices: Sequence[int],
) -> tuple[SingleBlockIntervention, ...]:
    if not trace_id or not transition_id:
        raise ValueError("trace and transition IDs must not be empty")
    ordered_indices = tuple(sorted(candidate_block_indices))
    if not ordered_indices:
        raise ValueError("counterfactual experiment requires candidate blocks")
    if any(index < 0 for index in ordered_indices):
        raise ValueError("candidate block indices must be non-negative")
    if len(set(ordered_indices)) != len(ordered_indices):
        raise ValueError("candidate block indices must be unique")
    return tuple(
        SingleBlockIntervention(
            trace_id=trace_id,
            transition_id=transition_id,
            candidate_block_indices=ordered_indices,
            reused_block_index=reused_block_index,
        )
        for reused_block_index in ordered_indices
    )


# Measure word-order agreement while allowing harmless wording differences.
def _word_similarity(reference: str, intervention: str) -> float:
    return SequenceMatcher(
        None, reference.split(), intervention.split(), autojunk=False
    ).ratio()


# Convert one reference/intervention output pair into a block-level decision.
def score_single_block_intervention(
    intervention: SingleBlockIntervention,
    *,
    execution_evidence: CounterfactualExecutionEvidence,
    reference_output: str | None,
    intervention_output: str | None,
    reference_quality_passed: bool,
    intervention_quality_passed: bool,
    require_exact_output_match: bool = False,
) -> CounterfactualLabelResult:
    reference = reference_output or ""
    candidate = intervention_output or ""
    exact_match = bool(reference) and reference == candidate
    similarity = _word_similarity(reference, candidate)

    if not reference or not reference_quality_passed:
        return CounterfactualLabelResult(
            intervention,
            False,
            execution_evidence.valid,
            None,
            exact_match,
            similarity,
            "Full-recompute reference did not pass its quality gate.",
        )
    if not execution_evidence.valid:
        return CounterfactualLabelResult(
            intervention,
            True,
            False,
            None,
            exact_match,
            similarity,
            execution_evidence.reason,
        )
    if require_exact_output_match and not exact_match:
        return CounterfactualLabelResult(
            intervention,
            True,
            True,
            None,
            False,
            similarity,
            "Non-exact output requires blinded manual review.",
        )
    decision = RepairDecision.REUSE
    reason = "Reusing this block preserved the configured output-quality gate."
    if not candidate or not intervention_quality_passed:
        decision = RepairDecision.REPAIR
        reason = "Reusing this block caused the task quality gate to fail."
    return CounterfactualLabelResult(
        intervention, True, True, decision, exact_match, similarity, reason
    )


# Convert valid counterfactual evidence into the shared training-label format.
def counterfactual_block_label(
    result: CounterfactualLabelResult,
) -> BlockRepairLabel:
    if not result.valid_reference or not result.valid_execution:
        raise ValueError("cannot label a block from an invalid experiment")
    if result.decision is None:
        raise ValueError("valid counterfactual experiment has no decision")
    return BlockRepairLabel(
        decision=result.decision,
        source=LabelSource.COUNTERFACTUAL_EXECUTION,
        reason=result.reason,
    )
