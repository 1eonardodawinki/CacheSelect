"""Plan and score single-block KV-reuse counterfactual experiments."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from benchmarks.block_dataset import BlockRepairLabel, LabelSource, RepairDecision

COUNTERFACTUAL_BLOCK_KEY = "cacheselect_counterfactual_reuse_block_index"


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
class CounterfactualLabelResult:
    """Auditable evidence and optional decision from one intervention."""

    intervention: SingleBlockIntervention
    valid_reference: bool
    decision: RepairDecision | None
    exact_output_match: bool
    word_similarity: float
    reason: str


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
    reference_output: str | None,
    intervention_output: str | None,
    reference_quality_passed: bool,
    intervention_quality_passed: bool,
) -> CounterfactualLabelResult:
    reference = reference_output or ""
    candidate = intervention_output or ""
    exact_match = bool(reference) and reference == candidate
    similarity = _word_similarity(reference, candidate)

    if not reference or not reference_quality_passed:
        return CounterfactualLabelResult(
            intervention,
            False,
            None,
            exact_match,
            similarity,
            "Full-recompute reference did not pass its quality gate.",
        )
    decision = RepairDecision.REUSE
    reason = "Reusing this block preserved the configured output-quality gate."
    if not candidate or not intervention_quality_passed:
        decision = RepairDecision.REPAIR
        reason = "Reusing this block caused the task quality gate to fail."
    return CounterfactualLabelResult(
        intervention, True, decision, exact_match, similarity, reason
    )


# Convert valid counterfactual evidence into the shared training-label format.
def counterfactual_block_label(
    result: CounterfactualLabelResult,
) -> BlockRepairLabel:
    if not result.valid_reference or result.decision is None:
        raise ValueError("cannot label a block from an invalid reference")
    return BlockRepairLabel(
        decision=result.decision,
        source=LabelSource.COUNTERFACTUAL_EXECUTION,
        reason=result.reason,
    )
