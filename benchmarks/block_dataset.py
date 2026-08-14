"""Benchmark-only labels for candidate KV-cache blocks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cacheselect.block_features import CandidateBlockFeatures


class RepairDecision(str, Enum):
    """The correct action for one candidate block."""

    REPAIR = "repair"
    REUSE = "reuse"


class LabelSource(str, Enum):
    """How a benchmark established a block's correct action."""

    SYNTHETIC_DEPENDENCY = "synthetic_dependency"
    COUNTERFACTUAL_EXECUTION = "counterfactual_execution"


@dataclass(frozen=True)
class BlockRepairLabel:
    """Benchmark ground truth kept separate from runtime model inputs."""

    decision: RepairDecision
    source: LabelSource
    reason: str


@dataclass(frozen=True)
class LabeledCandidateBlock:
    """One candidate block's model inputs and benchmark-only answer."""

    trace_id: str
    transition_id: str
    features: CandidateBlockFeatures
    label: BlockRepairLabel


# Attach benchmark ground truth to one runtime feature vector.
def label_candidate_block(
    *,
    trace_id: str,
    transition_id: str,
    features: CandidateBlockFeatures,
    decision: RepairDecision,
    source: LabelSource,
    reason: str,
) -> LabeledCandidateBlock:
    if not trace_id:
        raise ValueError("trace_id must not be empty")
    if not transition_id:
        raise ValueError("transition_id must not be empty")
    if not reason:
        raise ValueError("label reason must not be empty")
    return LabeledCandidateBlock(
        trace_id=trace_id,
        transition_id=transition_id,
        features=features,
        label=BlockRepairLabel(
            decision=decision,
            source=source,
            reason=reason,
        ),
    )
