"""Benchmark-only labels for candidate KV-cache blocks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from benchmarks.schema import RequestSpec, RequestTransition
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


@dataclass(frozen=True)
class SegmentTokenSpan:
    """The rendered token interval occupied by one logical prompt segment."""

    segment_id: str
    token_start: int
    token_end: int


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


# Find one exact and unambiguous segment occurrence in the rendered prompt.
def _locate_segment_characters(
    rendered_prompt: str,
    *,
    segment_id: str,
    content: str,
) -> tuple[int, int]:
    start = rendered_prompt.find(content)
    if start < 0:
        raise ValueError(f"segment {segment_id!r} is absent from rendered prompt")
    if rendered_prompt.find(content, start + 1) >= 0:
        raise ValueError(f"segment {segment_id!r} is ambiguous in rendered prompt")
    return start, start + len(content)


# Convert one character interval into the token interval that overlaps it.
def _character_span_to_tokens(
    token_offsets: Sequence[tuple[int, int]],
    *,
    segment_id: str,
    character_start: int,
    character_end: int,
) -> SegmentTokenSpan:
    overlapping = [
        token_index
        for token_index, (token_start, token_end) in enumerate(token_offsets)
        if token_start < character_end and character_start < token_end
    ]
    if not overlapping:
        raise ValueError(f"segment {segment_id!r} does not overlap any prompt token")
    return SegmentTokenSpan(
        segment_id=segment_id,
        token_start=overlapping[0],
        token_end=overlapping[-1] + 1,
    )


# Locate every benchmark-annotated dependency in the rendered token sequence.
def locate_dependent_token_spans(
    *,
    request: RequestSpec,
    dependent_segment_ids: Sequence[str],
    rendered_prompt: str,
    token_offsets: Sequence[tuple[int, int]],
) -> tuple[SegmentTokenSpan, ...]:
    if not dependent_segment_ids:
        raise ValueError("transition has no annotated dependent segments")
    segments_by_id = {segment.segment_id: segment for segment in request.segments}
    if len(segments_by_id) != len(request.segments):
        raise ValueError("request contains duplicate segment IDs")

    spans = []
    for segment_id in dependent_segment_ids:
        segment = segments_by_id.get(segment_id)
        if segment is None:
            raise ValueError(f"dependent segment {segment_id!r} is absent from request")
        character_start, character_end = _locate_segment_characters(
            rendered_prompt,
            segment_id=segment_id,
            content=segment.content,
        )
        spans.append(
            _character_span_to_tokens(
                token_offsets,
                segment_id=segment_id,
                character_start=character_start,
                character_end=character_end,
            )
        )
    return tuple(spans)


# Report which annotated semantic dependencies overlap one physical KV block.
def _overlapping_dependencies(
    features: CandidateBlockFeatures,
    dependent_spans: Sequence[SegmentTokenSpan],
) -> tuple[str, ...]:
    block_start = features.candidate_block_index * features.block_size
    block_end = min(
        block_start + features.block_size,
        features.current_token_count,
    )
    return tuple(
        span.segment_id
        for span in dependent_spans
        if block_start < span.token_end and span.token_start < block_end
    )


# Build deterministic training rows from a synthetic dependency annotation.
def build_synthetic_dependency_examples(
    *,
    trace_id: str,
    transition: RequestTransition,
    current_request: RequestSpec,
    rendered_prompt: str,
    token_offsets: Sequence[tuple[int, int]],
    features: Sequence[CandidateBlockFeatures],
) -> tuple[LabeledCandidateBlock, ...]:
    if transition.current_request_id != current_request.request_id:
        raise ValueError("transition does not target the supplied current request")
    dependent_spans = locate_dependent_token_spans(
        request=current_request,
        dependent_segment_ids=transition.ground_truth.dependent_segment_ids,
        rendered_prompt=rendered_prompt,
        token_offsets=token_offsets,
    )

    examples = []
    for feature_row in features:
        if feature_row.current_token_count != len(token_offsets):
            raise ValueError("token offsets do not match feature token count")
        dependencies = _overlapping_dependencies(feature_row, dependent_spans)
        decision = RepairDecision.REPAIR if dependencies else RepairDecision.REUSE
        reason = (
            "Block overlaps annotated dependent segments: "
            + ", ".join(dependencies)
            if dependencies
            else "Block does not overlap an annotated dependent segment."
        )
        examples.append(
            label_candidate_block(
                trace_id=trace_id,
                transition_id=transition.transition_id,
                features=feature_row,
                decision=decision,
                source=LabelSource.SYNTHETIC_DEPENDENCY,
                reason=reason,
            )
        )
    return tuple(examples)
