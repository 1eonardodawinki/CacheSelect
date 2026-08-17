"""Benchmark-only labels for candidate KV-cache blocks."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Sequence

from benchmarks.schema import RequestSpec, RequestTransition, WorkloadTrace
from cacheselect.block_features import (
    CandidateBlockFeatures,
    extract_candidate_block_features,
    locate_changed_token_region,
)
from cacheselect.context_features import (
    CandidateContextFeatures,
    extract_candidate_context_features,
)
from cacheselect.reuse_opportunity import analyze_reuse_opportunity
from cacheselect.tokenization import rendered_chat_tokenization


class RepairDecision(str, Enum):
    """The correct action for one candidate block."""

    REPAIR = "repair"
    REUSE = "reuse"


class LabelSource(str, Enum):
    """How a benchmark established a block's correct action."""

    SYNTHETIC_DEPENDENCY = "synthetic_dependency"
    COUNTERFACTUAL_EXECUTION = "counterfactual_execution"


class DatasetSplit(str, Enum):
    """A leakage-safe role assigned to every complete benchmark trace."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


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
    context_features: CandidateContextFeatures | None = None


@dataclass(frozen=True)
class SegmentTokenSpan:
    """The rendered token interval occupied by one logical prompt segment."""

    segment_id: str
    token_start: int
    token_end: int


@dataclass(frozen=True)
class SplitCandidateBlock:
    """One labelled candidate together with its dataset partition."""

    example: LabeledCandidateBlock
    split: DatasetSplit


# Attach benchmark ground truth to one runtime feature vector.
def label_candidate_block(
    *,
    trace_id: str,
    transition_id: str,
    features: CandidateBlockFeatures,
    decision: RepairDecision,
    source: LabelSource,
    reason: str,
    context_features: CandidateContextFeatures | None = None,
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
        context_features=context_features,
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
        span = _character_span_to_tokens(
            token_offsets,
            segment_id=segment_id,
            character_start=character_start,
            character_end=character_end,
        )
        # Chat-template tokens after the query also determine the first output.
        if segment.kind == "query":
            span = replace(span, token_end=len(token_offsets))
        spans.append(span)
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
    context_features: Sequence[CandidateContextFeatures] | None = None,
) -> tuple[LabeledCandidateBlock, ...]:
    if transition.current_request_id != current_request.request_id:
        raise ValueError("transition does not target the supplied current request")
    dependent_spans = locate_dependent_token_spans(
        request=current_request,
        dependent_segment_ids=transition.ground_truth.dependent_segment_ids,
        rendered_prompt=rendered_prompt,
        token_offsets=token_offsets,
    )

    if context_features is not None and len(context_features) != len(features):
        raise ValueError("context and baseline feature counts differ")
    examples = []
    for index, feature_row in enumerate(features):
        if feature_row.current_token_count != len(token_offsets):
            raise ValueError("token offsets do not match feature token count")
        dependencies = _overlapping_dependencies(feature_row, dependent_spans)
        decision = RepairDecision.REPAIR if dependencies else RepairDecision.REUSE
        reason = (
            "Block overlaps annotated dependent segments: " + ", ".join(dependencies)
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
                context_features=(
                    context_features[index] if context_features is not None else None
                ),
            )
        )
    return tuple(examples)


# Build labelled examples from every annotated transition in one trace.
def build_trace_dependency_examples(
    *,
    trace: WorkloadTrace,
    tokenizer: object,
    block_size: int = 16,
    include_context_features: bool = False,
) -> tuple[LabeledCandidateBlock, ...]:
    requests_by_id = {request.request_id: request for request in trace.requests}
    examples = []
    for transition in trace.transitions:
        # Empty annotations mean the benchmark has not established safe labels.
        if not transition.ground_truth.dependent_segment_ids:
            continue
        previous_request = requests_by_id[transition.previous_request_id]
        current_request = requests_by_id[transition.current_request_id]
        previous = rendered_chat_tokenization(tokenizer, previous_request.messages)
        current = rendered_chat_tokenization(tokenizer, current_request.messages)
        changed_region = locate_changed_token_region(
            previous.token_ids,
            current.token_ids,
        )
        # Native APC can only reuse complete blocks before the first changed token.
        native_cached_tokens = changed_region.current_start // block_size * block_size
        opportunity = analyze_reuse_opportunity(
            previous.token_ids,
            current.token_ids,
            native_cached_tokens=native_cached_tokens,
            block_size=block_size,
        )
        features = extract_candidate_block_features(
            previous.token_ids,
            current.token_ids,
            opportunity,
        )
        context_features = (
            extract_candidate_context_features(
                previous.token_ids,
                current.token_ids,
                opportunity,
            )
            if include_context_features
            else None
        )
        examples.extend(
            build_synthetic_dependency_examples(
                trace_id=trace.trace_id,
                transition=transition,
                current_request=current_request,
                rendered_prompt=current.text,
                token_offsets=current.token_offsets,
                features=features,
                context_features=context_features,
            )
        )
    return tuple(examples)


# Assign whole traces to explicit partitions so related blocks cannot leak.
def assign_trace_splits(
    examples: Sequence[LabeledCandidateBlock],
    *,
    validation_trace_ids: set[str],
    test_trace_ids: set[str],
) -> tuple[SplitCandidateBlock, ...]:
    overlap = validation_trace_ids & test_trace_ids
    if overlap:
        raise ValueError(f"trace IDs occur in multiple splits: {sorted(overlap)}")
    known_trace_ids = {example.trace_id for example in examples}
    unknown = (validation_trace_ids | test_trace_ids) - known_trace_ids
    if unknown:
        raise ValueError(f"split references unknown trace IDs: {sorted(unknown)}")

    rows = []
    for example in examples:
        split = DatasetSplit.TRAIN
        if example.trace_id in validation_trace_ids:
            split = DatasetSplit.VALIDATION
        elif example.trace_id in test_trace_ids:
            split = DatasetSplit.TEST
        rows.append(SplitCandidateBlock(example=example, split=split))
    return tuple(rows)


# Flatten nested feature and label objects into one model-friendly table row.
def flatten_dataset_row(row: SplitCandidateBlock) -> dict[str, object]:
    example = row.example
    flattened: dict[str, object] = {
        "trace_id": example.trace_id,
        "transition_id": example.transition_id,
        "split": row.split.value,
        "decision": example.label.decision.value,
        "label_source": example.label.source.value,
        "label_reason": example.label.reason,
    }
    flattened.update(asdict(example.features))
    if example.context_features is not None:
        flattened.update(asdict(example.context_features))
    return flattened


# Save a flat labelled dataset that ordinary ML libraries can read directly.
def save_block_dataset_csv(
    rows: Sequence[SplitCandidateBlock],
    path: Path,
) -> None:
    if not rows:
        raise ValueError("cannot save an empty block dataset")
    context_presence = {row.example.context_features is not None for row in rows}
    if len(context_presence) != 1:
        raise ValueError("cannot mix feature schemas in one block dataset")
    flattened = [flatten_dataset_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
