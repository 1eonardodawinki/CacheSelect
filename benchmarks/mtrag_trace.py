"""Convert selected IBM MTRAG tasks into CacheSelect benchmark traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import (
    MTRAG_PROMPT_TEMPLATE_VERSION,
    MTRAG_SPLIT_SEED,
    MTRAG_SYSTEM_PROMPT,
    MtragTask,
    mtrag_conversation_split,
    render_mtrag_messages,
)
from benchmarks.schema import (
    PromptSegment,
    ReferenceSimilarityGate,
    RequestGroundTruth,
    RequestSpec,
    RequestTransition,
    TransitionGroundTruth,
    WorkloadTrace,
)

# Build one request without exposing its answer or segments to vLLM.
def build_mtrag_request_spec(
    task: MtragTask,
    *,
    quality_gate: ReferenceSimilarityGate,
) -> RequestSpec:
    if not task.target_text:
        raise ValueError("MTRAG request requires a reference answer")
    history_segments = [
        PromptSegment(
            segment_id=f"message:{index}",
            role=message.role,
            kind="conversation_history",
            version=1,
            content=message.content,
        )
        for index, message in enumerate(task.messages[:-1])
    ]
    document_segments = [
        PromptSegment(
            segment_id=f"document:{context.document_id}",
            role="user",
            kind="retrieved_document",
            version=1,
            content=context.text,
        )
        for context in task.contexts
    ]
    return RequestSpec(
        request_id=task.task_id,
        workload="mtrag_rag",
        sequence_index=task.turn,
        messages=render_mtrag_messages(task),
        segments=[
            PromptSegment(
                segment_id="system",
                role="system",
                kind="instruction",
                version=MTRAG_PROMPT_TEMPLATE_VERSION,
                content=MTRAG_SYSTEM_PROMPT,
            ),
            *history_segments,
            *document_segments,
            PromptSegment(
                segment_id=f"message:{len(task.messages) - 1}",
                role="user",
                kind="query",
                version=task.turn,
                content=task.messages[-1].content,
            ),
        ],
        ground_truth=RequestGroundTruth(
            expected_answer=task.target_text,
            requirements=[],
            notes=(f"IBM MTRAG task {task.task_id} from collection {task.collection}."),
            reference_similarity_gate=quality_gate,
        ),
        # Keep the completion budget for the answer rather than Qwen3 reasoning.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


@dataclass(frozen=True)
class MtragCounterfactualCase:
    """One audited MTRAG transition and its complete frozen block plan."""

    trace: WorkloadTrace
    split: DatasetSplit
    collection: str
    block_size: int
    expected_candidate_block_indices: tuple[int, ...]
    expected_testable_block_indices: tuple[int, ...]
    target_block_indices: tuple[int, ...]


# Identify additions, removals, edits, and moves between ordered segment lists.
def _changed_segment_ids(
    previous: RequestSpec,
    current: RequestSpec,
) -> list[str]:
    previous_by_id = {
        segment.segment_id: (index, segment)
        for index, segment in enumerate(previous.segments)
    }
    current_by_id = {
        segment.segment_id: (index, segment)
        for index, segment in enumerate(current.segments)
    }
    if len(previous_by_id) != len(previous.segments) or len(current_by_id) != len(
        current.segments
    ):
        raise ValueError("MTRAG request contains duplicate segment IDs")
    return sorted(
        segment_id
        for segment_id in previous_by_id.keys() | current_by_id.keys()
        if previous_by_id.get(segment_id) != current_by_id.get(segment_id)
    )


# Pair two natural adjacent tasks into one explicit CacheSelect transition.
def build_mtrag_transition_trace(
    previous_task: MtragTask,
    current_task: MtragTask,
    *,
    quality_gate: ReferenceSimilarityGate,
) -> WorkloadTrace:
    if previous_task.conversation_id != current_task.conversation_id:
        raise ValueError("MTRAG transition cannot cross conversations")
    if previous_task.collection != current_task.collection:
        raise ValueError("MTRAG transition cannot cross collections")
    if current_task.turn != previous_task.turn + 1:
        raise ValueError("MTRAG transition tasks must be consecutive")
    previous = build_mtrag_request_spec(previous_task, quality_gate=quality_gate)
    current = build_mtrag_request_spec(current_task, quality_gate=quality_gate)
    transition_id = (
        f"mtrag:{current_task.conversation_id}:"
        f"{previous_task.turn}-to-{current_task.turn}"
    )
    return WorkloadTrace(
        trace_id=transition_id,
        workload="mtrag_rag",
        description="One natural adjacent IBM MTRAG request transition.",
        requests=[previous, current],
        transitions=[
            RequestTransition(
                transition_id=transition_id,
                previous_request_id=previous.request_id,
                current_request_id=current.request_id,
                ground_truth=TransitionGroundTruth(
                    change_type="natural_adjacent_rag_turn",
                    changed_segment_ids=_changed_segment_ids(previous, current),
                    expected_native_behavior="prefix_hit_until_first_rendered_change",
                    notes="Natural MTRAG transition; block safety is measured causally.",
                    dependent_segment_ids=[],
                ),
            )
        ],
    )


# Resolve the audited manifest against the unchanged public MTRAG tasks.
def build_mtrag_counterfactual_cases(
    tasks: Sequence[MtragTask],
    manifest: Mapping[str, object],
    *,
    quality_gate: ReferenceSimilarityGate,
    approved_task_ids: frozenset[str],
    expected_model: str,
) -> tuple[MtragCounterfactualCase, ...]:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("selection") != "mtrag-audited-counterfactual-pilot"
        or manifest.get("quality_calibration_id") != quality_gate.calibration_id
        or manifest.get("source_prompt_template_version")
        != MTRAG_PROMPT_TEMPLATE_VERSION
        or manifest.get("source_model") != expected_model
        or manifest.get("split_seed") != MTRAG_SPLIT_SEED
    ):
        raise ValueError("MTRAG pilot provenance is incompatible")
    if not approved_task_ids:
        raise ValueError("MTRAG pilot requires approved reference tasks")
    block_size = manifest.get("block_size")
    rows = manifest.get("transitions")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("MTRAG pilot block size must be positive")
    if not isinstance(rows, list) or manifest.get("transition_count") != len(rows):
        raise ValueError("MTRAG pilot transition count is inconsistent")

    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("MTRAG input contains duplicate task IDs")
    cases = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("MTRAG pilot transition must be an object")
        try:
            previous = tasks_by_id[row["previous_task_id"]]
            current = tasks_by_id[row["current_task_id"]]
        except (KeyError, TypeError) as error:
            raise ValueError("MTRAG pilot references a missing task") from error
        if (
            row.get("conversation_id") != current.conversation_id
            or row.get("collection") != current.collection
            or current.task_id not in approved_task_ids
        ):
            raise ValueError("MTRAG pilot task identity or approval is invalid")
        split = mtrag_conversation_split(current.conversation_id)
        if row.get("split") != split.value:
            raise ValueError("MTRAG pilot conversation split is inconsistent")
        shared_documents = sorted(
            {context.document_id for context in previous.contexts}
            & {context.document_id for context in current.contexts}
        )
        if row.get("shared_document_ids") != shared_documents:
            raise ValueError("MTRAG pilot shared documents are inconsistent")

        aligned = row.get("aligned_candidate_block_indices")
        testable = row.get("testable_block_indices")
        targets = row.get("target_block_indices")
        excluded = row.get("excluded_output_block_index")
        index_lists = (aligned, testable, targets)
        if any(not isinstance(indices, list) or not indices for indices in index_lists):
            raise ValueError("MTRAG pilot block lists must not be empty")
        if any(
            any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or indices != sorted(set(indices))
            for indices in index_lists
        ):
            raise ValueError("MTRAG pilot block lists must be sorted and unique")
        if excluded is not None and excluded not in aligned:
            raise ValueError("excluded output block is not an aligned candidate")
        if testable != [index for index in aligned if index != excluded]:
            raise ValueError("testable blocks do not match aligned candidates")
        max_targets = manifest.get("max_target_blocks")
        if (
            isinstance(max_targets, bool)
            or not isinstance(max_targets, int)
            or not set(targets).issubset(testable)
            or len(targets) > max_targets
        ):
            raise ValueError("pilot targets are not a bounded testable subset")

        cases.append(
            MtragCounterfactualCase(
                trace=build_mtrag_transition_trace(
                    previous,
                    current,
                    quality_gate=quality_gate,
                ),
                split=split,
                collection=current.collection,
                block_size=block_size,
                expected_candidate_block_indices=tuple(aligned),
                expected_testable_block_indices=tuple(testable),
                target_block_indices=tuple(targets),
            )
        )
    if (
        manifest.get("total_testable_blocks")
        != sum(len(case.expected_testable_block_indices) for case in cases)
        or manifest.get("total_target_blocks")
        != sum(len(case.target_block_indices) for case in cases)
    ):
        raise ValueError("MTRAG pilot block totals are inconsistent")
    return tuple(cases)
