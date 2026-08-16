"""Convert selected IBM MTRAG tasks into CacheSelect benchmark traces."""

from __future__ import annotations

from benchmarks.mtrag import (
    MTRAG_PROMPT_TEMPLATE_VERSION,
    MTRAG_SYSTEM_PROMPT,
    MtragTask,
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
    )


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
