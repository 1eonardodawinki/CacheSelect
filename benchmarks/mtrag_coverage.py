"""Measure natural vLLM block-reuse opportunities in MTRAG transitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from benchmarks.mtrag import (
    MTRAG_PROMPT_TEMPLATE_VERSION,
    MtragTask,
    adjacent_mtrag_transitions,
    render_mtrag_messages,
)
from cacheselect.block_features import locate_changed_token_region
from cacheselect.reuse_opportunity import analyze_reuse_opportunity
from cacheselect.tokenization import rendered_chat_token_ids


MTRAG_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


# Return document identities retained across one natural adjacent-turn edit.
def _shared_document_ids(previous: MtragTask, current: MtragTask) -> tuple[str, ...]:
    previous_ids = {context.document_id for context in previous.contexts}
    current_ids = {context.document_id for context in current.contexts}
    return tuple(sorted(previous_ids & current_ids))


# Analyze one adjacent request pair with the exact selected chat tokenizer.
def _analyze_transition(
    previous: MtragTask,
    current: MtragTask,
    *,
    tokenizer: Any,
    block_size: int,
) -> dict[str, Any]:
    if previous.collection != current.collection:
        raise ValueError("one MTRAG conversation changed collection")
    previous_tokens = rendered_chat_token_ids(
        tokenizer,
        render_mtrag_messages(previous),
        template_kwargs=MTRAG_CHAT_TEMPLATE_KWARGS,
    )
    current_tokens = rendered_chat_token_ids(
        tokenizer,
        render_mtrag_messages(current),
        template_kwargs=MTRAG_CHAT_TEMPLATE_KWARGS,
    )
    changed_region = locate_changed_token_region(previous_tokens, current_tokens)
    # This is a CPU estimate: real APC may lose resident blocks under load.
    estimated_native_cached_tokens = (
        changed_region.current_start // block_size * block_size
    )
    opportunity = analyze_reuse_opportunity(
        previous_tokens,
        current_tokens,
        native_cached_tokens=estimated_native_cached_tokens,
        block_size=block_size,
    )
    previous_order = tuple(context.document_id for context in previous.contexts)
    current_order = tuple(context.document_id for context in current.contexts)
    return {
        "conversation_id": current.conversation_id,
        "collection": current.collection,
        "previous_task_id": previous.task_id,
        "current_task_id": current.task_id,
        "previous_turn": previous.turn,
        "current_turn": current.turn,
        "shared_document_ids": list(_shared_document_ids(previous, current)),
        "same_document_set": set(previous_order) == set(current_order),
        "same_document_order": previous_order == current_order,
        "reuse_opportunity": opportunity.to_dict(),
    }


# Aggregate per-transition natural overlap without assigning safety labels.
def analyze_mtrag_coverage(
    tasks: Sequence[MtragTask],
    *,
    tokenizer: Any,
    block_size: int = 16,
) -> dict[str, Any]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not tasks:
        raise ValueError("MTRAG coverage requires at least one task")

    transitions = adjacent_mtrag_transitions(tuple(tasks))
    rows = [
        _analyze_transition(
            previous,
            current,
            tokenizer=tokenizer,
            block_size=block_size,
        )
        for previous, current in transitions
    ]
    opportunities = [row["reuse_opportunity"] for row in rows]
    return {
        "schema_version": 1,
        "analysis": "mtrag-natural-block-coverage",
        "prompt_template_version": MTRAG_PROMPT_TEMPLATE_VERSION,
        "block_size": block_size,
        "task_count": len(tasks),
        "conversation_count": len({task.conversation_id for task in tasks}),
        "transition_count": len(rows),
        "shared_document_transition_count": sum(
            bool(row["shared_document_ids"]) for row in rows
        ),
        "candidate_transition_count": sum(
            opportunity["candidate_block_count"] > 0 for opportunity in opportunities
        ),
        "whole_source_candidate_transition_count": sum(
            opportunity["whole_source_block_count"] > 0 for opportunity in opportunities
        ),
        "candidate_block_count": sum(
            opportunity["candidate_block_count"] for opportunity in opportunities
        ),
        "whole_source_block_count": sum(
            opportunity["whole_source_block_count"] for opportunity in opportunities
        ),
        "repacking_required_block_count": sum(
            opportunity["repacking_required_block_count"]
            for opportunity in opportunities
        ),
        "moved_candidate_block_count": sum(
            opportunity["moved_candidate_block_count"] for opportunity in opportunities
        ),
        "estimated_native_cached_tokens": sum(
            opportunity["native_cached_tokens"] for opportunity in opportunities
        ),
        "candidate_tokens": sum(
            opportunity["candidate_token_count"] for opportunity in opportunities
        ),
        "transitions": rows,
    }
