"""Typed loading and deterministic prompt rendering for IBM MTRAG tasks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.block_dataset import DatasetSplit


MTRAG_PROMPT_TEMPLATE_VERSION = 1
MTRAG_SPLIT_SEED = "cacheselect-mtrag-v1"
MTRAG_SYSTEM_PROMPT = (
    "Answer the current question using the retrieved passages and conversation "
    "history. If the passages do not contain enough information, say so."
)


# Hash a raw MTRAG release in chunks for selection and execution provenance.
def mtrag_source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Assign complete conversations to a stable split before inspecting examples.
def mtrag_conversation_split(
    conversation_id: str,
    *,
    seed: str = MTRAG_SPLIT_SEED,
) -> DatasetSplit:
    if not conversation_id or not seed:
        raise ValueError("conversation ID and split seed must not be empty")
    digest = hashlib.sha256(f"{seed}:{conversation_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return DatasetSplit.TRAIN
    if bucket < 85:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


@dataclass(frozen=True)
class MtragContext:
    """One retrieved passage attached to an MTRAG request."""

    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class MtragMessage:
    """One normalized user or assistant message in an MTRAG history."""

    role: str
    content: str


@dataclass(frozen=True)
class MtragTask:
    """One MTRAG generation task before model-specific tokenization."""

    task_id: str
    conversation_id: str
    turn: int
    collection: str
    contexts: tuple[MtragContext, ...]
    messages: tuple[MtragMessage, ...]
    target_text: str


# Require a non-empty string so malformed public rows fail visibly.
def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


# Normalize MTRAG's user/agent speaker names to chat API roles.
def _parse_message(raw: Any, field: str) -> MtragMessage:
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be an object")
    speaker = _required_string(raw.get("speaker"), f"{field}.speaker")
    roles = {"user": "user", "agent": "assistant"}
    if speaker not in roles:
        raise ValueError(f"{field}.speaker has unsupported value {speaker!r}")
    return MtragMessage(
        role=roles[speaker],
        content=_required_string(raw.get("text"), f"{field}.text"),
    )


# Parse one retrieved passage while preserving its stable document identity.
def _parse_context(raw: Any, field: str) -> MtragContext:
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be an object")
    title = raw.get("title", "")
    if not isinstance(title, str):
        raise ValueError(f"{field}.title must be a string")
    return MtragContext(
        document_id=_required_string(raw.get("document_id"), f"{field}.document_id"),
        title=title,
        text=_required_string(raw.get("text"), f"{field}.text"),
    )


# Parse one raw JSON object into the stable subset CacheSelect needs.
def _parse_task(raw: Any, line_number: int) -> MtragTask:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_number} must contain an object")
    turn = raw.get("turn")
    if isinstance(turn, bool) or not isinstance(turn, (int, str)):
        raise ValueError(f"line {line_number} turn must be a positive integer")
    try:
        normalized_turn = int(turn)
    except ValueError as error:
        raise ValueError(
            f"line {line_number} turn must be a positive integer"
        ) from error
    if normalized_turn < 1:
        raise ValueError(f"line {line_number} turn must be a positive integer")

    contexts = raw.get("contexts")
    messages = raw.get("input")
    targets = raw.get("targets")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"line {line_number} contexts must be a non-empty list")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_number} input must be a non-empty list")
    if not isinstance(targets, list) or len(targets) != 1:
        raise ValueError(f"line {line_number} targets must contain one response")

    # Public releases have used both collection and Collection spellings.
    collection = raw.get("collection", raw.get("Collection"))
    return MtragTask(
        task_id=_required_string(raw.get("task_id"), f"line {line_number} task_id"),
        conversation_id=_required_string(
            raw.get("conversation_id"), f"line {line_number} conversation_id"
        ),
        turn=normalized_turn,
        collection=_required_string(collection, f"line {line_number} collection"),
        contexts=tuple(
            _parse_context(context, f"line {line_number} contexts[{index}]")
            for index, context in enumerate(contexts)
        ),
        messages=tuple(
            _parse_message(message, f"line {line_number} input[{index}]")
            for index, message in enumerate(messages)
        ),
        target_text=_parse_message(
            targets[0], f"line {line_number} targets[0]"
        ).content,
    )


# Load MTRAG JSONL and reject inconsistent text for a repeated document ID.
def load_mtrag_tasks(path: Path) -> tuple[MtragTask, ...]:
    tasks: list[MtragTask] = []
    document_texts: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                task = _parse_task(json.loads(line), line_number)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number} is not valid JSON") from error
            for context in task.contexts:
                known_text = document_texts.setdefault(
                    context.document_id, context.text
                )
                if known_text != context.text:
                    raise ValueError(
                        f"document {context.document_id!r} has inconsistent text"
                    )
            tasks.append(task)
    if not tasks:
        raise ValueError("MTRAG input contains no tasks")
    return tuple(tasks)


# Pair consecutive turns within each conversation without crossing sessions.
def adjacent_mtrag_transitions(
    tasks: tuple[MtragTask, ...],
) -> tuple[tuple[MtragTask, MtragTask], ...]:
    conversations: dict[str, list[MtragTask]] = defaultdict(list)
    for task in tasks:
        conversations[task.conversation_id].append(task)

    transitions: list[tuple[MtragTask, MtragTask]] = []
    for conversation_id in sorted(conversations):
        ordered = sorted(conversations[conversation_id], key=lambda task: task.turn)
        if len({task.turn for task in ordered}) != len(ordered):
            raise ValueError(f"conversation {conversation_id!r} has duplicate turns")
        for previous, current in zip(ordered, ordered[1:]):
            if current.turn != previous.turn + 1:
                raise ValueError(f"conversation {conversation_id!r} has a turn gap")
            transitions.append((previous, current))
    return tuple(transitions)


# Render one task with retrieved passages attached only to its current question.
def render_mtrag_messages(task: MtragTask) -> list[dict[str, str]]:
    if task.messages[-1].role != "user":
        raise ValueError("MTRAG generation input must end with a user message")
    passages = []
    for index, context in enumerate(task.contexts, start=1):
        title = f"\nTitle: {context.title}" if context.title else ""
        passages.append(
            f"[Passage {index}; document_id={context.document_id}]{title}\n{context.text}"
        )
    augmented_question = (
        "Retrieved passages:\n\n"
        + "\n\n".join(passages)
        + f"\n\nCurrent question:\n{task.messages[-1].content}"
    )
    return [
        {"role": "system", "content": MTRAG_SYSTEM_PROMPT},
        *[
            {"role": message.role, "content": message.content}
            for message in task.messages[:-1]
        ],
        {"role": "user", "content": augmented_question},
    ]
