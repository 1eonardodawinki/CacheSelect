"""Load NVIDIA ChatRAG-Bench rows into CacheSelect's RAG task format."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag import MtragContext, MtragMessage, MtragTask

CHATRAG_SPLIT_SEED = "cacheselect-chatrag-v1"


def chatrag_conversation_split(conversation_id: str) -> DatasetSplit:
    """Assign complete ChatRAG conversations to a frozen 60/20/20 split."""
    if not conversation_id:
        raise ValueError("conversation ID must not be empty")
    digest = hashlib.sha256(
        f"{CHATRAG_SPLIT_SEED}:{conversation_id}".encode()
    ).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 60:
        return DatasetSplit.TRAIN
    if bucket < 80:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def _list(value: Any, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _messages(raw: Any, row: int) -> tuple[MtragMessage, ...]:
    parsed = []
    for index, message in enumerate(_list(raw, f"row {row} messages")):
        if not isinstance(message, dict):
            raise ValueError(f"row {row} messages[{index}] must be an object")
        role = _text(message.get("role"), f"row {row} messages[{index}].role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"row {row} messages[{index}].role is unsupported")
        parsed.append(
            MtragMessage(
                role=role,
                content=_text(
                    message.get("content"),
                    f"row {row} messages[{index}].content",
                ),
            )
        )
    if parsed[0].role != "user" or parsed[-1].role != "user":
        raise ValueError(f"row {row} messages must end with a user question")
    return tuple(parsed)


def chatrag_tasks(
    rows: Iterable[dict[str, Any]],
    *,
    subset: str,
    max_contexts: int = 5,
) -> tuple[MtragTask, ...]:
    """Convert ChatRAG rows while deriving stable conversation identities."""
    if not subset or max_contexts < 1:
        raise ValueError("subset and max_contexts must be valid")
    tasks = []
    conversation_id = ""
    previous_messages: tuple[MtragMessage, ...] = ()
    turn = 0
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"row {row_number} must be an object")
        messages = _messages(raw.get("messages"), row_number)
        document = _text(raw.get("document"), f"row {row_number} document")
        if len(messages) == 1:
            identity = f"{subset}\0{row_number}\0{document}\0{messages[0].content}"
            conversation_id = (
                f"{subset}-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
            )
            turn = 1
        elif not previous_messages or messages[: len(previous_messages)] != previous_messages:
            raise ValueError(f"row {row_number} does not extend its conversation")
        else:
            turn += 1
        previous_messages = messages

        contexts = []
        for index, context in enumerate(
            _list(raw.get("ctxs"), f"row {row_number} ctxs")[:max_contexts]
        ):
            if not isinstance(context, dict):
                raise ValueError(f"row {row_number} ctxs[{index}] must be an object")
            title = context.get("title", "")
            if not isinstance(title, str):
                raise ValueError(f"row {row_number} ctxs[{index}].title must be text")
            text = _text(context.get("text"), f"row {row_number} ctxs[{index}].text")
            context_id = hashlib.sha256(f"{title}\0{text}".encode()).hexdigest()
            contexts.append(MtragContext(context_id, title, text))

        answers = _list(raw.get("answers"), f"row {row_number} answers")
        tasks.append(
            MtragTask(
                task_id=f"{conversation_id}<::>{turn}",
                conversation_id=conversation_id,
                turn=turn,
                collection=f"chatrag-{subset}-{document}",
                contexts=tuple(contexts),
                messages=messages,
                target_text=_text(answers[0], f"row {row_number} answers[0]"),
            )
        )
    if not tasks:
        raise ValueError("ChatRAG input contains no tasks")
    return tuple(tasks)


def load_chatrag_tasks(
    path: Path,
    *,
    subset: str,
    max_contexts: int = 5,
) -> tuple[MtragTask, ...]:
    """Read a ChatRAG JSON array or Hugging Face Parquet export."""
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("reading ChatRAG Parquet requires pyarrow") from error
        rows = parquet.read_table(path).to_pylist()
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("ChatRAG input must contain a list of rows")
    return chatrag_tasks(rows, subset=subset, max_contexts=max_contexts)
