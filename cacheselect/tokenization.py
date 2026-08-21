"""Model-tokenizer helpers used before a request reaches vLLM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RenderedChatTokenization:
    """One rendered chat prompt with matching IDs and character offsets."""

    text: str
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]


# Normalize one tokenizer result into a plain, unbatched integer sequence.
def _single_token_ids(encoded: Any) -> list[int]:
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise ValueError("expected one rendered prompt, received a batch")
        token_ids = token_ids[0]
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError("tokenizer returned an unsupported input_ids value")
    if not token_ids:
        raise ValueError("tokenizer returned an empty rendered prompt")

    try:
        return [int(token_id) for token_id in token_ids]
    except (TypeError, ValueError) as error:
        raise TypeError("tokenizer returned non-integer token IDs") from error


# Render one chat request into the exact token IDs used by the model.
def rendered_chat_token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Render a chat request exactly as the model tokenizer sees it."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **dict(template_kwargs or {}),
    )
    return _single_token_ids(encoded)


# Render and retokenize one chat prompt to recover per-token text offsets.
def rendered_chat_tokenization(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    template_kwargs: Mapping[str, Any] | None = None,
) -> RenderedChatTokenization:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **dict(template_kwargs or {}),
    )
    if not isinstance(text, str) or not text:
        raise TypeError("tokenizer returned an invalid rendered prompt string")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = _single_token_ids(encoded)
    expected_ids = rendered_chat_token_ids(
        tokenizer,
        messages,
        template_kwargs=template_kwargs,
    )
    if token_ids != expected_ids:
        raise ValueError("text offsets do not match chat-template token IDs")

    offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(
        offsets[0][0], (list, tuple)
    ):
        if len(offsets) != 1:
            raise ValueError("expected one offset mapping, received a batch")
        offsets = offsets[0]
    if not isinstance(offsets, (list, tuple)) or len(offsets) != len(token_ids):
        raise ValueError("tokenizer returned an invalid offset mapping")
    try:
        normalized_offsets = tuple((int(start), int(end)) for start, end in offsets)
    except (TypeError, ValueError) as error:
        raise TypeError("tokenizer returned invalid token offsets") from error

    return RenderedChatTokenization(
        text=text,
        token_ids=tuple(token_ids),
        token_offsets=normalized_offsets,
    )
