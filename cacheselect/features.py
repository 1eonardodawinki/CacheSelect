"""Cheap transition features derived from already-tokenized prompts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _common_suffix_length(
    left: Sequence[int],
    right: Sequence[int],
    prefix_length: int,
) -> int:
    available = min(len(left), len(right)) - prefix_length
    length = 0
    while length < available and left[-1 - length] == right[-1 - length]:
        length += 1
    return length


def _multiset_overlap(left: Sequence[int], right: Sequence[int]) -> int:
    left_counts = Counter(left)
    right_counts = Counter(right)
    return sum((left_counts & right_counts).values())


def _ngram_set(tokens: Sequence[int], width: int) -> set[tuple[int, ...]]:
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    }


def token_transition_features(
    previous_tokens: Sequence[int],
    current_tokens: Sequence[int],
    *,
    ngram_width: int = 4,
) -> dict[str, int | float | bool]:
    """Compute linear-time features without rerunning the model."""
    previous_length = len(previous_tokens)
    current_length = len(current_tokens)
    minimum_length = min(previous_length, current_length)
    maximum_length = max(previous_length, current_length, 1)

    prefix_length = _common_prefix_length(previous_tokens, current_tokens)
    suffix_length = _common_suffix_length(
        previous_tokens,
        current_tokens,
        prefix_length,
    )
    same_position = sum(
        left == right for left, right in zip(previous_tokens, current_tokens)
    )
    overlap = _multiset_overlap(previous_tokens, current_tokens)

    previous_ngrams = _ngram_set(previous_tokens, ngram_width)
    current_ngrams = _ngram_set(current_tokens, ngram_width)
    ngram_union = previous_ngrams | current_ngrams
    ngram_intersection = previous_ngrams & current_ngrams

    return {
        "previous_token_count": previous_length,
        "current_token_count": current_length,
        "token_count_delta": current_length - previous_length,
        "exact_match": previous_tokens == current_tokens,
        "previous_is_exact_prefix": (
            previous_length <= current_length
            and prefix_length == previous_length
        ),
        "common_prefix_tokens": prefix_length,
        "common_prefix_ratio": prefix_length / maximum_length,
        "common_suffix_tokens": suffix_length,
        "common_suffix_ratio": suffix_length / maximum_length,
        "same_position_ratio": same_position / maximum_length,
        "token_multiset_overlap_ratio": overlap / maximum_length,
        "four_gram_jaccard": (
            len(ngram_intersection) / len(ngram_union)
            if ngram_union
            else 1.0
        ),
        "first_changed_token": (
            prefix_length if prefix_length < minimum_length else None
        ),
    }


def summarize_api_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract cheap structural fields while retaining no message text."""
    messages = payload.get("messages", [])
    documents = payload.get("documents") or []
    tools = payload.get("tools") or []

    return {
        "field_names": sorted(payload),
        "message_count": len(messages),
        "message_roles": [message.get("role") for message in messages],
        "message_content_lengths": [
            len(_content_text(message.get("content"))) for message in messages
        ],
        "document_count": len(documents),
        "tool_count": len(tools),
        "has_structured_output": bool(
            payload.get("response_format") or payload.get("structured_outputs")
        ),
        "has_chat_template_kwargs": bool(payload.get("chat_template_kwargs")),
        "priority": payload.get("priority", 0),
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        return "".join(str(part) for part in content)
    return "" if content is None else str(content)
