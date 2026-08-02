"""Model-tokenizer helpers used before a request reaches vLLM."""

from __future__ import annotations

from typing import Any


def rendered_chat_token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> list[int]:
    """Render a chat request exactly as the model tokenizer sees it."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
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
