"""Generate tokenizer-aware prompt-length calibration traces."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.length_calibration import (
    EDIT_POSITIONS,
    build_length_calibration_trace,
)
from benchmarks.schema import save_trace
from cacheselect.tokenization import rendered_chat_token_ids


def _rendered_token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    return len(rendered_chat_token_ids(tokenizer, messages))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-prompt-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def rendered_token_count(messages: list[dict[str, str]]) -> int:
        return _rendered_token_count(tokenizer, messages)

    for edit_position in EDIT_POSITIONS:
        trace = build_length_calibration_trace(
            target_prompt_tokens=args.target_prompt_tokens,
            edit_position=edit_position,
            token_counter=rendered_token_count,
            tokenizer_name=args.model,
        )
        path = args.output_dir / (
            f"tokens-{args.target_prompt_tokens}-{edit_position}.json"
        )
        save_trace(trace, path)
        counts = [rendered_token_count(request.messages) for request in trace.requests]
        print(f"Saved {path}: rendered_prompt_tokens={counts}")


if __name__ == "__main__":
    main()
