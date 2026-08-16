"""Analyze natural non-prefix vLLM block overlap in IBM MTRAG JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmarks.mtrag import load_mtrag_tasks
from benchmarks.mtrag_coverage import analyze_mtrag_coverage


# Define the CPU-only coverage command independently from its analysis logic.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Tokenizer whose chat template and token boundaries will be measured.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of downloading a tokenizer missing from the local cache.",
    )
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    return args


# Load the public rows, run token analysis, and save one auditable JSON artifact.
def main() -> None:
    args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    tasks = load_mtrag_tasks(args.input)
    result = analyze_mtrag_coverage(
        tasks,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )
    artifact = {
        **result,
        "input": str(args.input),
        "model": args.model,
        "tokenizer_class": type(tokenizer).__name__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Analyzed {result['transition_count']} MTRAG transitions")
    print(
        "Found "
        f"{result['whole_source_candidate_transition_count']} transitions "
        "with executable aligned candidates"
    )
    print(f"Saved coverage artifact to {args.output}")


if __name__ == "__main__":
    main()
