"""Measure when skipping a cached gap beats one continuous model forward."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.run_vllm_baseline import _post_json


def _prompts(gap_blocks: int, flank_blocks: int, block_size: int) -> tuple[list[int], list[int]]:
    flank = flank_blocks * block_size
    gap = list(range(1000, 1000 + gap_blocks * block_size))
    return [100] * flank + gap + [102] * flank, [101] * flank + gap + [103] * flank


def _request(
    base_url: str,
    model: str,
    prompt: list[int],
    request_id: str,
    cache_salt: str,
    xargs: dict[str, str] | None = None,
) -> dict[str, Any]:
    return _post_json(
        f"{base_url.rstrip('/')}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "request_id": request_id,
            "cache_salt": cache_salt,
            "vllm_xargs": xargs or {"cacheselect_request_id": request_id},
            "add_special_tokens": False,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": 1,
            "return_token_ids": True,
        },
        api_key=None,
        timeout_seconds=900,
    )


def _ttft(response: dict[str, Any]) -> float:
    value = (response.get("metrics") or {}).get("time_to_first_token_ms")
    if not isinstance(value, (int, float)) or value <= 0:
        raise RuntimeError("vLLM response omitted a positive TTFT measurement")
    return float(value)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = []
    for gap in sorted({row["gap_blocks"] for row in rows}):
        selected = [row for row in rows if row["gap_blocks"] == gap]
        merged = statistics.median(row["merged_ttft_ms"] for row in selected)
        split = statistics.median(row["split_ttft_ms"] for row in selected)
        cells.append(
            {
                "gap_blocks": gap,
                "gap_tokens": selected[0]["gap_tokens"],
                "repetitions": len(selected),
                "median_merged_ttft_ms": merged,
                "median_split_ttft_ms": split,
                "median_split_delta_ms": split - merged,
                "median_split_speedup": merged / split,
                "split_faster_repetitions": sum(
                    row["split_ttft_ms"] < row["merged_ttft_ms"]
                    for row in selected
                ),
            }
        )
    return {
        "cells": cells,
        "first_measured_split_win_blocks": next(
            (cell["gap_blocks"] for cell in cells if cell["median_split_delta_ms"] < 0),
            None,
        ),
    }


def run(base_url: str, model: str, gaps: tuple[int, ...], flank: int, size: int, repetitions: int) -> dict[str, Any]:
    _, warmup = _prompts(1, flank, size)
    for index in range(2):
        _request(base_url, model, warmup, f"warmup-{index}", f"warmup-{index}")

    rows = []
    for gap in gaps:
        source, target = _prompts(gap, flank, size)
        for repetition in range(repetitions):
            tag = f"gap-{gap}-rep-{repetition}"
            merged = None
            if repetition % 2 == 0:
                merged = _request(base_url, model, target, f"{tag}-merged", f"{tag}-merged")
            donor_id, salt = f"{tag}-donor", f"{tag}-split"
            _request(base_url, model, source, donor_id, salt)
            split = _request(
                base_url,
                model,
                target,
                f"{tag}-split",
                salt,
                {
                    "cacheselect_request_id": f"{tag}-split",
                    "cacheselect_source_request_id": donor_id,
                    "cacheselect_transition_id": tag,
                },
            )
            if merged is None:
                merged = _request(base_url, model, target, f"{tag}-merged", f"{tag}-merged")
            metrics = split.get("metrics") or {}
            expected = gap * size
            observed = (
                metrics.get("cacheselect_candidate_tokens"),
                metrics.get("cacheselect_reused_batch_rows"),
                metrics.get("cacheselect_repair_tokens"),
                metrics.get("cacheselect_compute_span_count"),
                metrics.get("cacheselect_compacted_batch_executed"),
            )
            if observed != (expected, expected, 0, 2, True):
                raise RuntimeError(f"invalid split execution for {tag}: {observed}")
            rows.append(
                {
                    "gap_blocks": gap,
                    "gap_tokens": expected,
                    "repetition": repetition,
                    "merged_ttft_ms": _ttft(merged),
                    "split_ttft_ms": _ttft(split),
                    "split_forward_ms": metrics.get("cacheselect_forward_time_ms"),
                    "split_copy_ms": metrics.get("cacheselect_copy_time_ms"),
                    "output_exact": merged["choices"][0].get("token_ids")
                    == split["choices"][0].get("token_ids"),
                }
            )
    return {
        "schema_version": 1,
        "analysis": "partial-reuse-span-break-even",
        "model": model,
        "block_size": size,
        "flank_blocks": flank,
        "gap_blocks": list(gaps),
        "repetitions": repetitions,
        "all_outputs_exact": all(row["output_exact"] for row in rows),
        "rows": rows,
        **summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--gap-blocks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--flank-blocks", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min((*args.gap_blocks, args.flank_blocks, args.block_size, args.repetitions)) < 1:
        parser.error("all counts must be positive")
    result = run(
        args.base_url,
        args.model,
        tuple(args.gap_blocks),
        args.flank_blocks,
        args.block_size,
        args.repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for cell in result["cells"]:
        print(
            f"gap={cell['gap_blocks']} blocks merged={cell['median_merged_ttft_ms']:.2f}ms "
            f"split={cell['median_split_ttft_ms']:.2f}ms "
            f"speedup={cell['median_split_speedup']:.3f}x"
        )
    print(f"first_measured_split_win_blocks={result['first_measured_split_win_blocks']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
