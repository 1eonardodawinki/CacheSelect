"""Build and summarize controlled GDN reuse break-even experiments."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


TokenizePrompt = Callable[[str], Sequence[int]]


@dataclass(frozen=True)
class HybridGDNBreakEvenCondition:
    """One requested reuse amount and repetition in the experiment."""

    reused_block_count: int
    repetition: int


@dataclass(frozen=True)
class HybridGDNBreakEvenPromptPair:
    """A calibrated source/edit pair with exact reusable block geometry."""

    source_prompt: str
    target_prompt: str
    prompt_token_count: int
    full_block_count: int
    reused_block_count: int
    shared_prefix_tokens: int
    shared_suffix_tokens: int
    filler_repetitions: int


# Interleave reuse amounts so warm-server drift is spread across conditions.
def build_hybrid_gdn_breakeven_conditions(
    reused_block_counts: tuple[int, ...],
    repetitions: int,
) -> tuple[HybridGDNBreakEvenCondition, ...]:
    """Return a deterministic and validated execution order."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not reused_block_counts:
        raise ValueError("reused_block_counts must not be empty")
    if len(set(reused_block_counts)) != len(reused_block_counts):
        raise ValueError("reused_block_counts must be unique")
    if any(count < 1 for count in reused_block_counts):
        raise ValueError("every reused block count must be positive")
    return tuple(
        HybridGDNBreakEvenCondition(count, repetition)
        for repetition in range(1, repetitions + 1)
        for count in reused_block_counts
    )


# Render prompts that differ only near the beginning and share the long suffix.
def _render_break_even_prompts(filler_repetitions: int) -> tuple[str, str]:
    """Return a source and edited prompt at one candidate length."""
    if filler_repetitions < 0:
        raise ValueError("filler_repetitions must not be negative")
    stable = (
        " Stable context follows the same deterministic order."
        * filler_repetitions
    )
    instruction = "\nReply with exactly: GDN break-even complete. /no_think"
    return (
        f"GDN break-even marker A.{stable}{instruction}",
        f"GDN break-even marker B.{stable}{instruction}",
    )


# Count the unchanged token prefix and suffix around the single early edit.
def _shared_token_edges(
    source_tokens: Sequence[int],
    target_tokens: Sequence[int],
) -> tuple[int, int]:
    """Return common-prefix and non-overlapping common-suffix lengths."""
    shared_prefix = 0
    maximum_prefix = min(len(source_tokens), len(target_tokens))
    while (
        shared_prefix < maximum_prefix
        and source_tokens[shared_prefix] == target_tokens[shared_prefix]
    ):
        shared_prefix += 1

    shared_suffix = 0
    maximum_suffix = maximum_prefix - shared_prefix
    while (
        shared_suffix < maximum_suffix
        and source_tokens[-shared_suffix - 1] == target_tokens[-shared_suffix - 1]
    ):
        shared_suffix += 1
    return shared_prefix, shared_suffix


# Calibrate with the live tokenizer so each condition has exactly N reusable blocks.
def calibrate_hybrid_gdn_breakeven_prompt(
    *,
    reused_block_count: int,
    block_size: int,
    tokenize_prompt: TokenizePrompt,
) -> HybridGDNBreakEvenPromptPair:
    """Find a prompt pair with one changed block followed by N shared blocks."""
    if reused_block_count < 1:
        raise ValueError("reused_block_count must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    required_full_blocks = reused_block_count + 1

    # Exponential search first avoids assuming how many tokens one filler adds.
    low = 0
    high = 1
    while True:
        source_prompt, _ = _render_break_even_prompts(high)
        if len(tokenize_prompt(source_prompt)) // block_size >= required_full_blocks:
            break
        low = high + 1
        high *= 2
        if high > 1_000_000:
            raise ValueError("could not calibrate the requested prompt length")

    # Binary search chooses the shortest prompt with the required full blocks.
    while low < high:
        middle = (low + high) // 2
        source_prompt, _ = _render_break_even_prompts(middle)
        if len(tokenize_prompt(source_prompt)) // block_size < required_full_blocks:
            low = middle + 1
        else:
            high = middle

    source_prompt, target_prompt = _render_break_even_prompts(low)
    source_tokens = tuple(tokenize_prompt(source_prompt))
    target_tokens = tuple(tokenize_prompt(target_prompt))
    if len(source_tokens) != len(target_tokens):
        raise ValueError("source and target prompts must have equal token counts")
    full_block_count = len(target_tokens) // block_size
    if full_block_count != required_full_blocks:
        raise ValueError("tokenizer skipped the required full-block geometry")
    shared_prefix, shared_suffix = _shared_token_edges(source_tokens, target_tokens)
    if shared_prefix >= block_size:
        raise ValueError("the edit must occur inside the first logical block")
    if source_tokens[:block_size] == target_tokens[:block_size]:
        raise ValueError("the first logical block must differ")
    shared_start = block_size
    shared_stop = required_full_blocks * block_size
    if source_tokens[shared_start:shared_stop] != target_tokens[shared_start:shared_stop]:
        raise ValueError("the requested reusable logical blocks are not identical")
    if shared_suffix < reused_block_count * block_size:
        raise ValueError("the shared suffix does not cover every reusable block")
    return HybridGDNBreakEvenPromptPair(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        prompt_token_count=len(target_tokens),
        full_block_count=full_block_count,
        reused_block_count=reused_block_count,
        shared_prefix_tokens=shared_prefix,
        shared_suffix_tokens=shared_suffix,
        filler_repetitions=low,
    )


# Aggregate repeated timings and identify the first consistently faster reuse size.
def summarize_hybrid_gdn_breakeven(
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report correctness and latency break-even points by reused block count."""
    if not trials:
        raise ValueError("trials must not be empty")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in trials:
        block_count = trial.get("reused_block_count")
        speedup = trial.get("speedup")
        ttft_speedup = trial.get("ttft_speedup")
        reuse_fraction = trial.get("reuse_fraction")
        if not isinstance(block_count, int) or block_count < 1:
            raise ValueError("trial has an invalid reused_block_count")
        if not isinstance(speedup, (int, float)) or speedup <= 0:
            raise ValueError("trial has an invalid speedup")
        if not isinstance(ttft_speedup, (int, float)) or ttft_speedup <= 0:
            raise ValueError("trial has an invalid TTFT speedup")
        if not isinstance(reuse_fraction, (int, float)) or not 0 < reuse_fraction < 1:
            raise ValueError("trial has an invalid reuse_fraction")
        grouped.setdefault(block_count, []).append(trial)

    cells = []
    for block_count, rows in sorted(grouped.items()):
        speedups = [float(row["speedup"]) for row in rows]
        ttft_speedups = [float(row["ttft_speedup"]) for row in rows]
        all_exact = all(row.get("exact_output_match") is True for row in rows)
        median_speedup = statistics.median(speedups)
        median_ttft_speedup = statistics.median(ttft_speedups)
        cells.append(
            {
                "reused_block_count": block_count,
                "repetitions": len(rows),
                "all_outputs_exact": all_exact,
                "prompt_token_count": rows[0]["prompt_token_count"],
                "reused_prompt_tokens": rows[0]["reused_prompt_tokens"],
                "mean_reuse_fraction": statistics.fmean(
                    float(row["reuse_fraction"]) for row in rows
                ),
                "mean_speedup": statistics.fmean(speedups),
                "median_speedup": median_speedup,
                "minimum_speedup": min(speedups),
                "maximum_speedup": max(speedups),
                "mean_ttft_speedup": statistics.fmean(ttft_speedups),
                "median_ttft_speedup": median_ttft_speedup,
                "minimum_ttft_speedup": min(ttft_speedups),
                "maximum_ttft_speedup": max(ttft_speedups),
                "wall_break_even_met": all_exact and median_speedup > 1.0,
                "ttft_break_even_met": all_exact and median_ttft_speedup > 1.0,
                "break_even_met": (
                    all_exact
                    and median_speedup > 1.0
                    and median_ttft_speedup > 1.0
                ),
            }
        )

    first_break_even = next(
        (
            cell["reused_block_count"]
            for cell in cells
            if cell["break_even_met"]
        ),
        None,
    )
    return {
        "schema_version": 1,
        "experiment": "hybrid_gdn_reuse_breakeven",
        "trial_count": len(trials),
        "all_outputs_exact": all(
            trial.get("exact_output_match") is True for trial in trials
        ),
        "first_break_even_reused_block_count": first_break_even,
        "cells": cells,
    }
