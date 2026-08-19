"""Build and summarize controlled GDN reuse break-even experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


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
