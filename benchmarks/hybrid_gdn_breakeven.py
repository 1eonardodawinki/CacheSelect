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
