"""Measure when active GDN block reuse becomes faster than full computation."""

from __future__ import annotations

from typing import Any

from benchmarks.hybrid_gdn_breakeven import HybridGDNBreakEvenPromptPair
from benchmarks.run_hybrid_apc_baseline import _post_json


# Tokenize one prompt with the exact template used by the running model server.
def _tokenize_prompt(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> tuple[int, ...]:
    """Return validated token IDs from the live vLLM tokenize endpoint."""
    response = _post_json(
        f"{base_url.rstrip('/')}/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout_seconds,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        raise RuntimeError("vLLM tokenize response omitted integer tokens")
    return tuple(tokens)


# Validate that the intervention really reused every requested block and stayed exact.
def _validate_active_trial(
    *,
    pair: HybridGDNBreakEvenPromptPair,
    block_size: int,
    reference: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    """Return one timing row only after all causal execution checks pass."""
    for role, row in (("reference", reference), ("source", source), ("target", target)):
        if row.get("finish_reason") != "stop":
            raise RuntimeError(f"{role} request did not finish normally")
        if row.get("cached_tokens") != 0:
            raise RuntimeError(f"{role} request unexpectedly used native APC")
        if row.get("prompt_tokens") != pair.prompt_token_count:
            raise RuntimeError(f"{role} prompt token count changed after calibration")

    exact_output_match = reference.get("output_text") == target.get("output_text")
    if not exact_output_match or not reference.get("output_text"):
        raise RuntimeError("active reuse changed the full-computation output")
    metrics = target.get("gdn_delta_reuse")
    if not isinstance(metrics, dict):
        raise RuntimeError("active target omitted GDN reuse metrics")
    plan = metrics.get("plan")
    if not isinstance(plan, dict):
        raise RuntimeError("active target omitted its GDN reuse plan")

    expected_candidates = tuple(range(1, pair.reused_block_count + 1))
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("active target plan omitted candidates")
    target_indices = tuple(candidate.get("target_block_index") for candidate in candidates)
    if target_indices != expected_candidates:
        raise RuntimeError("active target reused unexpected logical blocks")
    if plan.get("block_size") != block_size:
        raise RuntimeError("active target used the wrong GDN block size")
    if metrics.get("candidate_block_count") != pair.reused_block_count:
        raise RuntimeError("active target exposed the wrong candidate count")

    expected_layers = metrics.get("resolved_layer_count")
    executed_layers = metrics.get("active_executed_layer_count")
    reused_layer_tokens = metrics.get("active_reused_layer_tokens")
    recomputed_layer_tokens = metrics.get("active_recomputed_layer_tokens")
    expected_reused_layer_tokens = (
        pair.reused_block_count * block_size * executed_layers
        if isinstance(executed_layers, int)
        else None
    )
    if not (
        metrics.get("preflight_eligible") is True
        and metrics.get("active_complete") is True
        and isinstance(expected_layers, int)
        and expected_layers > 0
        and executed_layers == expected_layers
        and reused_layer_tokens == expected_reused_layer_tokens
        and isinstance(recomputed_layer_tokens, int)
        and recomputed_layer_tokens > 0
    ):
        raise RuntimeError("active target did not complete the requested GDN reuse")

    reference_wall = reference.get("client_wall_seconds")
    active_wall = target.get("client_wall_seconds")
    reference_ttft = reference.get("time_to_first_token_ms")
    active_ttft = target.get("time_to_first_token_ms")
    for name, value in (
        ("reference wall time", reference_wall),
        ("active wall time", active_wall),
        ("reference TTFT", reference_ttft),
        ("active TTFT", active_ttft),
    ):
        if not isinstance(value, (int, float)) or value <= 0:
            raise RuntimeError(f"{name} is missing or invalid")

    reused_prompt_tokens = pair.reused_block_count * block_size
    total_layer_tokens = reused_layer_tokens + recomputed_layer_tokens
    return {
        "reused_block_count": pair.reused_block_count,
        "prompt_token_count": pair.prompt_token_count,
        "reused_prompt_tokens": reused_prompt_tokens,
        "reuse_fraction": reused_prompt_tokens / pair.prompt_token_count,
        "reference_wall_seconds": reference_wall,
        "active_wall_seconds": active_wall,
        "speedup": reference_wall / active_wall,
        "reference_ttft_ms": reference_ttft,
        "active_ttft_ms": active_ttft,
        "ttft_speedup": reference_ttft / active_ttft,
        "exact_output_match": exact_output_match,
        "executed_layer_count": executed_layers,
        "active_reused_layer_tokens": reused_layer_tokens,
        "active_recomputed_layer_tokens": recomputed_layer_tokens,
        "active_layer_reuse_fraction": reused_layer_tokens / total_layer_tokens,
    }
