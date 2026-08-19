"""Measure when active GDN block reuse becomes faster than full computation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.hybrid_gdn_breakeven import (
    HybridGDNBreakEvenPromptPair,
    build_hybrid_gdn_breakeven_conditions,
    calibrate_hybrid_gdn_breakeven_prompt,
    summarize_hybrid_gdn_breakeven,
)
from benchmarks.run_hybrid_apc_baseline import _post_json, _run_recorded_request
from observability.request_recorder import RequestRecorder, validate_ledger


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


# Run all calibrated conditions sequentially against one already-warm vLLM server.
def run_hybrid_gdn_breakeven(
    *,
    base_url: str,
    model: str,
    run_id: str,
    output: Path,
    request_log_dir: Path,
    reused_block_counts: tuple[int, ...],
    repetitions: int,
    block_size: int,
    cache_capacity: int,
    max_completion_tokens: int = 64,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Execute full-reference and active-reuse requests for every condition."""
    conditions = build_hybrid_gdn_breakeven_conditions(
        reused_block_counts,
        repetitions,
    )
    if cache_capacity < max(reused_block_counts):
        raise ValueError("cache_capacity must cover the largest reuse condition")
    recorder = RequestRecorder(
        run_id=run_id,
        model=model,
        backend="vllm-hybrid-gdn-breakeven",
        log_dir=request_log_dir,
        invocation_metadata={"experiment": "hybrid_gdn_reuse_breakeven"},
    )

    # Calibration happens once per reuse size and uses no generation requests.
    pairs = {
        count: calibrate_hybrid_gdn_breakeven_prompt(
            reused_block_count=count,
            block_size=block_size,
            tokenize_prompt=lambda prompt: _tokenize_prompt(
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
            ),
        )
        for count in reused_block_counts
    }
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    trials = []
    for condition in conditions:
        pair = pairs[condition.reused_block_count]
        scenario = (
            f"reuse-{condition.reused_block_count:02d}"
            f"-rep-{condition.repetition:02d}"
        )
        reference_salt = hashlib.sha256(
            f"{recorder.invocation_id}:{scenario}:reference".encode()
        ).hexdigest()
        pair_salt = hashlib.sha256(
            f"{recorder.invocation_id}:{scenario}:pair".encode()
        ).hexdigest()
        source_request_id = f"{run_id}:{scenario}:source"

        # Separate salts make the reference uncached while source/target share state.
        reference = _run_recorded_request(
            recorder=recorder,
            chat_url=chat_url,
            metrics_url=metrics_url,
            model=model,
            scenario=scenario,
            role="reference",
            prompt=pair.target_prompt,
            cache_salt=reference_salt,
            cacheselect_request_id=f"{run_id}:{scenario}:reference",
            cacheselect_source_request_id=None,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
        )
        source = _run_recorded_request(
            recorder=recorder,
            chat_url=chat_url,
            metrics_url=metrics_url,
            model=model,
            scenario=scenario,
            role="source",
            prompt=pair.source_prompt,
            cache_salt=pair_salt,
            cacheselect_request_id=source_request_id,
            cacheselect_source_request_id=None,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
        )
        target = _run_recorded_request(
            recorder=recorder,
            chat_url=chat_url,
            metrics_url=metrics_url,
            model=model,
            scenario=scenario,
            role="target",
            prompt=pair.target_prompt,
            cache_salt=pair_salt,
            cacheselect_request_id=f"{run_id}:{scenario}:target",
            cacheselect_source_request_id=source_request_id,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
        )
        trial = _validate_active_trial(
            pair=pair,
            block_size=block_size,
            reference=reference,
            source=source,
            target=target,
        )
        trial.update(
            {
                "condition_id": scenario,
                "repetition": condition.repetition,
                "filler_repetitions": pair.filler_repetitions,
            }
        )
        trials.append(trial)

    ledger = validate_ledger(recorder.path)
    if not ledger.is_complete or ledger.failed:
        raise RuntimeError("request ledger is incomplete or contains failures")
    result = summarize_hybrid_gdn_breakeven(trials)
    result.update(
        {
            "run_id": run_id,
            "model": model,
            "block_size": block_size,
            "cache_capacity": cache_capacity,
            "reused_block_counts": list(reused_block_counts),
            "repetitions": repetitions,
            "request_ledger": str(recorder.path),
            "trials": trials,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
