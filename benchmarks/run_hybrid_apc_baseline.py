"""Measure stock vLLM prefix reuse for controlled hybrid-model edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from observability.request_recorder import RequestRecorder, validate_ledger


PREFIX_QUERY_METRIC = "vllm:prefix_cache_queries_total"
PREFIX_HIT_METRIC = "vllm:prefix_cache_hits_total"


@dataclass(frozen=True)
class HybridAPCScenario:
    """One source prompt and the related target prompt tested against it."""

    name: str
    source_prompt: str
    target_prompt: str


# Measure whether one edited prompt preserves later absolute token positions.
def assess_hybrid_token_alignment(
    source_tokens: list[int],
    target_tokens: list[int],
) -> dict[str, Any]:
    shared_prefix = 0
    for source_token, target_token in zip(source_tokens, target_tokens):
        if source_token != target_token:
            break
        shared_prefix += 1

    shared_suffix = 0
    for source_token, target_token in zip(
        reversed(source_tokens), reversed(target_tokens)
    ):
        if source_token != target_token:
            break
        shared_suffix += 1
    same_length = len(source_tokens) == len(target_tokens)
    return {
        "passed": same_length
        and shared_prefix < len(source_tokens)
        and shared_suffix > 0,
        "source_token_count": len(source_tokens),
        "target_token_count": len(target_tokens),
        "same_token_count": same_length,
        "shared_prefix_tokens": shared_prefix,
        "shared_suffix_tokens": shared_suffix,
    }


# Check that block checkpoints expose progressively longer edited prefixes.
def assess_hybrid_checkpoint_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = {row["scenario"]: row for row in rows if row.get("role") == "target"}
    expected = {"exact", "append_only", "early_edit", "middle_edit"}
    missing = expected.difference(targets)
    if missing:
        raise ValueError(f"missing hybrid target observations: {sorted(missing)}")

    cached = {
        scenario: int(targets[scenario]["cached_tokens"])
        for scenario in sorted(expected)
    }
    checks = {
        "exact_reused_prefix": cached["exact"] > 0,
        "append_reused_prefix": cached["append_only"] > 0,
        "early_edit_reused_prefix": cached["early_edit"] > 0,
        "middle_edit_reused_prefix": cached["middle_edit"] > 0,
        "later_edit_reused_more": cached["middle_edit"] > cached["early_edit"],
    }
    return {
        "passed": all(checks.values()),
        "cached_tokens": cached,
        "checks": checks,
    }


# Verify edited requests expose their hybrid plan and worker preflight result.
def assess_gdn_delta_preflight_observability(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    edited_targets = {
        row["scenario"]: row
        for row in rows
        if row.get("role") == "target"
        and row.get("scenario") in {"early_edit", "middle_edit"}
    }
    observed = {
        scenario: isinstance(row.get("gdn_delta_reuse"), dict)
        and "plan" in row["gdn_delta_reuse"]
        and "preflight_reason" in row["gdn_delta_reuse"]
        for scenario, row in edited_targets.items()
    }
    return {
        "passed": set(observed) == {"early_edit", "middle_edit"}
        and all(observed.values()),
        "observed": observed,
    }


# Verify edited requests completed every planned layer/block shadow comparison.
def assess_gdn_delta_shadow_execution(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    edited_targets = {
        row["scenario"]: row
        for row in rows
        if row.get("role") == "target"
        and row.get("scenario") in {"early_edit", "middle_edit"}
    }
    details = {}
    for scenario, row in edited_targets.items():
        metrics = row.get("gdn_delta_reuse")
        if not isinstance(metrics, dict):
            details[scenario] = {"passed": False, "reason": "metrics_missing"}
            continue
        expected = metrics.get("shadow_expected_count")
        compared = metrics.get("shadow_compared_count")
        output_error = metrics.get("shadow_max_output_relative_l2")
        state_error = metrics.get("shadow_max_final_state_relative_l2")
        passed = (
            metrics.get("preflight_eligible") is True
            and isinstance(expected, int)
            and expected > 0
            and compared == expected
            and metrics.get("shadow_complete") is True
            and isinstance(output_error, (int, float))
            and math.isfinite(output_error)
            and isinstance(state_error, (int, float))
            and math.isfinite(state_error)
        )
        details[scenario] = {
            "passed": passed,
            "expected_comparisons": expected,
            "completed_comparisons": compared,
            "max_output_relative_l2": output_error,
            "max_final_state_relative_l2": state_error,
        }
    return {
        "passed": set(details) == {"early_edit", "middle_edit"}
        and all(detail["passed"] for detail in details.values()),
        "details": details,
    }


# Verify both edited requests actively reused GDN work across every model layer.
def assess_gdn_delta_active_execution(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require complete layer execution plus nonzero reuse and recomputation."""
    edited_targets = {
        row["scenario"]: row
        for row in rows
        if row.get("role") == "target"
        and row.get("scenario") in {"early_edit", "middle_edit"}
    }
    details = {}
    for scenario, row in edited_targets.items():
        metrics = row.get("gdn_delta_reuse")
        if not isinstance(metrics, dict):
            details[scenario] = {"passed": False, "reason": "metrics_missing"}
            continue
        expected_layers = metrics.get("resolved_layer_count")
        executed_layers = metrics.get("active_executed_layer_count")
        reused_tokens = metrics.get("active_reused_layer_tokens")
        recomputed_tokens = metrics.get("active_recomputed_layer_tokens")
        passed = (
            metrics.get("preflight_eligible") is True
            and isinstance(expected_layers, int)
            and expected_layers > 0
            and executed_layers == expected_layers
            and metrics.get("active_complete") is True
            and isinstance(reused_tokens, int)
            and reused_tokens > 0
            and isinstance(recomputed_tokens, int)
            and recomputed_tokens > 0
        )
        details[scenario] = {
            "passed": passed,
            "expected_layers": expected_layers,
            "executed_layers": executed_layers,
            "reused_layer_tokens": reused_tokens,
            "recomputed_layer_tokens": recomputed_tokens,
        }
    return {
        "passed": set(details) == {"early_edit", "middle_edit"}
        and all(detail["passed"] for detail in details.values()),
        "details": details,
    }


# Build prompts with edits at stable early and middle relative positions.
def build_hybrid_apc_scenarios(
    record_count: int = 160,
) -> tuple[HybridAPCScenario, ...]:
    """Return controlled prompt pairs at one configurable context length."""
    if record_count < 20:
        raise ValueError("record_count must be at least 20")
    early_edit_index = record_count // 20
    middle_edit_index = record_count // 2
    scenarios: list[HybridAPCScenario] = []
    for name in ("exact", "append_only", "early_edit", "middle_edit"):
        sentences = [
            f"Hybrid APC {name} record {index:03d} says marker A and cached state "
            "follows prompt order."
            for index in range(record_count)
        ]
        source = " ".join(sentences) + "\nSummarize this principle. /no_think"
        target_sentences = list(sentences)
        if name == "append_only":
            target_sentences.append(
                "The appended record keeps every earlier prompt token unchanged."
            )
        elif name == "early_edit":
            target_sentences[early_edit_index] = (
                f"Hybrid APC early_edit record {early_edit_index:03d} says marker B "
                "and cached state "
                "follows prompt order."
            )
        elif name == "middle_edit":
            target_sentences[middle_edit_index] = (
                f"Hybrid APC middle_edit record {middle_edit_index:03d} says marker B "
                "and cached state "
                "follows prompt order."
            )
        target = " ".join(target_sentences) + "\nSummarize this principle. /no_think"
        scenarios.append(HybridAPCScenario(name, source, target))
    return tuple(scenarios)


# Read one Prometheus counter, summing values from all matching label sets.
def parse_prometheus_counter(metrics_text: str, metric_name: str) -> float:
    pattern = re.compile(
        rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$"
    )
    values = [
        float(match.group(1))
        for line in metrics_text.splitlines()
        if (match := pattern.match(line)) is not None
    ]
    if not values:
        raise RuntimeError(f"Prometheus response omitted {metric_name}")
    return sum(values)


# Fetch the server's current prefix-query and prefix-hit counters.
def _read_prefix_counters(metrics_url: str, timeout_seconds: float) -> tuple[int, int]:
    with urllib.request.urlopen(metrics_url, timeout=timeout_seconds) as response:
        metrics_text = response.read().decode()
    return (
        round(parse_prometheus_counter(metrics_text, PREFIX_QUERY_METRIC)),
        round(parse_prometheus_counter(metrics_text, PREFIX_HIT_METRIC)),
    )


# Submit one JSON request to the OpenAI-compatible chat endpoint.
def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


# Tokenize edited scenarios with the running server's exact chat template.
def inspect_hybrid_token_alignment(
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
    record_count: int = 160,
) -> dict[str, Any]:
    details = {}
    tokenize_url = f"{base_url.rstrip('/')}/tokenize"
    for scenario in build_hybrid_apc_scenarios(record_count):
        if scenario.name not in {"early_edit", "middle_edit"}:
            continue
        token_lists = []
        for prompt in (scenario.source_prompt, scenario.target_prompt):
            response = _post_json(
                tokenize_url,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout_seconds,
            )
            tokens = response.get("tokens")
            if not isinstance(tokens, list) or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in tokens
            ):
                raise RuntimeError("vLLM tokenize response omitted integer tokens")
            token_lists.append(tokens)
        details[scenario.name] = assess_hybrid_token_alignment(*token_lists)
    return {
        "passed": set(details) == {"early_edit", "middle_edit"}
        and all(detail["passed"] for detail in details.values()),
        "details": details,
    }


# Execute and record one request together with its exact counter deltas.
def _run_recorded_request(
    *,
    recorder: RequestRecorder,
    chat_url: str,
    metrics_url: str,
    model: str,
    scenario: str,
    role: str,
    prompt: str,
    cache_salt: str,
    cacheselect_request_id: str,
    cacheselect_source_request_id: str | None,
    max_completion_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "request_id": f"{recorder.run_id}-{scenario}-{role}",
        "temperature": 0.0,
        "seed": 0,
        "max_completion_tokens": max_completion_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
        "cache_salt": cache_salt,
        "vllm_xargs": {
            "cacheselect_request_id": cacheselect_request_id,
            **(
                {"cacheselect_source_request_id": cacheselect_source_request_id}
                if cacheselect_source_request_id is not None
                else {}
            ),
        },
    }
    pending = recorder.start(
        model_input={"api": "OpenAI Chat Completions", "payload": payload},
        sampling={"temperature": 0.0, "seed": 0},
        metadata={"scenario": scenario, "role": role},
    )
    before_queries, before_hits = _read_prefix_counters(metrics_url, timeout_seconds)
    started = time.perf_counter()
    try:
        response = _post_json(chat_url, payload, timeout_seconds)
        elapsed = time.perf_counter() - started
        after_queries, after_hits = _read_prefix_counters(metrics_url, timeout_seconds)
    except BaseException as error:
        pending.fail(error, metadata={"scenario": scenario, "role": role})
        raise

    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    usage = response.get("usage") or {}
    response_metrics = response.get("metrics") or {}
    row = {
        "scenario": scenario,
        "role": role,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "queried_tokens": after_queries - before_queries,
        "cached_tokens": after_hits - before_hits,
        "client_wall_seconds": elapsed,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "output_text": message.get("content"),
        "gdn_delta_reuse": response_metrics.get("gdn_delta_reuse"),
    }
    pending.complete(output={"raw_response": response}, metrics=row)
    return row


# Run every isolated source-to-target transition and save its audit summary.
def run_hybrid_apc_baseline(
    *,
    base_url: str,
    model: str,
    run_id: str,
    request_log_dir: Path,
    output: Path,
    max_completion_tokens: int = 16,
    timeout_seconds: float = 120.0,
    validate_against_reference: bool = False,
    require_token_aligned_edits: bool = False,
    record_count: int = 160,
) -> dict[str, Any]:
    recorder = RequestRecorder(
        run_id=run_id,
        model=model,
        backend="vllm-hybrid-apc",
        log_dir=request_log_dir,
        invocation_metadata={"experiment": "hybrid_apc_baseline"},
    )
    token_alignment = inspect_hybrid_token_alignment(
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        record_count=record_count,
    )
    if require_token_aligned_edits and not token_alignment["passed"]:
        # Abort before issuing any generation requests for a misaligned workload.
        raise RuntimeError(
            f"hybrid edits are not token aligned: {token_alignment['details']}"
        )
    rows: list[dict[str, Any]] = []
    for scenario in build_hybrid_apc_scenarios(record_count):
        if validate_against_reference:
            # A separate salt guarantees that this target performs a full prefill.
            reference_salt = hashlib.sha256(
                f"{recorder.invocation_id}:{scenario.name}:reference".encode()
            ).hexdigest()
            rows.append(
                _run_recorded_request(
                    recorder=recorder,
                    chat_url=f"{base_url.rstrip('/')}/v1/chat/completions",
                    metrics_url=f"{base_url.rstrip('/')}/metrics",
                    model=model,
                    scenario=scenario.name,
                    role="reference",
                    prompt=scenario.target_prompt,
                    cache_salt=reference_salt,
                    cacheselect_request_id=(f"{run_id}:{scenario.name}:reference"),
                    cacheselect_source_request_id=None,
                    max_completion_tokens=max_completion_tokens,
                    timeout_seconds=timeout_seconds,
                )
            )
        # A per-invocation salt prevents previous smoke requests contaminating a pair.
        cache_salt = hashlib.sha256(
            f"{recorder.invocation_id}:{scenario.name}".encode()
        ).hexdigest()
        source_request_id = f"{run_id}:{scenario.name}:source"
        for role, prompt in (
            ("source", scenario.source_prompt),
            ("target", scenario.target_prompt),
        ):
            rows.append(
                _run_recorded_request(
                    recorder=recorder,
                    chat_url=f"{base_url.rstrip('/')}/v1/chat/completions",
                    metrics_url=f"{base_url.rstrip('/')}/metrics",
                    model=model,
                    scenario=scenario.name,
                    role=role,
                    prompt=prompt,
                    cache_salt=cache_salt,
                    cacheselect_request_id=f"{run_id}:{scenario.name}:{role}",
                    cacheselect_source_request_id=(
                        source_request_id if role == "target" else None
                    ),
                    max_completion_tokens=max_completion_tokens,
                    timeout_seconds=timeout_seconds,
                )
            )

    ledger = validate_ledger(recorder.path)
    checkpoint_reuse = assess_hybrid_checkpoint_reuse(rows)
    gdn_delta_preflight = assess_gdn_delta_preflight_observability(rows)
    gdn_delta_shadow = assess_gdn_delta_shadow_execution(rows)
    gdn_delta_active = assess_gdn_delta_active_execution(rows)
    reference_checks: list[dict[str, Any]] = []
    if validate_against_reference:
        for scenario in build_hybrid_apc_scenarios(record_count):
            reference = next(
                row
                for row in rows
                if row["scenario"] == scenario.name and row["role"] == "reference"
            )
            target = next(
                row
                for row in rows
                if row["scenario"] == scenario.name and row["role"] == "target"
            )
            reference_checks.append(
                {
                    "scenario": scenario.name,
                    "exact_output_match": (
                        reference["output_text"] == target["output_text"]
                    ),
                    "reference_finish_reason": reference["finish_reason"],
                    "target_finish_reason": target["finish_reason"],
                }
            )

    result = {
        "schema_version": 1,
        "experiment": "hybrid_apc_baseline",
        "run_id": run_id,
        "model": model,
        "record_count": record_count,
        "request_ledger": str(recorder.path),
        "ledger_complete": ledger.is_complete,
        "scenarios": [
            asdict(item) for item in build_hybrid_apc_scenarios(record_count)
        ],
        "token_alignment": token_alignment,
        "observations": rows,
        "checkpoint_reuse": checkpoint_reuse,
        "gdn_delta_preflight": gdn_delta_preflight,
        "gdn_delta_shadow": gdn_delta_shadow,
        "gdn_delta_active": gdn_delta_active,
        "reference_checks": reference_checks,
        "reference_validation_passed": bool(reference_checks)
        and all(
            check["exact_output_match"]
            and check["reference_finish_reason"] == "stop"
            and check["target_finish_reason"] == "stop"
            for check in reference_checks
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


# Parse command-line arguments for the standalone live-server runner.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-completion-tokens", type=int, default=16)
    parser.add_argument("--record-count", type=int, default=160)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--validate-against-reference", action="store_true")
    parser.add_argument("--require-edited-prefix-reuse", action="store_true")
    parser.add_argument("--require-token-aligned-edits", action="store_true")
    parser.add_argument("--require-gdn-preflight-observability", action="store_true")
    parser.add_argument("--require-gdn-shadow-execution", action="store_true")
    parser.add_argument("--require-gdn-active-execution", action="store_true")
    return parser.parse_args()


# Run the hybrid APC baseline from the command line.
def main() -> None:
    args = _parse_args()
    result = run_hybrid_apc_baseline(
        base_url=args.base_url,
        model=args.model,
        run_id=args.run_id,
        request_log_dir=args.request_log_dir,
        output=args.output,
        max_completion_tokens=args.max_completion_tokens,
        timeout_seconds=args.timeout_seconds,
        validate_against_reference=args.validate_against_reference,
        require_token_aligned_edits=args.require_token_aligned_edits,
        record_count=args.record_count,
    )
    targets = [row for row in result["observations"] if row["role"] == "target"]
    for row in targets:
        print(
            f"{row['scenario']}: queried={row['queried_tokens']} "
            f"cached={row['cached_tokens']} seconds={row['client_wall_seconds']:.3f}"
        )
    for check in result["reference_checks"]:
        print(f"{check['scenario']}: exact_output_match={check['exact_output_match']}")
    print(f"Saved {result['request_ledger']}")
    if args.require_edited_prefix_reuse and not result["checkpoint_reuse"]["passed"]:
        raise SystemExit("Hybrid checkpoint edited-prefix reuse validation failed")
    if (
        args.require_gdn_preflight_observability
        and not result["gdn_delta_preflight"]["passed"]
    ):
        raise SystemExit("Hybrid GDN delta preflight observability validation failed")
    if args.require_gdn_shadow_execution and not result["gdn_delta_shadow"]["passed"]:
        raise SystemExit("Hybrid GDN delta shadow execution validation failed")
    if args.require_gdn_active_execution and not result["gdn_delta_active"]["passed"]:
        raise SystemExit("Hybrid GDN delta active execution validation failed")
    if args.validate_against_reference and not result["reference_validation_passed"]:
        raise SystemExit("Hybrid checkpoint output validation failed")


if __name__ == "__main__":
    main()
