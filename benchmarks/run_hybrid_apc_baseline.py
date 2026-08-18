"""Measure stock vLLM prefix reuse for controlled hybrid-model edits."""

from __future__ import annotations

import argparse
import hashlib
import json
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


# Build long prompts whose edits occur before, within, or after cache pages.
def build_hybrid_apc_scenarios() -> tuple[HybridAPCScenario, ...]:
    scenarios: list[HybridAPCScenario] = []
    for name in ("exact", "append_only", "early_edit", "middle_edit"):
        sentences = [
            f"Hybrid APC {name} record {index:03d} says cached state follows prompt order."
            for index in range(160)
        ]
        source = " ".join(sentences) + "\nSummarize this principle. /no_think"
        target_sentences = list(sentences)
        if name == "append_only":
            target_sentences.append(
                "The appended record keeps every earlier prompt token unchanged."
            )
        elif name == "early_edit":
            target_sentences[8] = (
                "Hybrid APC early_edit record 008 says edited state changes promptly."
            )
        elif name == "middle_edit":
            target_sentences[80] = (
                "Hybrid APC middle_edit record 080 says edited state changes midway."
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
        "stream": False,
        "cache_salt": cache_salt,
    }
    pending = recorder.start(
        model_input={"api": "OpenAI Chat Completions", "payload": payload},
        sampling={"temperature": 0.0, "seed": 0},
        metadata={"scenario": scenario, "role": role},
    )
    before_queries, before_hits = _read_prefix_counters(
        metrics_url, timeout_seconds
    )
    started = time.perf_counter()
    try:
        response = _post_json(chat_url, payload, timeout_seconds)
        elapsed = time.perf_counter() - started
        after_queries, after_hits = _read_prefix_counters(
            metrics_url, timeout_seconds
        )
    except BaseException as error:
        pending.fail(error, metadata={"scenario": scenario, "role": role})
        raise

    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    usage = response.get("usage") or {}
    row = {
        "scenario": scenario,
        "role": role,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "queried_tokens": after_queries - before_queries,
        "cached_tokens": after_hits - before_hits,
        "client_wall_seconds": elapsed,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "output_text": message.get("content"),
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
) -> dict[str, Any]:
    recorder = RequestRecorder(
        run_id=run_id,
        model=model,
        backend="vllm-hybrid-apc",
        log_dir=request_log_dir,
        invocation_metadata={"experiment": "hybrid_apc_baseline"},
    )
    rows: list[dict[str, Any]] = []
    for scenario in build_hybrid_apc_scenarios():
        # A per-invocation salt prevents previous smoke requests contaminating a pair.
        cache_salt = hashlib.sha256(
            f"{recorder.invocation_id}:{scenario.name}".encode()
        ).hexdigest()
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
                    max_completion_tokens=max_completion_tokens,
                    timeout_seconds=timeout_seconds,
                )
            )

    ledger = validate_ledger(recorder.path)
    result = {
        "schema_version": 1,
        "experiment": "hybrid_apc_baseline",
        "run_id": run_id,
        "model": model,
        "request_ledger": str(recorder.path),
        "ledger_complete": ledger.is_complete,
        "scenarios": [asdict(item) for item in build_hybrid_apc_scenarios()],
        "observations": rows,
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
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
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
    )
    targets = [row for row in result["observations"] if row["role"] == "target"]
    for row in targets:
        print(
            f"{row['scenario']}: queried={row['queried_tokens']} "
            f"cached={row['cached_tokens']} seconds={row['client_wall_seconds']:.3f}"
        )
    print(f"Saved {result['request_ledger']}")


if __name__ == "__main__":
    main()
