"""Run a CacheSelect trace against an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cacheselect.features import (
    summarize_api_request,
    token_transition_features,
)
from benchmarks.evaluation import score_response
from benchmarks.schema import RequestSpec, load_trace
from observability.request_recorder import RequestRecorder, validate_ledger


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"vLLM returned HTTP {error.code} for {url}: {body}"
        ) from error


def _output_text(response: dict[str, Any]) -> str | None:
    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    return message.get("content")


def _cached_tokens(response: dict[str, Any]) -> int | None:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens")


def _validate_observability(response: dict[str, Any]) -> None:
    required = {
        "prompt_text": response.get("prompt_text"),
        "prompt_token_ids": response.get("prompt_token_ids"),
        "metrics": response.get("metrics"),
        "usage.prompt_tokens_details": (
            (response.get("usage") or {}).get("prompt_tokens_details")
        ),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "vLLM omitted required observation fields "
            f"{missing}. Use this repository's vLLM checkout and start it with "
            "--enable-prompt-tokens-details and --enable-per-request-metrics."
        )


def _observe_request(
    request: RequestSpec,
    *,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder | None = None,
    policy_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = request.api_payload(model, max_completion_tokens)
    pending = None
    if recorder is not None:
        pending = recorder.start(
            model_input={
                "api": "OpenAI Chat Completions",
                "endpoint": url,
                "payload": payload,
            },
            sampling={
                "temperature": payload.get("temperature"),
                "max_completion_tokens": payload.get(
                    "max_completion_tokens"
                ),
                "stream": payload.get("stream"),
            },
            metadata={
                "benchmark_request_id": request.request_id,
                "workload": request.workload,
                "sequence_index": request.sequence_index,
                **(policy_metadata or {}),
            },
            evaluation={
                # These controlled labels describe the benchmark request but
                # are not sent to vLLM or exposed to a future planner.
                "prompt_segments": [
                    asdict(segment) for segment in request.segments
                ],
                "ground_truth": asdict(request.ground_truth),
            },
        )

    started = time.perf_counter()
    try:
        response = _post_json(
            url,
            payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        wall_seconds = time.perf_counter() - started
        _validate_observability(response)
    except BaseException as error:
        if pending is not None:
            pending.fail(
                error,
                metadata={
                    "benchmark_request_id": request.request_id,
                    "client_wall_seconds": time.perf_counter() - started,
                },
            )
        raise

    output_text = _output_text(response)
    quality = score_response(output_text, request.ground_truth)
    observation = {
        "request_id": request.request_id,
        "sequence_index": request.sequence_index,
        "request_payload": payload,
        "request_structure": summarize_api_request(payload),
        "rendered_prompt": response.get("prompt_text"),
        "prompt_token_ids": response.get("prompt_token_ids"),
        "prompt_token_count": (response.get("usage") or {}).get("prompt_tokens"),
        "cached_tokens": _cached_tokens(response),
        "client_wall_seconds": wall_seconds,
        "server_metrics": response.get("metrics"),
        "usage": response.get("usage"),
        "output_text": output_text,
        "quality": quality,
        "response_field_names": sorted(response),
        "raw_response": response,
    }
    if pending is not None:
        pending.complete(
            output={
                "text": observation["output_text"],
                "raw_response": response,
            },
            metrics={
                "client_wall_seconds": wall_seconds,
                "cached_tokens": observation["cached_tokens"],
                "prompt_token_count": observation["prompt_token_count"],
                "server_metrics": observation["server_metrics"],
                "usage": observation["usage"],
                "quality": quality,
            },
            metadata={"benchmark_request_id": request.request_id},
        )
    return observation


def _transition_results(
    trace,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_request = {
        observation["request_id"]: observation for observation in observations
    }
    results = []
    for transition in trace.transitions:
        previous = by_request[transition.previous_request_id]
        current = by_request[transition.current_request_id]
        previous_tokens = previous["prompt_token_ids"]
        current_tokens = current["prompt_token_ids"]
        features = None
        if previous_tokens is not None and current_tokens is not None:
            features = token_transition_features(previous_tokens, current_tokens)
        results.append(
            {
                **asdict(transition),
                "measured_features": features,
                "current_cached_tokens": current["cached_tokens"],
                "current_prompt_token_count": current["prompt_token_count"],
                "current_ttft_ms": (
                    (current["server_metrics"] or {}).get(
                        "time_to_first_token_ms"
                    )
                ),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--apc-label", choices=["on", "off"], required=True)
    parser.add_argument("--max-completion-tokens", type=int, default=48)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Request-ledger run ID (defaults to a unique trace/APC ID).",
    )
    parser.add_argument(
        "--request-log-dir",
        type=Path,
        default=None,
        help="Directory for the append-only full-input/output JSONL ledger.",
    )
    args = parser.parse_args()

    trace = load_trace(args.trace)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    run_id = args.run_id or (
        f"{trace.trace_id}-apc-{args.apc_label}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    recorder = RequestRecorder(
        run_id=run_id,
        model=args.model,
        backend="vllm-openai-compatible-server",
        log_dir=args.request_log_dir,
        invocation_metadata={
            "trace_id": trace.trace_id,
            "trace_path": str(args.trace),
            "result_path": str(args.output),
            "endpoint": endpoint,
            "apc": args.apc_label,
        },
    )
    observations = []
    for request in trace.requests:
        print(f"Running {request.request_id}...")
        observations.append(
            _observe_request(
                request,
                url=endpoint,
                model=args.model,
                max_completion_tokens=args.max_completion_tokens,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
                recorder=recorder,
                policy_metadata={
                    "policy": (
                        "VLLM_NATIVE_APC"
                        if args.apc_label == "on"
                        else "FULL_RECOMPUTE"
                    ),
                    "apc": args.apc_label,
                },
            )
        )

    if args.apc_label == "off":
        unexpected_hits = [
            observation["request_id"]
            for observation in observations
            if observation["cached_tokens"] not in (None, 0)
        ]
        if unexpected_hits:
            raise RuntimeError(
                "The run was labelled APC off, but vLLM reported cache hits for "
                f"{unexpected_hits}. Restart the server with "
                "--no-enable-prefix-caching."
            )

    ledger_summary = validate_ledger(recorder.path)
    if not ledger_summary.is_complete:
        raise RuntimeError(
            "Request ledger is incomplete: "
            + ", ".join(ledger_summary.incomplete_request_ids)
        )

    result = {
        "trace_id": trace.trace_id,
        "workload": trace.workload,
        "model": args.model,
        "base_url": args.base_url,
        "apc": args.apc_label,
        "run_id": run_id,
        "request_ledger": str(recorder.path),
        "request_ledger_summary": {
            "started": ledger_summary.started,
            "completed": ledger_summary.completed,
            "failed": ledger_summary.failed,
        },
        "observations": observations,
        "transitions": _transition_results(trace, observations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")
    print(f"Saved request ledger {recorder.path}")


if __name__ == "__main__":
    main()
