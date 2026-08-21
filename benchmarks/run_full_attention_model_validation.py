"""Validate a deterministic BF16 full-attention model before reuse experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.run_hybrid_apc_baseline import _run_recorded_request
from observability.request_recorder import RequestRecorder, validate_ledger


EXPECTED_OUTPUT = "Qwen3-14B BF16 validation passed."
VALIDATION_PROMPT = (
    "Reply with exactly the following text and nothing else: " + EXPECTED_OUTPUT
)


# Fail closed if the live server is not the intended conventional Qwen3 model.
def assess_full_attention_configuration(
    server_info: dict[str, Any],
    hf_config: dict[str, Any],
) -> dict[str, Any]:
    config = server_info.get("vllm_config") or {}
    model = config.get("model_config") or {}
    cache = config.get("cache_config") or {}
    architectures = hf_config.get("architectures") or []
    checks = {
        "qwen3_architecture": "Qwen3ForCausalLM" in architectures,
        "qwen3_model_type": hf_config.get("model_type") == "qwen3",
        "forty_attention_layers": hf_config.get("num_hidden_layers") == 40,
        "no_hybrid_layer_types": not hf_config.get("layer_types"),
        "no_sliding_window": not hf_config.get("sliding_window"),
        "bfloat16_runtime": model.get("dtype") == "torch.bfloat16",
        "unquantized_runtime": model.get("quantization") is None,
        "sixteen_token_kv_blocks": cache.get("block_size") == 16,
        "prefix_caching_enabled": cache.get("enable_prefix_caching") is True,
        "cacheselect_disabled": cache.get("enable_cacheselect") is False,
        "partial_reuse_disabled": (
            cache.get("cacheselect_execute_partial_reuse") is False
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "architecture": architectures,
        "model_type": hf_config.get("model_type"),
        "num_hidden_layers": hf_config.get("num_hidden_layers"),
        "runtime_dtype": model.get("dtype"),
        "runtime_quantization": model.get("quantization"),
        "kv_block_size": cache.get("block_size"),
    }


# Require two genuinely uncached requests to produce the same expected answer.
def assess_uncached_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = [row.get("output_text") for row in rows]
    checks = {
        "two_repetitions": len(rows) == 2,
        "both_uncached": len(rows) == 2
        and all(row.get("cached_tokens") == 0 for row in rows),
        "both_complete": len(rows) == 2
        and all(row.get("finish_reason") == "stop" for row in rows),
        "both_nonempty": len(rows) == 2
        and all(isinstance(output, str) and bool(output.strip()) for output in outputs),
        "exactly_stable": len(rows) == 2 and outputs[0] == outputs[1],
        "instruction_followed": len(rows) == 2
        and all(output == EXPECTED_OUTPUT for output in outputs),
    }
    return {"passed": all(checks.values()), "checks": checks, "outputs": outputs}


# Run the small validation workload and retain an append-only request ledger.
def run_full_attention_model_validation(
    *,
    base_url: str,
    model: str,
    run_id: str,
    request_log_dir: Path,
    server_info: dict[str, Any],
    hf_config: dict[str, Any],
    output: Path,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    configuration = assess_full_attention_configuration(server_info, hf_config)
    if not configuration["passed"]:
        raise RuntimeError(f"unexpected model configuration: {configuration['checks']}")

    recorder = RequestRecorder(
        run_id=run_id,
        model=model,
        backend="vllm-full-attention-validation",
        log_dir=request_log_dir,
        invocation_metadata={"experiment": "full_attention_model_validation"},
    )
    rows = []
    for repetition in range(2):
        salt = hashlib.sha256(
            f"{recorder.invocation_id}:uncached:{repetition}".encode()
        ).hexdigest()
        rows.append(
            _run_recorded_request(
                recorder=recorder,
                chat_url=f"{base_url.rstrip('/')}/v1/chat/completions",
                metrics_url=f"{base_url.rstrip('/')}/metrics",
                model=model,
                scenario="uncached_stability",
                role=f"repetition_{repetition + 1}",
                prompt=VALIDATION_PROMPT,
                cache_salt=salt,
                cacheselect_request_id=f"{run_id}:repetition:{repetition + 1}",
                cacheselect_source_request_id=None,
                max_completion_tokens=32,
                timeout_seconds=timeout_seconds,
            )
        )

    stability = assess_uncached_stability(rows)
    ledger = validate_ledger(recorder.path)
    result = {
        "schema_version": 1,
        "experiment": "full_attention_model_validation",
        "run_id": run_id,
        "model": model,
        "configuration": configuration,
        "uncached_stability": stability,
        "observations": rows,
        "request_ledger": str(recorder.path),
        "ledger_complete": ledger.is_complete,
        "passed": configuration["passed"]
        and stability["passed"]
        and ledger.is_complete,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


# Parse the standalone runner arguments.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-log-dir", type=Path, required=True)
    parser.add_argument("--server-info", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


# Validate the live server and return a failing status on any mismatch.
def main() -> None:
    args = _parse_args()
    result = run_full_attention_model_validation(
        base_url=args.base_url,
        model=args.model,
        run_id=args.run_id,
        request_log_dir=args.request_log_dir,
        server_info=json.loads(args.server_info.read_text(encoding="utf-8")),
        hf_config=json.loads(args.hf_config.read_text(encoding="utf-8")),
        output=args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"uncached_outputs_exact={result['uncached_stability']['passed']}")
    print(f"Saved {args.output}")
    if not result["passed"]:
        raise SystemExit("Qwen3-14B BF16 validation failed")


if __name__ == "__main__":
    main()
