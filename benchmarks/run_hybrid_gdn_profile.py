"""Profile full computation and active GDN reuse for one controlled edit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from benchmarks.hybrid_gdn_breakeven import calibrate_hybrid_gdn_breakeven_prompt
from benchmarks.hybrid_gdn_profile import capture_profiled_request
from benchmarks.run_hybrid_apc_baseline import _run_recorded_request
from benchmarks.run_hybrid_gdn_breakeven import (
    _tokenize_prompt,
    _validate_active_trial,
)
from observability.request_recorder import RequestRecorder, validate_ledger


# Preserve the single-GPU scope totals before the next profile overwrites them.
def _copy_scope_summary(profile_dir: Path, output: Path) -> Path:
    """Copy the current worker summary to one role-specific artifact."""
    source = profile_dir / "cacheselect_scope_summary_0.json"
    if not source.is_file():
        raise RuntimeError("Torch profiler omitted the CacheSelect scope summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return output


# Compare one full prefill with one proven active-reuse prefill on a warm server.
def run_hybrid_gdn_component_profile(
    *,
    base_url: str,
    model: str,
    run_id: str,
    output: Path,
    request_log_dir: Path,
    profile_dir: Path,
    reused_block_count: int,
    block_size: int,
    max_completion_tokens: int = 64,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Run and save an isolated full-versus-active component profile."""
    if reused_block_count < 1:
        raise ValueError("reused_block_count must be positive")
    profile_dir.mkdir(parents=True, exist_ok=True)
    pair = calibrate_hybrid_gdn_breakeven_prompt(
        reused_block_count=reused_block_count,
        block_size=block_size,
        tokenize_prompt=lambda prompt: _tokenize_prompt(
            base_url=base_url,
            model=model,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
        ),
    )
    recorder = RequestRecorder(
        run_id=run_id,
        model=model,
        backend="vllm-hybrid-gdn-profile",
        log_dir=request_log_dir,
        invocation_metadata={"experiment": "hybrid_gdn_component_profile"},
    )
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    reference_salt = hashlib.sha256(
        f"{recorder.invocation_id}:reference".encode()
    ).hexdigest()
    pair_salt = hashlib.sha256(
        f"{recorder.invocation_id}:active-pair".encode()
    ).hexdigest()
    source_request_id = f"{run_id}:source"

    # Profile the independent reference before populating the reusable sidecar.
    reference, reference_trace = capture_profiled_request(
        base_url=base_url,
        profile_dir=profile_dir,
        timeout_seconds=timeout_seconds,
        run_request=lambda: _run_recorded_request(
            recorder=recorder,
            chat_url=chat_url,
            metrics_url=metrics_url,
            model=model,
            scenario="component-profile",
            role="reference",
            prompt=pair.target_prompt,
            cache_salt=reference_salt,
            cacheselect_request_id=f"{run_id}:reference",
            cacheselect_source_request_id=None,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
        ),
    )
    reference_scopes = _copy_scope_summary(
        profile_dir,
        profile_dir.parent / "full-reference-scopes.json",
    )
    source = _run_recorded_request(
        recorder=recorder,
        chat_url=chat_url,
        metrics_url=metrics_url,
        model=model,
        scenario="component-profile",
        role="source",
        prompt=pair.source_prompt,
        cache_salt=pair_salt,
        cacheselect_request_id=source_request_id,
        cacheselect_source_request_id=None,
        max_completion_tokens=max_completion_tokens,
        timeout_seconds=timeout_seconds,
    )

    # Profile only the edited target; the donor request remains outside the window.
    target, active_trace = capture_profiled_request(
        base_url=base_url,
        profile_dir=profile_dir,
        timeout_seconds=timeout_seconds,
        run_request=lambda: _run_recorded_request(
            recorder=recorder,
            chat_url=chat_url,
            metrics_url=metrics_url,
            model=model,
            scenario="component-profile",
            role="target",
            prompt=pair.target_prompt,
            cache_salt=pair_salt,
            cacheselect_request_id=f"{run_id}:target",
            cacheselect_source_request_id=source_request_id,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
        ),
    )
    active_scopes = _copy_scope_summary(
        profile_dir,
        profile_dir.parent / "active-reuse-scopes.json",
    )
    execution = _validate_active_trial(
        pair=pair,
        block_size=block_size,
        reference=reference,
        source=source,
        target=target,
    )
    ledger = validate_ledger(recorder.path)
    if not ledger.is_complete or ledger.failed:
        raise RuntimeError("request ledger is incomplete or contains failures")

    result = {
        "schema_version": 1,
        "experiment": "hybrid_gdn_component_profile",
        "run_id": run_id,
        "model": model,
        "block_size": block_size,
        "reused_block_count": reused_block_count,
        "prompt_token_count": pair.prompt_token_count,
        "all_outputs_exact": execution["exact_output_match"],
        "execution": execution,
        "profile_traces": {
            "full_reference": str(reference_trace),
            "active_reuse": str(active_trace),
        },
        "profile_scope_summaries": {
            "full_reference": str(reference_scopes),
            "active_reuse": str(active_scopes),
        },
        "request_ledger": str(recorder.path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


# Parse one profiling condition and its artifact locations.
def _parse_args() -> argparse.Namespace:
    """Return command-line arguments for the component profiler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-log-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--reused-block-count", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


# Run the profile and print only its decisive artifact locations.
def main() -> None:
    """Execute the controlled component profile."""
    args = _parse_args()
    result = run_hybrid_gdn_component_profile(
        base_url=args.base_url,
        model=args.model,
        run_id=args.run_id,
        output=args.output,
        request_log_dir=args.request_log_dir,
        profile_dir=args.profile_dir,
        reused_block_count=args.reused_block_count,
        block_size=args.block_size,
        max_completion_tokens=args.max_completion_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"all_outputs_exact={result['all_outputs_exact']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
