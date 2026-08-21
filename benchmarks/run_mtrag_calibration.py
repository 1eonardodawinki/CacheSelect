"""Run uncached MTRAG answers and save quality-calibration evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.mtrag import load_mtrag_tasks, mtrag_source_sha256
from benchmarks.mtrag_calibration import (
    MTRAG_UNCALIBRATED_GATE_ID,
    build_mtrag_reference_calibration_cases,
    run_mtrag_reference_calibration,
)
from benchmarks.schema import ReferenceSimilarityGate
from observability.request_recorder import RequestRecorder, validate_ledger


# Define the calibration command independently from its HTTP workflow.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--manifest-model",
        default=None,
        help="Model that originally selected the frozen task manifest.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=384)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--request-log-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.max_completion_tokens < 1 or args.timeout_seconds <= 0:
        parser.error("completion tokens and timeout must be positive")
    return args


# Load, validate, execute, and save one complete reference calibration run.
def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_sha256 = mtrag_source_sha256(args.input)
    if manifest.get("source_sha256") != source_sha256:
        raise ValueError("raw MTRAG input does not match the frozen manifest")
    gate = ReferenceSimilarityGate(
        0.0,
        0.0,
        1.0,
        MTRAG_UNCALIBRATED_GATE_ID,
    )
    manifest_model = args.manifest_model or args.model
    cases = build_mtrag_reference_calibration_cases(
        load_mtrag_tasks(args.input),
        manifest,
        quality_gate=gate,
        expected_model=manifest_model,
    )
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    run_id = args.run_id or (
        "mtrag-reference-calibration-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    recorder = RequestRecorder(
        run_id=run_id,
        model=args.model,
        backend="vllm-openai-compatible-server",
        log_dir=args.request_log_dir,
        invocation_metadata={
            "experiment": "mtrag-reference-quality-calibration",
            "manifest": str(args.manifest),
            "manifest_selection_model": manifest_model,
            "source_sha256": source_sha256,
            "output": str(args.output),
            "endpoint": endpoint,
        },
    )
    result = run_mtrag_reference_calibration(
        cases,
        url=endpoint,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        recorder=recorder,
    )
    ledger = validate_ledger(recorder.path)
    if not ledger.is_complete or ledger.failed or ledger.completed != len(cases):
        raise RuntimeError("MTRAG calibration request ledger is incomplete")
    artifact = {
        **result,
        "run_id": run_id,
        "model": args.model,
        "manifest_selection_model": manifest_model,
        "manifest": str(args.manifest),
        "source_sha256": source_sha256,
        "endpoint": endpoint,
        "request_ledger": str(ledger.path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(cases)} uncached MTRAG reference answers")
    print(f"Saved calibration evidence to {args.output}")
    print(f"Saved request ledger to {ledger.path}")


if __name__ == "__main__":
    main()
