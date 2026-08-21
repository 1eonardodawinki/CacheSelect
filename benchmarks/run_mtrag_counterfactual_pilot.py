"""Run the audited MTRAG counterfactual pilot against one vLLM server."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.mtrag import load_mtrag_tasks, mtrag_source_sha256
from benchmarks.mtrag_counterfactual import run_mtrag_counterfactual_cases
from benchmarks.mtrag_quality import load_mtrag_manual_quality_audit
from benchmarks.mtrag_trace import build_mtrag_counterfactual_cases
from benchmarks.reviewed_mtrag_counterfactual import (
    build_reviewed_mtrag_counterfactual_cases,
    load_reviewed_mtrag_reference_set,
)
from observability.request_recorder import RequestRecorder, validate_ledger


# Define the command-line inputs independently from the experiment workflow.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    approval = parser.add_mutually_exclusive_group(required=True)
    approval.add_argument("--audit", type=Path)
    approval.add_argument("--references", type=Path)
    parser.add_argument("--reference-artifact", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=384)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--request-log-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.max_completion_tokens < 1 or args.timeout_seconds <= 0:
        parser.error("completion tokens and timeout must be positive")
    if args.audit and not args.reference_artifact:
        parser.error("--audit requires --reference-artifact")
    return args


# Validate frozen inputs, run every case, and save the complete pilot summary.
def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_sha256 = mtrag_source_sha256(args.input)
    if args.references:
        references = load_reviewed_mtrag_reference_set(
            args.references, expected_model=args.model
        )
        if references.source_dataset_sha256 != source_sha256:
            raise ValueError("MTRAG source and reviewed references do not match")
        cases = build_reviewed_mtrag_counterfactual_cases(
            load_mtrag_tasks(args.input),
            manifest,
            references,
            expected_model=args.model,
        )
        outputs = references.reference_outputs
        require_exact_reference = True
        experiment = "qwen3-reviewed-mtrag-counterfactual"
        approval_metadata = {
            "references": str(args.references),
            "reference_manifest_sha256": references.manifest_sha256,
            "review_status": references.status,
        }
    else:
        audit = load_mtrag_manual_quality_audit(
            args.audit,
            reference_artifact_path=args.reference_artifact,
            expected_model=args.model,
        )
        if (
            manifest.get("source_sha256") != source_sha256
            or audit.source_dataset_sha256 != source_sha256
            or manifest.get("source_revision") != audit.source_dataset_revision
        ):
            raise ValueError("MTRAG source, pilot and audit provenance do not match")
        cases = build_mtrag_counterfactual_cases(
            load_mtrag_tasks(args.input),
            manifest,
            quality_gate=audit.quality_gate,
            approved_task_ids=audit.approved_task_ids,
            expected_model=args.model,
        )
        outputs = audit.approved_reference_outputs
        require_exact_reference = False
        experiment = "mtrag-counterfactual-pilot"
        approval_metadata = {"manual_audit": str(args.audit)}
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    run_id = args.run_id or (
        "mtrag-counterfactual-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    summary_path = args.summary_output or args.output_dir / "summary.json"
    recorder = RequestRecorder(
        run_id=run_id,
        model=args.model,
        backend="vllm-mtrag-counterfactual",
        log_dir=args.request_log_dir,
        invocation_metadata={
            "experiment": experiment,
            "manifest": str(args.manifest),
            **approval_metadata,
            "source_sha256": source_sha256,
            "output_dir": str(args.output_dir),
            "endpoint": endpoint,
        },
    )
    result = run_mtrag_counterfactual_cases(
        cases,
        reference_outputs=outputs,
        output_dir=args.output_dir,
        url=endpoint,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        recorder=recorder,
        require_reference_output_match=require_exact_reference,
    )
    ledger = validate_ledger(recorder.path)
    if not ledger.is_complete or ledger.failed:
        raise RuntimeError("MTRAG counterfactual request ledger is incomplete")
    summary = {
        **result,
        "experiment": experiment,
        "run_id": run_id,
        "model": args.model,
        "endpoint": endpoint,
        "source_sha256": source_sha256,
        "manifest": str(args.manifest),
        **approval_metadata,
        "request_ledger": str(ledger.path),
        "recorded_requests": ledger.started,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Completed {result['case_count']} MTRAG cases and "
        f"{result['trial_count']} block trials"
    )
    print(f"Saved pilot summary to {summary_path}")
    print(f"Saved full request ledger to {ledger.path}")


if __name__ == "__main__":
    main()
