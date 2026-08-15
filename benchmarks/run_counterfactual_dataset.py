"""Run one vLLM counterfactual transition and save its training rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.block_dataset import DatasetSplit
from benchmarks.counterfactual_workflow import run_counterfactual_dataset_workflow
from benchmarks.schema import load_trace
from observability.request_recorder import RequestRecorder


# Define the command-line surface independently from the workflow implementation.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--transition-id", required=True)
    parser.add_argument(
        "--split",
        choices=[split.value for split in DatasetSplit],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique request-ledger run ID; generated automatically when omitted.",
    )
    parser.add_argument(
        "--request-log-dir",
        type=Path,
        default=None,
        help="Directory for the append-only full-input/output request ledger.",
    )
    args = parser.parse_args()
    if args.max_completion_tokens < 1:
        parser.error("--max-completion-tokens must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


# Create one recorder, run the workflow, and persist its completion summary.
def main() -> None:
    args = _parse_args()
    trace = load_trace(args.trace)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    run_id = args.run_id or (
        f"counterfactual-{trace.trace_id}-{args.transition_id}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    recorder = RequestRecorder(
        run_id=run_id,
        model=args.model,
        backend="vllm-openai-compatible-server",
        log_dir=args.request_log_dir,
        invocation_metadata={
            "experiment": "counterfactual-block-dataset",
            "trace_id": trace.trace_id,
            "trace_path": str(args.trace),
            "transition_id": args.transition_id,
            "split": args.split,
            "dataset_path": str(args.output),
            "summary_path": str(summary_path),
            "endpoint": endpoint,
        },
    )
    result = run_counterfactual_dataset_workflow(
        trace=trace,
        transition_id=args.transition_id,
        split=DatasetSplit(args.split),
        output_path=args.output,
        url=endpoint,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        recorder=recorder,
    )
    summary = {
        "schema_version": 1,
        "experiment": "counterfactual-block-dataset",
        "run_id": run_id,
        "model": args.model,
        "endpoint": endpoint,
        "trace_path": str(args.trace),
        **result,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {result['valid_training_rows']} rows to {args.output}")
    print(f"Saved request ledger {result['request_ledger']}")
    print(f"Saved summary {summary_path}")


if __name__ == "__main__":
    main()
