"""Run a graduated-context active GDN evaluation against one live vLLM server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.analyze_hybrid_gdn_active import analyze_hybrid_gdn_active
from benchmarks.hybrid_gdn_matrix import (
    build_hybrid_gdn_matrix_conditions,
    summarize_hybrid_gdn_matrix,
)
from benchmarks.run_hybrid_apc_baseline import run_hybrid_apc_baseline


# Run every isolated matrix condition and preserve its raw and analyzed artifacts.
def run_hybrid_gdn_matrix(
    *,
    base_url: str,
    model: str,
    run_id: str,
    output_root: Path,
    request_log_root: Path,
    record_counts: tuple[int, ...],
    repetitions: int,
    max_completion_tokens: int = 64,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Execute active runs sequentially against one warm model server."""
    conditions = build_hybrid_gdn_matrix_conditions(record_counts, repetitions)
    analyses = []
    runs = []
    for condition in conditions:
        condition_id = (
            f"records-{condition.record_count}-rep-{condition.repetition:02d}"
        )
        condition_run_id = f"{run_id}-{condition_id}"
        condition_root = output_root / condition_id
        summary_path = condition_root / "summary.json"
        analysis_path = condition_root / "active-analysis.json"
        result = run_hybrid_apc_baseline(
            base_url=base_url,
            model=model,
            run_id=condition_run_id,
            request_log_dir=request_log_root / condition_id,
            output=summary_path,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
            validate_against_reference=True,
            require_token_aligned_edits=True,
            record_count=condition.record_count,
        )
        if result["gdn_delta_active"]["passed"] is not True:
            raise RuntimeError(f"{condition_id} did not execute active GDN reuse")
        if result["reference_validation_passed"] is not True:
            raise RuntimeError(f"{condition_id} changed a full-reference output")

        analysis = analyze_hybrid_gdn_active(result)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(
            json.dumps(analysis, indent=2) + "\n",
            encoding="utf-8",
        )
        analyses.append(analysis)
        runs.append(
            {
                "condition_id": condition_id,
                "record_count": condition.record_count,
                "repetition": condition.repetition,
                "summary": str(summary_path),
                "analysis": str(analysis_path),
                "request_ledger": result["request_ledger"],
            }
        )

    matrix = summarize_hybrid_gdn_matrix(analyses)
    matrix.update(
        {
            "run_id": run_id,
            "model": model,
            "record_counts": list(record_counts),
            "repetitions": repetitions,
            "max_completion_tokens": max_completion_tokens,
            "runs": runs,
        }
    )
    matrix_path = output_root / "matrix-summary.json"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    return matrix


# Parse the live server and matrix geometry for the standalone runner.
def _parse_args() -> argparse.Namespace:
    """Return command-line arguments for a graduated active evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-log-root", type=Path, required=True)
    parser.add_argument("--record-counts", type=int, nargs="+", default=(64, 112, 160))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


# Execute and print the location of the consolidated matrix artifact.
def main() -> None:
    """Run the requested matrix and report its high-level outcome."""
    args = _parse_args()
    matrix = run_hybrid_gdn_matrix(
        base_url=args.base_url,
        model=args.model,
        run_id=args.run_id,
        output_root=args.output_root,
        request_log_root=args.request_log_root,
        record_counts=tuple(args.record_counts),
        repetitions=args.repetitions,
        max_completion_tokens=args.max_completion_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"Completed {matrix['run_count']} active GDN runs")
    print(f"all_outputs_exact={matrix['all_outputs_exact']}")
    print(f"Saved {args.output_root / 'matrix-summary.json'}")


if __name__ == "__main__":
    main()
