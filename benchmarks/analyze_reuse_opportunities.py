"""Analyze block-level reuse opportunities in one benchmark result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cacheselect.reuse_opportunity import analyze_reuse_opportunity


def _native_cached_tokens(
    observation: dict[str, Any],
    transition: dict[str, Any],
) -> int:
    runtime_policy = observation.get("runtime_policy") or {}
    native_candidate = runtime_policy.get("native_cached_tokens")
    if native_candidate is not None:
        return int(native_candidate)

    transition_policy = transition.get("runtime_policy") or {}
    native_candidate = transition_policy.get("native_cached_tokens")
    if native_candidate is not None:
        return int(native_candidate)

    cached_tokens = observation.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = transition.get("current_cached_tokens")
    if cached_tokens is None:
        raise ValueError(
            f"{transition.get('transition_id')}: native cached tokens are missing"
        )
    return int(cached_tokens)


def analyze_benchmark_result(
    result: dict[str, Any],
    *,
    block_size: int,
    source: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe reuse-opportunity report from a benchmark result."""
    observations = {
        observation["request_id"]: observation for observation in result["observations"]
    }
    transition_reports: list[dict[str, Any]] = []
    for transition in result["transitions"]:
        transition_id = transition["transition_id"]
        try:
            previous = observations[transition["previous_request_id"]]
            current = observations[transition["current_request_id"]]
        except KeyError as error:
            raise ValueError(
                f"{transition_id}: referenced observation {error.args[0]!r} is missing"
            ) from error

        previous_tokens = previous.get("prompt_token_ids")
        current_tokens = current.get("prompt_token_ids")
        if previous_tokens is None or current_tokens is None:
            raise ValueError(f"{transition_id}: rendered prompt token IDs are missing")

        native_cached_tokens = _native_cached_tokens(current, transition)
        opportunity = analyze_reuse_opportunity(
            previous_tokens,
            current_tokens,
            native_cached_tokens=native_cached_tokens,
            block_size=block_size,
        )
        transition_reports.append(
            {
                "transition_id": transition_id,
                "previous_request_id": transition["previous_request_id"],
                "current_request_id": transition["current_request_id"],
                "ground_truth": transition.get("ground_truth"),
                "opportunity": opportunity.to_dict(),
            }
        )

    opportunities = [report["opportunity"] for report in transition_reports]
    total_native_recompute = sum(
        item["current_token_count"] - item["native_cached_tokens"]
        for item in opportunities
    )
    total_candidate_tokens = sum(
        item["candidate_token_count"] for item in opportunities
    )
    summary = {
        "transition_count": len(transition_reports),
        "native_recompute_tokens": total_native_recompute,
        "location_independent_candidate_tokens": total_candidate_tokens,
        "candidate_share_of_native_recompute": (
            total_candidate_tokens / total_native_recompute
            if total_native_recompute
            else 0.0
        ),
        "candidate_block_count": sum(
            item["candidate_block_count"] for item in opportunities
        ),
        "whole_source_block_count": sum(
            item["whole_source_block_count"] for item in opportunities
        ),
        "repacking_required_block_count": sum(
            item["repacking_required_block_count"] for item in opportunities
        ),
        "moved_candidate_block_count": sum(
            item["moved_candidate_block_count"] for item in opportunities
        ),
    }
    return {
        "schema_version": 1,
        "analysis": "location-independent-block-content-opportunity",
        "source": source,
        "trace_id": result.get("trace_id"),
        "workload": result.get("workload"),
        "model": result.get("model"),
        "block_size": block_size,
        "interpretation": (
            "Candidate blocks have token content found in the previous rendered "
            "prompt. They are not safe KV hits: contextual state still requires "
            "validation or repair."
        ),
        "summary": summary,
        "transitions": transition_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    result = json.loads(args.input.read_text())
    report = analyze_benchmark_result(
        result,
        block_size=args.block_size,
        source=str(args.input),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    summary = report["summary"]
    print(f"Saved {args.output}")
    print(
        "candidate_tokens="
        f"{summary['location_independent_candidate_tokens']} "
        f"native_recompute_tokens={summary['native_recompute_tokens']} "
        "candidate_share="
        f"{summary['candidate_share_of_native_recompute']:.1%}"
    )


if __name__ == "__main__":
    main()
