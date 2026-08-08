"""Validate and summarize the CacheSelect shadow repair-policy matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITIONS = {
    "full-block": {
        "selector": "full_block",
        "candidate_tokens": 80,
        "repair_tokens": 80,
        "skipped_repair_tokens": 0,
    },
    "edit-radius-0": {
        "selector": "edit_proximity",
        "candidate_tokens": 80,
        "repair_tokens": 0,
        "skipped_repair_tokens": 80,
    },
    "edit-radius-1": {
        "selector": "edit_proximity",
        "candidate_tokens": 80,
        "repair_tokens": 48,
        "skipped_repair_tokens": 32,
    },
    "edit-radius-2": {
        "selector": "edit_proximity",
        "candidate_tokens": 80,
        "repair_tokens": 80,
        "skipped_repair_tokens": 0,
    },
}


# Load one condition result produced by the vLLM benchmark runner.
def _load_result(input_dir: Path, condition: str) -> dict[str, Any]:
    path = input_dir / f"rag-{condition}.json"
    if not path.is_file():
        raise ValueError(f"missing repair-policy result: {path}")
    return json.loads(path.read_text())


# Extract the one RAG observation that contains aligned reuse candidates.
def _candidate_observation(
    result: dict[str, Any], condition: str
) -> dict[str, Any]:
    observations = result.get("observations") or []
    candidates = [
        observation
        for observation in observations
        if (observation.get("server_metrics") or {}).get(
            "cacheselect_candidate_tokens"
        )
        is not None
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{condition}: expected one candidate observation, got {len(candidates)}"
        )
    return candidates[0]


# Build a deterministic signature for behavior that shadow policies cannot change.
def _behavior_signature(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": observation["request_id"],
            "prompt_token_ids": observation["prompt_token_ids"],
            "cached_tokens": observation["cached_tokens"],
            "output_text": observation["output_text"],
            "quality_passed": observation["quality"]["passed"],
        }
        for observation in result["observations"]
    ]


# Validate one matrix condition and return its normalized selector metrics.
def validate_repair_policy_condition(
    result: dict[str, Any], condition: str
) -> dict[str, Any]:
    expected = CONDITIONS.get(condition)
    if expected is None:
        raise ValueError(f"unknown repair-policy condition: {condition}")

    observations = result.get("observations") or []
    ledger = result.get("request_ledger_summary")
    if len(observations) != 4:
        raise ValueError(
            f"{condition}: expected 4 observations, got {len(observations)}"
        )
    if ledger != {"started": 4, "completed": 4, "failed": 0}:
        raise ValueError(f"{condition}: incomplete request ledger {ledger}")
    if not all(
        observation["quality"]["passed"] for observation in observations
    ):
        raise ValueError(f"{condition}: at least one quality check failed")

    observation = _candidate_observation(result, condition)
    metrics = observation["server_metrics"]
    actual = {
        "selector": metrics.get("cacheselect_repair_selector"),
        "candidate_tokens": metrics.get("cacheselect_candidate_tokens"),
        "repair_tokens": metrics.get("cacheselect_repair_tokens"),
        "skipped_repair_tokens": metrics.get(
            "cacheselect_skipped_repair_tokens"
        ),
    }
    if actual != expected:
        raise ValueError(
            f"{condition}: expected repair metrics {expected}, got {actual}"
        )
    return actual


# Validate all four policy conditions and return a compact experiment summary.
def analyze_repair_policy_shadow(input_dir: Path) -> dict[str, Any]:
    results = {
        condition: _load_result(input_dir, condition)
        for condition in CONDITIONS
    }
    reference_signature = _behavior_signature(results["full-block"])
    summary: dict[str, Any] = {"conditions": {}}

    for condition in CONDITIONS:
        result = results[condition]
        actual = validate_repair_policy_condition(result, condition)
        if _behavior_signature(result) != reference_signature:
            raise ValueError(f"{condition}: shadow policy changed request behavior")
        summary["conditions"][condition] = actual

    summary["behavior_identical"] = True
    summary["quality_passed"] = True
    return summary


# Parse command-line paths and persist the validated summary.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = analyze_repair_policy_shadow(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
