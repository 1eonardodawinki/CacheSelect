"""Validate and summarize repeated CacheSelect performance trials."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

POSITIONS = ("early", "middle", "late")
MODES = ("shadow", "active")
RESULT_NAME = re.compile(
    r"^tokens-(\d+)-(early|middle|late)-(shadow|active)-rep-(\d+)\.json$"
)


# Extract one edited-request measurement from a completed two-request trial.
def _load_trial(path: Path) -> dict[str, Any]:
    match = RESULT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected performance result: {path.name}")
    target, position, mode, repetition = match.groups()
    result = json.loads(path.read_text(encoding="utf-8"))
    observations = result.get("observations") or []
    if len(observations) != 2:
        raise ValueError(f"{path}: expected source and edited observations")
    if result.get("request_ledger_summary") != {
        "started": 2,
        "completed": 2,
        "failed": 0,
    }:
        raise ValueError(f"{path}: incomplete request ledger")

    edited = observations[1]
    metrics = edited.get("server_metrics") or {}
    timing_names = (
        "cacheselect_copy_time_ms",
        "cacheselect_preparation_time_ms",
        "cacheselect_forward_time_ms",
        "time_to_first_token_ms",
    )
    if any(metrics.get(name) is None for name in timing_names):
        raise ValueError(f"{path}: missing CacheSelect timing metrics")
    if any(float(metrics[name]) < 0 for name in timing_names):
        raise ValueError(f"{path}: negative CacheSelect timing metric")
    if not (edited.get("quality") or {}).get("passed"):
        raise ValueError(f"{path}: edited response failed its quality gate")
    choices = (edited.get("raw_response") or {}).get("choices") or []
    if not choices or choices[0].get("finish_reason") != "stop":
        raise ValueError(f"{path}: edited response was truncated")
    prompt_tokens = int(edited["prompt_token_count"])
    if abs(prompt_tokens - int(target)) > 16:
        raise ValueError(
            f"{path}: prompt length {prompt_tokens} missed target {target}"
        )
    return {
        "target_tokens": int(target),
        "position": position,
        "mode": mode,
        "repetition": int(repetition),
        "prompt_tokens": prompt_tokens,
        "native_cached_tokens": int(edited["cached_tokens"]),
        "candidate_tokens": int(metrics["cacheselect_candidate_tokens"]),
        "reused_rows": int(metrics["cacheselect_reused_batch_rows"]),
        "compute_rows": int(metrics["cacheselect_compute_batch_rows"]),
        "span_count": int(metrics["cacheselect_compute_span_count"]),
        "executed": bool(metrics["cacheselect_compacted_batch_executed"]),
        "copy_time_ms": float(metrics["cacheselect_copy_time_ms"]),
        "preparation_time_ms": float(metrics["cacheselect_preparation_time_ms"]),
        "forward_time_ms": float(metrics["cacheselect_forward_time_ms"]),
        "ttft_ms": float(metrics["time_to_first_token_ms"]),
        "result_file": path.name,
    }


# Pair identical repetitions so condition deltas never mix unrelated trials.
def _pair_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["target_tokens"], row["position"], row["repetition"])
        grouped[key][row["mode"]] = row

    paired = []
    for (target, position, repetition), modes in sorted(grouped.items()):
        if set(modes) != set(MODES):
            raise ValueError(f"unpaired trial: {target}/{position}/rep-{repetition}")
        shadow, active = modes["shadow"], modes["active"]
        if shadow["executed"]:
            raise ValueError(f"shadow trial executed reuse: {shadow['result_file']}")
        if active["candidate_tokens"] > 0 and not active["executed"]:
            raise ValueError(f"active trial did not execute: {active['result_file']}")
        comparable_names = ("prompt_tokens", "candidate_tokens", "reused_rows")
        if any(shadow[name] != active[name] for name in comparable_names):
            raise ValueError(
                f"condition geometry differs: {shadow['result_file']} and "
                f"{active['result_file']}"
            )
        paired.append(
            {
                "target_tokens": target,
                "position": position,
                "repetition": repetition,
                "prompt_tokens": active["prompt_tokens"],
                "candidate_tokens": active["candidate_tokens"],
                "reused_rows": active["reused_rows"],
                "compute_rows": active["compute_rows"],
                "span_count": active["span_count"],
                "shadow_forward_ms": shadow["forward_time_ms"],
                "active_preparation_ms": active["preparation_time_ms"],
                "active_copy_ms": active["copy_time_ms"],
                "active_forward_ms": active["forward_time_ms"],
                "forward_delta_ms": active["forward_time_ms"]
                - shadow["forward_time_ms"],
                "shadow_ttft_ms": shadow["ttft_ms"],
                "active_ttft_ms": active["ttft_ms"],
                "ttft_delta_ms": active["ttft_ms"] - shadow["ttft_ms"],
            }
        )
    return paired


# Write stable tabular output for later plotting and report analysis.
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Compute one arithmetic mean from a group of paired trials.
def _mean(rows: list[dict[str, Any]], name: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows)


# Aggregate paired repetitions by prompt length and edit position.
def _summarize(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["target_tokens"], pair["position"])].append(pair)
    summaries = []
    for (target, position), rows in sorted(grouped.items()):
        summaries.append(
            {
                "target_tokens": target,
                "position": position,
                "repetitions": len(rows),
                "mean_reused_rows": _mean(rows, "reused_rows"),
                "mean_span_count": _mean(rows, "span_count"),
                "mean_shadow_forward_ms": _mean(rows, "shadow_forward_ms"),
                "mean_active_preparation_ms": _mean(
                    rows, "active_preparation_ms"
                ),
                "mean_active_copy_ms": _mean(rows, "active_copy_ms"),
                "mean_active_forward_ms": _mean(rows, "active_forward_ms"),
                "mean_forward_delta_ms": _mean(rows, "forward_delta_ms"),
                "mean_ttft_delta_ms": _mean(rows, "ttft_delta_ms"),
            }
        )
    return summaries


# Validate matrix completeness and emit paired and aggregate artifacts.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    args = parser.parse_args()

    rows = [_load_trial(path) for path in sorted(args.input_dir.glob("*.json"))]
    rows = [row for row in rows if row["target_tokens"] == args.target_tokens]
    expected = len(POSITIONS) * len(MODES) * args.repetitions
    if len(rows) != expected:
        raise ValueError(f"expected {expected} trials, found {len(rows)}")
    pairs = _pair_trials(rows)
    summaries = _summarize(pairs)
    _write_csv(args.output_dir / "paired-trials.csv", pairs)
    _write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    for summary in summaries:
        print(
            f"tokens={summary['target_tokens']} position={summary['position']} "
            f"reused={summary['mean_reused_rows']:.1f} "
            f"spans={summary['mean_span_count']:.1f} "
            f"forward_delta_ms={summary['mean_forward_delta_ms']:+.3f} "
            f"ttft_delta_ms={summary['mean_ttft_delta_ms']:+.3f}"
        )


if __name__ == "__main__":
    main()
