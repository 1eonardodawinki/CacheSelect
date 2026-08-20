"""Combine repeated hybrid GDN break-even runs without hiding timing noise."""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from typing import Any

from benchmarks.analyze_hybrid_gdn_breakeven import (
    analyze_hybrid_gdn_breakeven,
)


# Estimate uncertainty in a median without assuming normally distributed timings.
def _bootstrap_median_interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: str,
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap 95% interval."""
    if not values:
        raise ValueError("bootstrap values must not be empty")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    )
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return float(medians[lower_index]), float(medians[upper_index])


# Validate one source artifact before any trials are pooled across experiments.
def _validated_run_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized timing rows from one internally consistent run."""
    validated = analyze_hybrid_gdn_breakeven(summary)
    run_id = summary.get("run_id")
    capacity = summary.get("cache_capacity")
    trials = summary.get("trials")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("source run has no run_id")
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError("source run has an invalid cache capacity")
    if not isinstance(trials, list) or len(trials) != summary.get("trial_count"):
        raise ValueError("source run has inconsistent trial rows")

    rows = []
    for trial in trials:
        count = trial.get("reused_block_count")
        wall = trial.get("speedup")
        ttft = trial.get("ttft_speedup")
        reuse_fraction = trial.get("reuse_fraction")
        if not isinstance(count, int) or count < 1:
            raise ValueError("trial has an invalid reused block count")
        if not isinstance(wall, (int, float)) or wall <= 0:
            raise ValueError("trial has an invalid wall speedup")
        if not isinstance(ttft, (int, float)) or ttft <= 0:
            raise ValueError("trial has an invalid TTFT speedup")
        if not isinstance(reuse_fraction, (int, float)) or not 0 < reuse_fraction < 1:
            raise ValueError("trial has an invalid reuse fraction")
        rows.append(
            {
                "run_id": run_id,
                "cache_capacity": capacity,
                "reused_block_count": count,
                "reuse_fraction": float(reuse_fraction),
                "wall_speedup": float(wall),
                "ttft_speedup": float(ttft),
                "exact_output_match": trial.get("exact_output_match") is True,
                "source_all_outputs_exact": validated["all_outputs_exact"],
            }
        )
    return rows


# Pool compatible trials and require replication before claiming a real speedup.
def analyze_hybrid_gdn_replication(
    summaries: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return uncertainty-aware evidence across two or more source runs."""
    if len(summaries) < 2:
        raise ValueError("replication analysis requires at least two runs")
    run_ids = [summary.get("run_id") for summary in summaries]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("source run IDs must be unique")
    models = {summary.get("model") for summary in summaries}
    block_sizes = {summary.get("block_size") for summary in summaries}
    if len(models) != 1 or not all(isinstance(model, str) for model in models):
        raise ValueError("source runs must use one model")
    if len(block_sizes) != 1 or not all(
        isinstance(block_size, int) and block_size > 0 for block_size in block_sizes
    ):
        raise ValueError("source runs must use one block size")

    rows = [row for summary in summaries for row in _validated_run_rows(summary)]
    cells = []
    for count in sorted({row["reused_block_count"] for row in rows}):
        selected = [row for row in rows if row["reused_block_count"] == count]
        wall = [row["wall_speedup"] for row in selected]
        ttft = [row["ttft_speedup"] for row in selected]
        wall_interval = _bootstrap_median_interval(
            wall, samples=bootstrap_samples, seed=f"{seed}:{count}:wall"
        )
        ttft_interval = _bootstrap_median_interval(
            ttft, samples=bootstrap_samples, seed=f"{seed}:{count}:ttft"
        )
        source_runs = sorted({row["run_id"] for row in selected})
        all_exact = all(
            row["exact_output_match"] and row["source_all_outputs_exact"]
            for row in selected
        )
        replicated_faster = (
            all_exact
            and len(source_runs) >= 2
            and wall_interval[0] > 1.0
            and ttft_interval[0] > 1.0
        )
        if not all_exact:
            evidence = "OUTPUT_DRIFT"
        elif replicated_faster:
            evidence = "REPLICATED_FASTER"
        elif wall_interval[1] < 1.0 and ttft_interval[1] < 1.0:
            evidence = "OBSERVED_SLOWER"
        else:
            evidence = "INCONCLUSIVE"
        cells.append(
            {
                "reused_block_count": count,
                "trial_count": len(selected),
                "source_run_count": len(source_runs),
                "source_run_ids": source_runs,
                "cache_capacities": sorted(
                    {row["cache_capacity"] for row in selected}
                ),
                "mean_reuse_fraction": statistics.fmean(
                    row["reuse_fraction"] for row in selected
                ),
                "median_wall_speedup": statistics.median(wall),
                "wall_speedup_95pct_interval": list(wall_interval),
                "wall_faster_trial_count": sum(value > 1.0 for value in wall),
                "median_ttft_speedup": statistics.median(ttft),
                "ttft_speedup_95pct_interval": list(ttft_interval),
                "ttft_faster_trial_count": sum(value > 1.0 for value in ttft),
                "all_outputs_exact": all_exact,
                "evidence": evidence,
            }
        )

    all_exact = all(cell["all_outputs_exact"] for cell in cells)
    replicated_counts = [
        cell["reused_block_count"]
        for cell in cells
        if cell["evidence"] == "REPLICATED_FASTER"
    ]
    decision = (
        "REJECT_OUTPUT_DRIFT"
        if not all_exact
        else "REPLICATED_SPEEDUP"
        if replicated_counts
        else "NO_RELIABLE_SPEEDUP"
    )
    return {
        "schema_version": 1,
        "experiment": "hybrid_gdn_breakeven_replication_analysis",
        "model": next(iter(models)),
        "block_size": next(iter(block_sizes)),
        "source_run_ids": run_ids,
        "source_run_count": len(run_ids),
        "trial_count": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "all_outputs_exact": all_exact,
        "decision": decision,
        "replicated_speedup_block_counts": replicated_counts,
        "cells": cells,
    }
