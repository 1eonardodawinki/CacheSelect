"""Plan and aggregate graduated-context active GDN evaluation runs."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HybridGDNMatrixCondition:
    """One context-size repetition in the active evaluation matrix."""

    record_count: int
    repetition: int


# Interleave context sizes so time drift does not affect one size exclusively.
def build_hybrid_gdn_matrix_conditions(
    record_counts: tuple[int, ...],
    repetitions: int,
) -> tuple[HybridGDNMatrixCondition, ...]:
    """Return a deterministic, validated evaluation order."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not record_counts:
        raise ValueError("record_counts must not be empty")
    if len(set(record_counts)) != len(record_counts):
        raise ValueError("record_counts must be unique")
    if any(count < 20 for count in record_counts):
        raise ValueError("every record count must be at least 20")
    return tuple(
        HybridGDNMatrixCondition(record_count, repetition)
        for repetition in range(1, repetitions + 1)
        for record_count in record_counts
    )


# Aggregate active analyses by context size while retaining every raw run.
def summarize_hybrid_gdn_matrix(
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report output validity and latency distributions by scenario and size."""
    if not analyses:
        raise ValueError("analyses must not be empty")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for analysis in analyses:
        record_count = analysis.get("record_count")
        comparisons = analysis.get("comparisons")
        if not isinstance(record_count, int) or record_count < 20:
            raise ValueError("analysis has an invalid record_count")
        if not isinstance(comparisons, list) or not comparisons:
            raise ValueError("analysis has no comparisons")
        for comparison in comparisons:
            scenario = comparison.get("scenario")
            speedup = comparison.get("speedup")
            if not isinstance(scenario, str) or not scenario:
                raise ValueError("comparison has an invalid scenario")
            if not isinstance(speedup, (int, float)) or speedup <= 0:
                raise ValueError("comparison has an invalid speedup")
            grouped.setdefault((record_count, scenario), []).append(comparison)

    cells = []
    for (record_count, scenario), rows in sorted(grouped.items()):
        speedups = [float(row["speedup"]) for row in rows]
        cells.append(
            {
                "record_count": record_count,
                "scenario": scenario,
                "repetitions": len(rows),
                "all_outputs_exact": all(
                    row.get("exact_output_match") is True for row in rows
                ),
                "mean_speedup": statistics.fmean(speedups),
                "minimum_speedup": min(speedups),
                "maximum_speedup": max(speedups),
                "mean_native_cached_tokens": statistics.fmean(
                    float(row["native_cached_tokens"]) for row in rows
                ),
                "mean_reused_layer_tokens": statistics.fmean(
                    float(row["reused_layer_tokens"]) for row in rows
                ),
            }
        )
    return {
        "schema_version": 1,
        "experiment": "hybrid_gdn_active_matrix",
        "run_count": len(analyses),
        "all_outputs_exact": all(
            analysis.get("all_outputs_exact") is True for analysis in analyses
        ),
        "cells": cells,
    }
