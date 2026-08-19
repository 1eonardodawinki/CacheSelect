"""Validate and render one hybrid GDN break-even result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Validate the saved matrix and derive one conservative runtime recommendation.
def analyze_hybrid_gdn_breakeven(summary: dict[str, Any]) -> dict[str, Any]:
    """Return compact evidence only when all matrix accounting is consistent."""
    if summary.get("experiment") != "hybrid_gdn_reuse_breakeven":
        raise ValueError("input is not a hybrid GDN break-even result")
    raw_cells = summary.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("break-even result has no cells")

    cells = []
    seen_counts = set()
    for raw in raw_cells:
        count = raw.get("reused_block_count")
        repetitions = raw.get("repetitions")
        reuse_fraction = raw.get("mean_reuse_fraction")
        wall_speedup = raw.get("median_speedup")
        ttft_speedup = raw.get("median_ttft_speedup")
        if not isinstance(count, int) or count < 1 or count in seen_counts:
            raise ValueError("cell has an invalid or duplicate block count")
        if not isinstance(repetitions, int) or repetitions < 1:
            raise ValueError("cell has an invalid repetition count")
        if not isinstance(reuse_fraction, (int, float)) or not 0 < reuse_fraction < 1:
            raise ValueError("cell has an invalid reuse fraction")
        if not isinstance(wall_speedup, (int, float)) or wall_speedup <= 0:
            raise ValueError("cell has an invalid wall speedup")
        if not isinstance(ttft_speedup, (int, float)) or ttft_speedup <= 0:
            raise ValueError("cell has an invalid TTFT speedup")
        seen_counts.add(count)
        all_exact = raw.get("all_outputs_exact") is True
        measured_break_even = all_exact and wall_speedup > 1 and ttft_speedup > 1
        if raw.get("break_even_met") is not measured_break_even:
            raise ValueError("cell break-even flag disagrees with its measurements")
        cells.append(
            {
                "reused_block_count": count,
                "repetitions": repetitions,
                "reuse_fraction": float(reuse_fraction),
                "median_wall_speedup": float(wall_speedup),
                "median_ttft_speedup": float(ttft_speedup),
                "all_outputs_exact": all_exact,
                "break_even_met": measured_break_even,
            }
        )

    cells.sort(key=lambda cell: cell["reused_block_count"])
    if summary.get("trial_count") != sum(cell["repetitions"] for cell in cells):
        raise ValueError("trial count disagrees with matrix cells")
    first_break_even = next(
        (cell["reused_block_count"] for cell in cells if cell["break_even_met"]),
        None,
    )
    if summary.get("first_break_even_reused_block_count") != first_break_even:
        raise ValueError("reported first break-even point is inconsistent")
    all_exact = summary.get("all_outputs_exact") is True and all(
        cell["all_outputs_exact"] for cell in cells
    )
    decision = (
        "REJECT_OUTPUT_DRIFT"
        if not all_exact
        else "NO_BREAK_EVEN"
        if first_break_even is None
        else "MEASURED_BREAK_EVEN"
    )
    return {
        "schema_version": 1,
        "experiment": "hybrid_gdn_breakeven_analysis",
        "source_run_id": summary.get("run_id"),
        "model": summary.get("model"),
        "block_size": summary.get("block_size"),
        "all_outputs_exact": all_exact,
        "decision": decision,
        "recommended_minimum_reused_blocks": first_break_even,
        "cells": cells,
    }


# Render the validated evidence as a compact table for review and reporting.
def render_hybrid_gdn_breakeven_markdown(analysis: dict[str, Any]) -> str:
    """Return a human-readable summary without changing measured precision."""
    lines = [
        "# Hybrid GDN reuse break-even",
        "",
        f"Decision: **{analysis['decision']}**",
        "",
        "| Reused blocks | Reuse share | Median wall speedup | "
        "Median TTFT speedup | Exact output | Break-even |",
        "|---:|---:|---:|---:|:---:|:---:|",
    ]
    for cell in analysis["cells"]:
        lines.append(
            f"| {cell['reused_block_count']} | {cell['reuse_fraction']:.1%} | "
            f"{cell['median_wall_speedup']:.3f}x | "
            f"{cell['median_ttft_speedup']:.3f}x | "
            f"{'yes' if cell['all_outputs_exact'] else 'no'} | "
            f"{'yes' if cell['break_even_met'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


# Parse source and destination paths for the standalone analyzer.
def _parse_args() -> argparse.Namespace:
    """Return command-line arguments for one completed break-even matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


# Validate one result and save both machine-readable and report-ready artifacts.
def main() -> None:
    """Run the analyzer from the command line."""
    args = _parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze_hybrid_gdn_breakeven(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(
        render_hybrid_gdn_breakeven_markdown(analysis),
        encoding="utf-8",
    )
    print(f"decision={analysis['decision']}")
    print(f"Saved {args.output}")
    print(f"Saved {args.markdown_output}")


if __name__ == "__main__":
    main()
