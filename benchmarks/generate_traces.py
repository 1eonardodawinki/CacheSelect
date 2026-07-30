"""Generate the initial CacheSelect request-transition traces."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.schema import save_trace
from benchmarks.workloads import TRACE_BUILDERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/traces"),
    )
    parser.add_argument(
        "--workload",
        choices=[*TRACE_BUILDERS, "all"],
        default="all",
    )
    args = parser.parse_args()

    selected = (
        TRACE_BUILDERS
        if args.workload == "all"
        else {args.workload: TRACE_BUILDERS[args.workload]}
    )
    for name, builder in selected.items():
        trace = builder()
        path = args.output_dir / f"{name}.json"
        save_trace(trace, path)
        print(
            f"Saved {path}: {len(trace.requests)} requests, "
            f"{len(trace.transitions)} transitions"
        )


if __name__ == "__main__":
    main()
