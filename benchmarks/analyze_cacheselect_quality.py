"""Validate and consolidate a CacheSelect quality-stress matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

TARGETS = (256, 1024, 4096)
RADII = (0, 1, 2)
POSITIONS = ("early", "middle", "late")


# Parse the simple key-value provenance emitted beside each condition.
def _load_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing condition metadata: {path}")
    metadata = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"invalid metadata line in {path}: {line}")
        metadata[key] = value
    return metadata


# Load and validate one prompt-length and repair-radius condition summary.
def _load_condition(
    input_root: Path,
    target_tokens: int,
    edit_radius: int,
) -> list[dict[str, Any]]:
    condition_dir = input_root / f"tokens-{target_tokens}" / f"radius-{edit_radius}"
    metadata = _load_metadata(condition_dir / "metadata.env")
    expected_metadata = {
        "target_tokens": str(target_tokens),
        "edit_radius": str(edit_radius),
        "answer_sensitive": "1",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{condition_dir}: expected {key}={expected}, "
                f"got {metadata.get(key)!r}"
            )

    summary_path = condition_dir / "analysis" / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing condition summary: {summary_path}")
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    if {row.get("position") for row in summaries} != set(POSITIONS):
        raise ValueError(f"{summary_path}: incomplete edit-position coverage")

    rows = []
    for summary in summaries:
        if int(summary.get("target_tokens", -1)) != target_tokens:
            raise ValueError(f"{summary_path}: incorrect target token label")
        for quality_name in (
            "active_quality_pass_rate",
            "active_shadow_exact_match_rate",
            "mean_active_shadow_word_similarity",
        ):
            quality_value = float(summary.get(quality_name, -1.0))
            if not 0.0 <= quality_value <= 1.0:
                raise ValueError(f"{summary_path}: invalid {quality_name}")
        rows.append(
            {
                "target_tokens": target_tokens,
                "edit_radius": edit_radius,
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "target_tokens"
                },
            }
        )
    return rows


# Validate all nine conditions and return rows in stable report order.
def analyze_quality_matrix(input_root: Path) -> list[dict[str, Any]]:
    rows = []
    for target_tokens in TARGETS:
        for edit_radius in RADII:
            rows.extend(_load_condition(input_root, target_tokens, edit_radius))
    position_order = {position: index for index, position in enumerate(POSITIONS)}
    return sorted(
        rows,
        key=lambda row: (
            row["target_tokens"],
            row["edit_radius"],
            position_order[row["position"]],
        ),
    )


# Write the complete normalized matrix for plotting and statistical analysis.
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Render the central speed-quality evidence as a report-ready Markdown table.
def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# CacheSelect quality matrix",
        "",
        "Negative TTFT deltas mean active CacheSelect was faster than native vLLM.",
        "",
        "| Tokens | Radius | Position | Reused rows | TTFT delta | Quality | Exact match |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_tokens']} | {row['edit_radius']} | "
            f"{row['position']} | {float(row['mean_reused_rows']):.1f} | "
            f"{float(row['mean_active_vs_native_ttft_ms']):+.3f} ms | "
            f"{float(row['active_quality_pass_rate']):.1%} | "
            f"{float(row['active_shadow_exact_match_rate']):.1%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Parse paths, validate the matrix, and write all consolidated artifacts.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = analyze_quality_matrix(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "quality-matrix.csv", rows)
    (args.output_dir / "quality-matrix.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "quality-matrix.md", rows)
    print(f"Validated {len(rows)} quality-matrix cells")
    print(f"Saved consolidated artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
