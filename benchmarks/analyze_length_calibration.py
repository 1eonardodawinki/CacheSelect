"""Validate and analyse native-vLLM prompt-length calibration artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from observability.request_recorder import validate_ledger


TARGETS = (256, 1024, 4096)
EDIT_POSITIONS = ("early", "middle", "late")
APC_LABELS = ("off", "on")
RESULT_NAME = re.compile(
    r"^tokens-(256|1024|4096)-(early|middle|late)-apc-(off|on)-rep-1\.json$"
)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(materialized)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _condition_key(path: Path) -> tuple[int, str, str]:
    match = RESULT_NAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"unexpected calibration result filename: {path.name}")
    target_text, edit_position, apc = match.groups()
    return int(target_text), edit_position, apc


def _selected_results(
    input_roots: list[Path],
) -> tuple[
    dict[tuple[int, str, str], tuple[Path, Path]],
    list[dict[str, Any]],
    int,
]:
    selected: dict[tuple[int, str, str], tuple[Path, Path]] = {}
    superseded: list[dict[str, Any]] = []
    raw_result_count = 0
    for input_root in input_roots:
        result_paths = sorted(
            path
            for path in (input_root / "results").glob("*.json")
            if not path.name.endswith(".manifest.json")
        )
        raw_result_count += len(result_paths)
        for path in result_paths:
            key = _condition_key(path)
            if key in selected:
                previous_root, previous_path = selected[key]
                superseded.append(
                    {
                        "condition": {
                            "target_prompt_tokens": key[0],
                            "edit_position": key[1],
                            "apc": key[2],
                        },
                        "superseded_root": previous_root.name,
                        "superseded_result": previous_path.name,
                        "selected_root": input_root.name,
                        "selected_result": path.name,
                    }
                )
            selected[key] = (input_root, path)

    expected = {
        (target, edit_position, apc)
        for target in TARGETS
        for edit_position in EDIT_POSITIONS
        for apc in APC_LABELS
    }
    discovered = set(selected)
    if discovered != expected:
        raise ValueError(
            "calibration matrix is incomplete: "
            f"missing={sorted(expected - discovered)}, "
            f"extra={sorted(discovered - expected)}"
        )
    return selected, superseded, raw_result_count


def _load_calibration(
    input_roots: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, superseded, raw_result_count = _selected_results(input_roots)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    position_order = {position: index for index, position in enumerate(EDIT_POSITIONS)}
    selected_items = sorted(
        selected.items(),
        key=lambda item: (
            item[0][0],
            position_order[item[0][1]],
            item[0][2],
        ),
    )

    for key, (input_root, path) in selected_items:
        target, edit_position, apc = key
        result = json.loads(path.read_text(encoding="utf-8"))
        observations = result.get("observations") or []
        if len(observations) != 2:
            raise ValueError(f"{path}: expected two observations")
        if result.get("request_ledger_summary") != {
            "started": 2,
            "completed": 2,
            "failed": 0,
        }:
            raise ValueError(f"{path}: incomplete result ledger summary")

        manifest_path = path.with_name(path.name.replace(".json", ".manifest.json"))
        if not manifest_path.exists():
            raise ValueError(f"missing manifest for {path.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_key = (
            int(manifest["target_prompt_tokens"]),
            manifest["edit_position"],
            manifest["apc"],
        )
        if manifest_key != key or int(manifest["repetition"]) != 1:
            raise ValueError(f"manifest/result mismatch for {path.name}")
        manifests.append(manifest)

        job_id = str(manifest["calibration_job_id"])
        ledger_path = (
            input_root
            / "request-logs"
            / f"length-calibration-{job_id}-{path.stem}.jsonl"
        )
        if not ledger_path.exists():
            raise ValueError(f"missing request ledger for {path.name}")
        ledger = validate_ledger(ledger_path)
        if not ledger.is_complete or (
            ledger.started,
            ledger.completed,
            ledger.failed,
        ) != (
            2,
            2,
            0,
        ):
            raise ValueError(f"invalid request ledger for {path.name}: {ledger}")

        for observation in observations:
            sequence_index = int(observation["sequence_index"])
            if sequence_index not in {0, 1}:
                raise ValueError(f"unexpected sequence index in {path}")
            prompt_tokens = int(observation["prompt_token_count"])
            if prompt_tokens != target:
                raise ValueError(
                    f"{path}: target={target}, observed prompt={prompt_tokens}"
                )
            cached_tokens = int(observation["cached_tokens"])
            raw_response = observation.get("raw_response") or {}
            choices = raw_response.get("choices") or []
            finish_reason = choices[0].get("finish_reason") if choices else None
            quality = observation.get("quality") or {}
            metrics = observation.get("server_metrics") or {}
            rows.append(
                {
                    "target_prompt_tokens": target,
                    "edit_position": edit_position,
                    "apc": apc,
                    "request_role": "base" if sequence_index == 0 else "edited",
                    "request_id": observation["request_id"],
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "cache_hit_fraction": cached_tokens / prompt_tokens,
                    "ttft_ms": float(metrics["time_to_first_token_ms"]),
                    "quality_passed": bool(quality["passed"]),
                    "quality_score": float(quality["score"]),
                    "finish_reason": finish_reason,
                    "prompt_token_ids": observation["prompt_token_ids"],
                    "result_file": path.name,
                    "ledger_file": ledger_path.name,
                    "source_root": input_root.name,
                    "calibration_job_id": job_id,
                }
            )

    if any(row["cached_tokens"] != 0 for row in rows if row["apc"] == "off"):
        raise ValueError("APC-off calibration contains cached tokens")
    if any(
        row["cached_tokens"] != 0
        for row in rows
        if row["apc"] == "on" and row["request_role"] == "base"
    ):
        raise ValueError("cold APC-on calibration request contains cached tokens")
    if any(
        row["cached_tokens"] <= 0
        for row in rows
        if row["apc"] == "on" and row["request_role"] == "edited"
    ):
        raise ValueError("warm APC-on calibration request has no cache hit")
    if any(row["finish_reason"] != "stop" for row in rows):
        raise ValueError("calibration contains a non-stop finish reason")
    if any(not row["quality_passed"] for row in rows):
        raise ValueError("selected calibration contains a failed quality check")

    provenance = {
        "calibration_job_ids": sorted(
            {str(item["calibration_job_id"]) for item in manifests}
        ),
        "models": sorted({item["model"] for item in manifests}),
        "project_commits": sorted({item["project_commit"] for item in manifests}),
        "vllm_versions": sorted({item["vllm_version"] for item in manifests}),
        "gpus": sorted({item["gpu"] for item in manifests}),
        "hosts": sorted({item["host"] for item in manifests}),
        "input_roots": [root.name for root in input_roots],
        "superseded_conditions": superseded,
        "raw_result_count": raw_result_count,
        "selected_result_count": len(selected),
        "request_count": len(rows),
        "ledger_count": len(selected),
    }
    return rows, provenance


def _pair_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["target_prompt_tokens"],
            row["edit_position"],
            row["request_role"],
        )
        grouped[key][row["apc"]] = row

    paired = []
    for (target, edit_position, request_role), modes in grouped.items():
        if set(modes) != {"off", "on"}:
            raise ValueError(f"unpaired calibration modes: {target}/{edit_position}")
        off = modes["off"]
        on = modes["on"]
        if off["prompt_token_ids"] != on["prompt_token_ids"]:
            raise ValueError(
                "APC mode changed the prompt tokens for "
                f"{target}/{edit_position}/{request_role}"
            )
        paired.append(
            {
                "target_prompt_tokens": target,
                "edit_position": edit_position,
                "request_role": request_role,
                "cached_tokens_on": on["cached_tokens"],
                "cache_hit_fraction_on": on["cache_hit_fraction"],
                "ttft_off_ms": off["ttft_ms"],
                "ttft_on_ms": on["ttft_ms"],
                "ttft_delta_ms": off["ttft_ms"] - on["ttft_ms"],
                "ttft_reduction_fraction": (off["ttft_ms"] - on["ttft_ms"])
                / off["ttft_ms"],
                "ttft_speedup": off["ttft_ms"] / on["ttft_ms"],
                "quality_passed_off": off["quality_passed"],
                "quality_passed_on": on["quality_passed"],
                "off_result_file": off["result_file"],
                "on_result_file": on["result_file"],
            }
        )

    position_order = {position: index for index, position in enumerate(EDIT_POSITIONS)}
    return sorted(
        paired,
        key=lambda row: (
            row["target_prompt_tokens"],
            position_order[row["edit_position"]],
            row["request_role"],
        ),
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_summary(
    path: Path,
    provenance: dict[str, Any],
    paired_rows: list[dict[str, Any]],
) -> None:
    edited = [row for row in paired_rows if row["request_role"] == "edited"]
    table = [
        [
            str(row["target_prompt_tokens"]),
            row["edit_position"],
            f"{row['cached_tokens_on']}/{row['target_prompt_tokens']}",
            f"{100 * row['cache_hit_fraction_on']:.1f}%",
            f"{row['ttft_off_ms']:.2f}",
            f"{row['ttft_on_ms']:.2f}",
            f"{100 * row['ttft_reduction_fraction']:.1f}%",
            f"{row['ttft_speedup']:.2f}×",
        ]
        for row in edited
    ]
    cold = [row for row in paired_rows if row["request_role"] == "base"]
    cold_table = []
    for target in TARGETS:
        target_rows = [row for row in cold if row["target_prompt_tokens"] == target]
        off = _mean(row["ttft_off_ms"] for row in target_rows)
        on = _mean(row["ttft_on_ms"] for row in target_rows)
        cold_table.append(
            [str(target), f"{off:.2f}", f"{on:.2f}", f"{100 * (off - on) / off:.1f}%"]
        )

    content = f"""# Native-vLLM prompt-length calibration

## Validation

- Jobs: {", ".join(provenance["calibration_job_ids"])}.
- {provenance["selected_result_count"]} selected conditions, {provenance["ledger_count"]} complete request ledgers, and {provenance["request_count"]} requests validated.
- All selected prompts matched their 256, 1,024, or 4,096 token target exactly.
- APC-off and cold APC-on requests reused zero tokens; every edited APC-on request reused at least one complete cache block.
- All selected outputs finished naturally and passed semantic fact retrieval.
- A failed 4K condition from job 269226 was superseded by the equivalent corrected condition from job 269267; the API prompt itself was unchanged.

## Edited-request results

{_markdown_table(["Prompt tokens", "Edit", "Cached/prompt", "Hit rate", "APC off TTFT", "APC on TTFT", "Reduction", "Speedup"], table)}

## Cold-request control

Each value is the mean across the three edit-position conditions at that length.

{_markdown_table(["Prompt tokens", "APC off TTFT", "APC on TTFT", "Difference"], cold_table)}

## Interpretation

- Native prefix caching reuses only the exact token prefix before the changed marker. Early edits therefore reuse only 32 tokens at every tested length.
- Moving the edit later increases the reusable prefix: middle edits reuse approximately one third and late edits approximately two thirds of each prompt.
- TTFT savings grow with both prompt length and the number of cached tokens. This confirms that the final CacheSelect evaluation must include graduated context lengths rather than extrapolating from the initial 100–230-token traces.
- The early 256-token condition shows that a small cache hit can be slower than recomputation because lookup and block-management overhead can exceed saved prefill work.
- This is a one-repetition calibration, not the final performance result. It establishes suitable scales and validates the measurement pipeline; the final matrix needs repeated, order-randomized comparisons.

Generated files: `request_metrics.csv`, `paired_request_comparison.csv`, the plots in this directory, and `provenance.json`.
"""
    path.write_text(content, encoding="utf-8")


def _write_plots(output_dir: Path, paired_rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt

    edited = [row for row in paired_rows if row["request_role"] == "edited"]
    colors = {"early": "#D55E00", "middle": "#0072B2", "late": "#009E73"}

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for position in EDIT_POSITIONS:
        rows = [row for row in edited if row["edit_position"] == position]
        axis.plot(
            [row["target_prompt_tokens"] for row in rows],
            [100 * row["ttft_reduction_fraction"] for row in rows],
            marker="o",
            linewidth=2,
            label=f"{position.title()} edit",
            color=colors[position],
        )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(TARGETS, ["256", "1K", "4K"])
    axis.set_xlabel("Rendered prompt tokens")
    axis.set_ylabel("TTFT reduction with native APC (%)")
    axis.set_title("Longer reusable prefixes produce larger TTFT savings")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"ttft_reduction_by_length.{suffix}")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    width = 0.22
    x = list(range(len(TARGETS)))
    for offset, position in zip((-1, 0, 1), EDIT_POSITIONS, strict=True):
        rows = [row for row in edited if row["edit_position"] == position]
        axis.bar(
            [value + offset * width for value in x],
            [100 * row["cache_hit_fraction_on"] for row in rows],
            width,
            label=f"{position.title()} edit",
            color=colors[position],
        )
    axis.set_xticks(x, ["256", "1K", "4K"])
    axis.set_xlabel("Rendered prompt tokens")
    axis.set_ylabel("APC-on cached prompt tokens (%)")
    axis.set_ylim(0, 75)
    axis.set_title("Edit position determines the reusable exact prefix")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cache_hit_by_length.{suffix}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "Downloaded calibration artifact root. Repeat to combine jobs; "
            "later roots supersede duplicate target/edit/APC conditions."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    input_roots = [root.resolve() for root in args.input_root]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, provenance = _load_calibration(input_roots)
    paired_rows = _pair_modes(rows)

    csv_rows = [
        {key: value for key, value in row.items() if key != "prompt_token_ids"}
        for row in rows
    ]
    _write_csv(output_dir / "request_metrics.csv", csv_rows)
    _write_csv(output_dir / "paired_request_comparison.csv", paired_rows)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary(output_dir / "summary.md", provenance, paired_rows)
    if not args.no_plots:
        _write_plots(output_dir, paired_rows)

    print(f"Validated {provenance['selected_result_count']} selected conditions")
    print(f"Validated {provenance['ledger_count']} complete request ledgers")
    print(f"Aggregated {provenance['request_count']} requests")
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
