"""Validate and summarize online CacheSelect locator calibration artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from benchmarks.analyze_reuse_opportunities import analyze_benchmark_result
from observability.request_recorder import validate_ledger


TARGETS = (256, 1024, 4096)
EDIT_POSITIONS = ("early", "middle", "late")
RESULT_NAME = re.compile(
    r"^tokens-(256|1024|4096)-(early|middle|late)-"
    r"cacheselect-rep-1\.json$"
)


def analyze_locator_result(
    result: dict[str, Any],
    *,
    target_prompt_tokens: int,
    edit_position: str,
    block_size: int,
) -> dict[str, Any]:
    """Validate one cold-source/edit result and return its measurement row."""
    observations = result.get("observations") or []
    if len(observations) != 2:
        raise ValueError("expected two observations")
    if result.get("request_ledger_summary") != {
        "started": 2,
        "completed": 2,
        "failed": 0,
    }:
        raise ValueError("incomplete result ledger summary")

    source, edited = observations
    prompt_counts = [
        int(source["prompt_token_count"]),
        int(edited["prompt_token_count"]),
    ]
    if any(abs(count - target_prompt_tokens) > 16 for count in prompt_counts):
        raise ValueError(
            f"prompt counts {prompt_counts} miss target {target_prompt_tokens}"
        )
    if not source["quality"]["passed"] or not edited["quality"]["passed"]:
        raise ValueError("quality check failed")
    if int(source["cached_tokens"]) != 0:
        raise ValueError("cold source request unexpectedly hit the cache")

    native_cached_tokens = int(edited["cached_tokens"])
    runtime = edited.get("runtime_policy") or {}
    if runtime.get("policy") != "VLLM_NATIVE_APC":
        raise ValueError("edited request did not preserve native APC")
    if int(runtime.get("native_cached_tokens", -1)) != native_cached_tokens:
        raise ValueError("runtime decision disagrees with executed APC hit")
    plan = runtime.get("partial_reuse_plan")
    if plan is None:
        raise ValueError("edited request omitted its online partial-reuse plan")
    if int(plan["block_size"]) != block_size:
        raise ValueError("unexpected online block size")

    transitions = result.get("transitions") or []
    if len(transitions) != 1:
        raise ValueError("expected one transition")
    transition = transitions[0]
    if (
        plan["transition_id"] != transition["transition_id"]
        or plan["source_request_id"] != transition["previous_request_id"]
        or plan["target_request_id"] != transition["current_request_id"]
    ):
        raise ValueError("online plan references the wrong transition")

    offline_report = analyze_benchmark_result(result, block_size=block_size)
    opportunity = offline_report["transitions"][0]["opportunity"]
    online_candidate_tokens = int(plan["candidate_token_count"])
    online_candidate_blocks = int(plan["candidate_block_count"])
    resident_candidate_tokens = int(plan["resident_candidate_token_count"])
    resident_candidate_blocks = int(plan["resident_candidate_block_count"])
    expected_aligned_blocks = int(opportunity["whole_source_block_count"])
    if online_candidate_blocks != expected_aligned_blocks:
        raise ValueError(
            "online/offline aligned candidate mismatch: "
            f"online={online_candidate_blocks}, offline={expected_aligned_blocks}"
        )
    if online_candidate_tokens != online_candidate_blocks * block_size:
        raise ValueError("online candidate token count is not block aligned")
    if resident_candidate_tokens != resident_candidate_blocks * block_size:
        raise ValueError("resident candidate token count is not block aligned")
    if resident_candidate_blocks > online_candidate_blocks:
        raise ValueError("resident candidate count exceeds located candidates")
    if any("source_block_id" in candidate for candidate in plan["candidates"]):
        raise ValueError("online plan exposed internal GPU block IDs")

    edited_prompt_tokens = prompt_counts[1]
    native_recomputed_tokens = edited_prompt_tokens - native_cached_tokens
    all_candidate_tokens = int(opportunity["candidate_token_count"])
    repacking_candidate_tokens = (
        int(opportunity["repacking_required_block_count"]) * block_size
    )
    metrics = edited.get("server_metrics") or {}
    return {
        "target_prompt_tokens": target_prompt_tokens,
        "edit_position": edit_position,
        "source_prompt_tokens": prompt_counts[0],
        "edited_prompt_tokens": edited_prompt_tokens,
        "native_cached_tokens": native_cached_tokens,
        "native_recomputed_tokens": native_recomputed_tokens,
        "native_cache_hit_fraction": native_cached_tokens / edited_prompt_tokens,
        "offline_all_candidate_tokens": all_candidate_tokens,
        "offline_aligned_candidate_tokens": expected_aligned_blocks * block_size,
        "offline_repacking_candidate_tokens": repacking_candidate_tokens,
        "online_candidate_tokens": online_candidate_tokens,
        "resident_candidate_tokens": resident_candidate_tokens,
        "online_candidate_share_of_native_recompute": (
            online_candidate_tokens / native_recomputed_tokens
            if native_recomputed_tokens
            else 0.0
        ),
        "resident_candidate_share_of_native_recompute": (
            resident_candidate_tokens / native_recomputed_tokens
            if native_recomputed_tokens
            else 0.0
        ),
        "online_reason": plan["reason"],
        "online_block_mappings": [
            {
                "source_block_index": candidate["source_block_index"],
                "target_block_index": candidate["target_block_index"],
                "source_resident": candidate["source_resident"],
                "requires_repair": candidate["requires_repair"],
            }
            for candidate in plan["candidates"]
        ],
        "edited_ttft_ms": float(metrics["time_to_first_token_ms"]),
        "quality_passed": True,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty locator calibration")

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        native_recomputed = sum(row["native_recomputed_tokens"] for row in selected)
        online = sum(row["online_candidate_tokens"] for row in selected)
        resident = sum(row["resident_candidate_tokens"] for row in selected)
        return {
            "condition_count": len(selected),
            "native_recomputed_tokens": native_recomputed,
            "offline_all_candidate_tokens": sum(
                row["offline_all_candidate_tokens"] for row in selected
            ),
            "online_candidate_tokens": online,
            "resident_candidate_tokens": resident,
            "online_candidate_share_of_native_recompute": (
                online / native_recomputed if native_recomputed else 0.0
            ),
            "resident_candidate_share_of_native_recompute": (
                resident / native_recomputed if native_recomputed else 0.0
            ),
        }

    return {
        "overall": aggregate(rows),
        "by_target_prompt_tokens": {
            str(target): aggregate(
                [row for row in rows if row["target_prompt_tokens"] == target]
            )
            for target in TARGETS
        },
        "by_edit_position": {
            position: aggregate(
                [row for row in rows if row["edit_position"] == position]
            )
            for position in EDIT_POSITIONS
        },
    }


def _condition_key(path: Path) -> tuple[int, str]:
    match = RESULT_NAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"unexpected locator result filename: {path.name}")
    target, edit_position = match.groups()
    return int(target), edit_position


def load_locator_calibration(input_root: Path) -> dict[str, Any]:
    result_dir = input_root / "results"
    selected: dict[tuple[int, str], Path] = {}
    for path in sorted(result_dir.glob("*.json")):
        if path.name.endswith(".manifest.json"):
            continue
        key = _condition_key(path)
        if key in selected:
            raise ValueError(f"duplicate locator condition: {key}")
        selected[key] = path

    expected = {
        (target, edit_position)
        for target in TARGETS
        for edit_position in EDIT_POSITIONS
    }
    if set(selected) != expected:
        raise ValueError(
            "locator matrix is incomplete: "
            f"missing={sorted(expected - set(selected))}, "
            f"extra={sorted(set(selected) - expected)}"
        )

    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for key, path in selected.items():
        target, edit_position = key
        manifest_path = path.with_name(path.name.replace(".json", ".manifest.json"))
        if not manifest_path.exists():
            raise ValueError(f"missing manifest for {path.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("experiment") != "online_locator_length_calibration"
            or int(manifest["target_prompt_tokens"]) != target
            or manifest["edit_position"] != edit_position
            or manifest["planner_mode"] != "vllm"
            or manifest["apc"] != "on"
        ):
            raise ValueError(f"manifest/result mismatch for {path.name}")
        manifests.append(manifest)

        job_id = str(manifest["calibration_job_id"])
        ledger_path = (
            input_root
            / "request-logs"
            / f"locator-calibration-{job_id}-{path.stem}.jsonl"
        )
        if not ledger_path.exists():
            raise ValueError(f"missing request ledger for {path.name}")
        ledger = validate_ledger(ledger_path)
        if not ledger.is_complete or (
            ledger.started,
            ledger.completed,
            ledger.failed,
        ) != (2, 2, 0):
            raise ValueError(f"invalid request ledger for {path.name}: {ledger}")

        result = json.loads(path.read_text(encoding="utf-8"))
        row = analyze_locator_result(
            result,
            target_prompt_tokens=target,
            edit_position=edit_position,
            block_size=int(manifest["block_size"]),
        )
        row.update(
            {
                "result_file": path.name,
                "ledger_file": ledger_path.name,
                "calibration_job_id": job_id,
            }
        )
        rows.append(row)

    position_order = {position: index for index, position in enumerate(EDIT_POSITIONS)}
    rows.sort(
        key=lambda row: (
            row["target_prompt_tokens"],
            position_order[row["edit_position"]],
        )
    )
    provenance = {
        "calibration_job_ids": sorted(
            {str(manifest["calibration_job_id"]) for manifest in manifests}
        ),
        "project_commits": sorted(
            {manifest["project_commit"] for manifest in manifests}
        ),
        "models": sorted({manifest["model"] for manifest in manifests}),
        "vllm_versions": sorted({manifest["vllm_version"] for manifest in manifests}),
        "gpus": sorted({manifest["gpu"] for manifest in manifests}),
        "hosts": sorted({manifest["host"] for manifest in manifests}),
        "condition_count": len(rows),
        "ledger_count": len(rows),
    }
    return {
        "schema_version": 1,
        "experiment": "online_locator_length_calibration",
        "provenance": provenance,
        "summary": summarize_rows(rows),
        "conditions": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["online_block_mappings"] = json.dumps(flat["online_block_mappings"])
        flat_rows.append(flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Online locator calibration",
        "",
        "| Prompt | Edit | APC hit | Native recompute | Online candidates | "
        "Resident | Candidate share |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["conditions"]:
        lines.append(
            "| "
            f"{row['target_prompt_tokens']:,} | {row['edit_position']} | "
            f"{row['native_cached_tokens']:,} | "
            f"{row['native_recomputed_tokens']:,} | "
            f"{row['online_candidate_tokens']:,} | "
            f"{row['resident_candidate_tokens']:,} | "
            f"{row['online_candidate_share_of_native_recompute']:.1%} |"
        )
    overall = report["summary"]["overall"]
    lines.extend(
        [
            "",
            "All reported candidates remain shadow-only and require KV repair.",
            "The aggregate online candidate share of native recomputation is "
            f"{overall['online_candidate_share_of_native_recompute']:.1%}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = load_locator_calibration(args.input_root)
    (output_dir / "locator-calibration.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "locator-conditions.csv", report["conditions"])
    _write_markdown(output_dir / "summary.md", report)

    overall = report["summary"]["overall"]
    print(
        f"Validated {overall['condition_count']} conditions: "
        f"online_candidates={overall['online_candidate_tokens']} "
        f"resident={overall['resident_candidate_tokens']} "
        "candidate_share="
        f"{overall['online_candidate_share_of_native_recompute']:.1%}"
    )
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
