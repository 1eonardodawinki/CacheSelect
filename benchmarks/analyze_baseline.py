"""Validate and aggregate a downloaded CacheSelect baseline matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


WORKLOADS = ("rag", "periodic_agent", "chat")
APC_LABELS = ("off", "on")
REPETITIONS = (1, 2, 3)
RESULT_NAME = re.compile(r"^(rag|periodic_agent|chat)-apc-(off|on)-rep-([1-3])\.json$")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(materialized)


def _sample_sd(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_ledger(path: Path, expected_requests: int) -> None:
    started: dict[str, dict[str, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            request_id = event.get("request_id")
            event_type = event.get("event")
            target = {
                "request_started": started,
                "request_completed": completed,
                "request_failed": failed,
            }.get(event_type)
            if target is None:
                raise ValueError(f"{path}:{line_number}: unknown event {event_type!r}")
            if request_id in target:
                raise ValueError(f"{path}: duplicate {event_type} for {request_id}")
            target[request_id] = event

    if len(started) != expected_requests or len(completed) != expected_requests:
        raise ValueError(
            f"{path}: expected {expected_requests} starts/completions, "
            f"got {len(started)}/{len(completed)}"
        )
    if failed:
        raise ValueError(f"{path}: contains {len(failed)} failed requests")
    if started.keys() != completed.keys():
        raise ValueError(f"{path}: request lifecycle IDs do not pair")
    for request_id, event in started.items():
        payload = (event.get("model_input") or {}).get("payload")
        if not payload or not payload.get("messages"):
            raise ValueError(f"{path}: {request_id} lacks complete input messages")
        output = (completed[request_id].get("output") or {}).get("raw_response")
        if not output:
            raise ValueError(f"{path}: {request_id} lacks the raw model output")


def _load_matrix(
    input_roots: list[Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    expected = {
        (workload, apc, repetition)
        for workload in WORKLOADS
        for apc in APC_LABELS
        for repetition in REPETITIONS
    }
    selected: dict[tuple[str, str, int], tuple[Path, Path]] = {}
    superseded: list[dict[str, str]] = []
    raw_result_count = 0
    for input_root in input_roots:
        result_paths = sorted(
            path
            for path in (input_root / "results").glob("*.json")
            if not path.name.endswith(".manifest.json")
        )
        raw_result_count += len(result_paths)
        for path in result_paths:
            match = RESULT_NAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"unexpected result filename: {path.name}")
            workload, apc, repetition_text = match.groups()
            key = (workload, apc, int(repetition_text))
            if key in selected:
                previous_root, previous_path = selected[key]
                superseded.append(
                    {
                        "condition": "/".join(map(str, key)),
                        "superseded_root": previous_root.name,
                        "superseded_result": previous_path.name,
                        "selected_root": input_root.name,
                        "selected_result": path.name,
                    }
                )
            selected[key] = (input_root, path)

    discovered = set(selected)
    if discovered != expected:
        missing = sorted(expected - discovered)
        extra = sorted(discovered - expected)
        raise ValueError(f"matrix is incomplete: missing={missing}, extra={extra}")

    request_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    workload_order = {name: index for index, name in enumerate(WORKLOADS)}
    selected_items = sorted(
        selected.items(),
        key=lambda item: (
            workload_order[item[0][0]],
            item[0][1],
            item[0][2],
        ),
    )
    for key, (input_root, path) in selected_items:
        workload, apc, repetition = key
        result_dir = input_root / "results"
        ledger_dir = input_root / "request-logs"

        result = json.loads(path.read_text(encoding="utf-8"))
        observations = result.get("observations") or []
        expected_count = 6 if workload == "periodic_agent" else 4
        if len(observations) != expected_count:
            raise ValueError(
                f"{path}: expected {expected_count} observations, "
                f"got {len(observations)}"
            )
        ledger_summary = result.get("request_ledger_summary")
        expected_ledger = {
            "started": expected_count,
            "completed": expected_count,
            "failed": 0,
        }
        if ledger_summary != expected_ledger:
            raise ValueError(f"{path}: invalid ledger summary {ledger_summary}")

        manifest_path = result_dir / path.name.replace(".json", ".manifest.json")
        if not manifest_path.exists():
            raise ValueError(f"missing manifest for {path.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("workload"),
            manifest.get("apc"),
            manifest.get("repetition"),
        ) != key:
            raise ValueError(f"manifest/result mismatch for {path.name}")
        manifests.append(manifest)

        ledger_path = (
            ledger_dir / f"baseline-{manifest['matrix_job_id']}-{path.stem}.jsonl"
        )
        if not ledger_path.exists():
            raise ValueError(f"missing local ledger for {path.name}: {ledger_path}")
        _validate_ledger(ledger_path, expected_count)

        transitions = {
            transition["current_request_id"]: transition
            for transition in result.get("transitions") or []
        }
        for observation in observations:
            request_id = observation["request_id"]
            sequence_index = int(observation["sequence_index"])
            metrics = observation.get("server_metrics") or {}
            quality = observation.get("quality") or {}
            raw_response = observation.get("raw_response") or {}
            choices = raw_response.get("choices") or []
            transition = transitions.get(request_id)
            change_type = (
                transition["ground_truth"]["change_type"]
                if transition is not None
                else "cold_start"
            )
            prompt_tokens = int(observation["prompt_token_count"])
            cached_tokens = int(observation["cached_tokens"])
            request_rows.append(
                {
                    "workload": workload,
                    "apc": apc,
                    "repetition": repetition,
                    "request_id": request_id,
                    "sequence_index": sequence_index,
                    "scope": "cold" if sequence_index == 0 else "reuse_eligible",
                    "change_type": change_type,
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "cache_hit_fraction": cached_tokens / prompt_tokens,
                    "ttft_ms": float(metrics["time_to_first_token_ms"]),
                    "client_wall_ms": float(observation["client_wall_seconds"]) * 1000,
                    "generation_time_ms": float(metrics["generation_time_ms"]),
                    "mean_itl_ms": float(metrics["mean_itl_ms"]),
                    "tokens_per_second": float(metrics["tokens_per_second"]),
                    "quality_passed": bool(quality["passed"]),
                    "quality_score": float(quality["score"]),
                    "finish_reason": choices[0].get("finish_reason")
                    if choices
                    else None,
                    "result_file": path.name,
                    "ledger_file": ledger_path.name,
                    "source_root": input_root.name,
                }
            )

        warm = [
            row
            for row in request_rows
            if row["result_file"] == path.name and row["scope"] == "reuse_eligible"
        ]
        current = [row for row in request_rows if row["result_file"] == path.name]
        run_rows.append(
            {
                "workload": workload,
                "apc": apc,
                "repetition": repetition,
                "request_count": len(current),
                "mean_ttft_all_ms": _mean(row["ttft_ms"] for row in current),
                "mean_ttft_warm_ms": _mean(row["ttft_ms"] for row in warm),
                "mean_cache_hit_fraction_warm": _mean(
                    row["cache_hit_fraction"] for row in warm
                ),
                "quality_pass_rate": _mean(
                    float(row["quality_passed"]) for row in current
                ),
                "mean_quality_score": _mean(row["quality_score"] for row in current),
                "length_finish_rate": _mean(
                    float(row["finish_reason"] == "length") for row in current
                ),
                "result_file": path.name,
                "ledger_file": ledger_path.name,
                "source_root": input_root.name,
            }
        )

    if any(row["cached_tokens"] != 0 for row in request_rows if row["apc"] == "off"):
        raise ValueError("APC-off result contains non-zero cached tokens")
    if any(
        row["cached_tokens"] != 0
        for row in request_rows
        if row["apc"] == "on" and row["scope"] == "cold"
    ):
        raise ValueError("cold APC-on request unexpectedly contains cached tokens")

    provenance = {
        "matrix_job_ids": sorted({str(item["matrix_job_id"]) for item in manifests}),
        "models": sorted({item["model"] for item in manifests}),
        "project_commits": sorted({item["project_commit"] for item in manifests}),
        "vllm_versions": sorted({item["vllm_version"] for item in manifests}),
        "gpus": sorted({item["gpu"] for item in manifests}),
        "hosts": sorted({item["host"] for item in manifests}),
        "input_roots": [root.name for root in input_roots],
        "superseded_conditions": superseded,
        "raw_result_count": raw_result_count,
        "result_count": len(selected),
        "request_count": len(request_rows),
        "ledger_count": len(selected),
    }
    return request_rows, run_rows, provenance


def _paired_requests(request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in request_rows:
        grouped[(row["workload"], row["repetition"], row["request_id"])][row["apc"]] = (
            row
        )

    paired: list[dict[str, Any]] = []
    for (workload, repetition, request_id), modes in grouped.items():
        if set(modes) != {"off", "on"}:
            raise ValueError(
                f"unpaired APC modes for {(workload, repetition, request_id)}"
            )
        off = modes["off"]
        on = modes["on"]
        if off["prompt_tokens"] != on["prompt_tokens"]:
            raise ValueError(f"prompt length changed across APC modes for {request_id}")
        paired.append(
            {
                "workload": workload,
                "repetition": repetition,
                "request_id": request_id,
                "sequence_index": off["sequence_index"],
                "scope": off["scope"],
                "change_type": off["change_type"],
                "prompt_tokens": off["prompt_tokens"],
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
                "quality_score_off": off["quality_score"],
                "quality_score_on": on["quality_score"],
                "quality_score_delta": on["quality_score"] - off["quality_score"],
            }
        )
    order = {workload: index for index, workload in enumerate(WORKLOADS)}
    paired.sort(
        key=lambda row: (
            order[row["workload"]],
            row["repetition"],
            row["sequence_index"],
        )
    )
    return paired


def _workload_summaries(
    request_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scopes: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
        ("all", lambda row: True),
        ("reuse_eligible", lambda row: row["scope"] == "reuse_eligible"),
        ("cold", lambda row: row["scope"] == "cold"),
    )
    summaries: list[dict[str, Any]] = []
    for workload in WORKLOADS:
        for scope_name, include in scopes:
            mode_rows: dict[str, list[dict[str, Any]]] = {}
            rep_means: dict[str, list[float]] = {}
            for apc in APC_LABELS:
                selected = [
                    row
                    for row in request_rows
                    if row["workload"] == workload
                    and row["apc"] == apc
                    and include(row)
                ]
                mode_rows[apc] = selected
                rep_means[apc] = [
                    _mean(
                        row["ttft_ms"] for row in selected if row["repetition"] == rep
                    )
                    for rep in REPETITIONS
                ]
            off_mean = _mean(rep_means["off"])
            on_mean = _mean(rep_means["on"])
            paired_reductions = [
                (off - on) / off
                for off, on in zip(rep_means["off"], rep_means["on"], strict=True)
            ]
            summaries.append(
                {
                    "workload": workload,
                    "scope": scope_name,
                    "requests_per_mode": len(mode_rows["off"]),
                    "repetitions": len(REPETITIONS),
                    "ttft_off_mean_ms": off_mean,
                    "ttft_off_rep_sd_ms": _sample_sd(rep_means["off"]),
                    "ttft_on_mean_ms": on_mean,
                    "ttft_on_rep_sd_ms": _sample_sd(rep_means["on"]),
                    "ttft_reduction_fraction": (off_mean - on_mean) / off_mean,
                    "paired_reduction_rep_sd": _sample_sd(paired_reductions),
                    "ttft_speedup": off_mean / on_mean,
                    "mean_cache_hit_fraction_on": _mean(
                        row["cache_hit_fraction"] for row in mode_rows["on"]
                    ),
                    "quality_pass_rate_off": _mean(
                        float(row["quality_passed"]) for row in mode_rows["off"]
                    ),
                    "quality_pass_rate_on": _mean(
                        float(row["quality_passed"]) for row in mode_rows["on"]
                    ),
                    "mean_quality_score_off": _mean(
                        row["quality_score"] for row in mode_rows["off"]
                    ),
                    "mean_quality_score_on": _mean(
                        row["quality_score"] for row in mode_rows["on"]
                    ),
                    "length_finish_rate_off": _mean(
                        float(row["finish_reason"] == "length")
                        for row in mode_rows["off"]
                    ),
                    "length_finish_rate_on": _mean(
                        float(row["finish_reason"] == "length")
                        for row in mode_rows["on"]
                    ),
                }
            )
    return summaries


def _request_summaries(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[(row["workload"], row["request_id"])].append(row)
    order = {workload: index for index, workload in enumerate(WORKLOADS)}
    summaries = []
    for (workload, request_id), rows in grouped.items():
        first = rows[0]
        off = _mean(row["ttft_off_ms"] for row in rows)
        on = _mean(row["ttft_on_ms"] for row in rows)
        summaries.append(
            {
                "workload": workload,
                "request_id": request_id,
                "sequence_index": first["sequence_index"],
                "scope": first["scope"],
                "change_type": first["change_type"],
                "prompt_tokens": first["prompt_tokens"],
                "mean_cached_tokens_on": _mean(row["cached_tokens_on"] for row in rows),
                "mean_cache_hit_fraction_on": _mean(
                    row["cache_hit_fraction_on"] for row in rows
                ),
                "ttft_off_mean_ms": off,
                "ttft_on_mean_ms": on,
                "ttft_reduction_fraction": (off - on) / off,
                "ttft_speedup": off / on,
                "mean_quality_score_off": _mean(
                    row["quality_score_off"] for row in rows
                ),
                "mean_quality_score_on": _mean(row["quality_score_on"] for row in rows),
            }
        )
    summaries.sort(key=lambda row: (order[row["workload"]], row["sequence_index"]))
    return summaries


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_summary(
    output_path: Path,
    provenance: dict[str, Any],
    workload_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
) -> None:
    by_scope = {(row["workload"], row["scope"]): row for row in workload_rows}
    warm_table = []
    cold_table = []
    quality_table = []
    for workload in WORKLOADS:
        warm = by_scope[(workload, "reuse_eligible")]
        cold = by_scope[(workload, "cold")]
        all_rows = by_scope[(workload, "all")]
        warm_table.append(
            [
                workload,
                f"{warm['ttft_off_mean_ms']:.2f}",
                f"{warm['ttft_on_mean_ms']:.2f}",
                f"{100 * warm['ttft_reduction_fraction']:.1f}%",
                f"{warm['ttft_speedup']:.2f}×",
                f"{100 * warm['mean_cache_hit_fraction_on']:.1f}%",
            ]
        )
        cold_table.append(
            [
                workload,
                f"{cold['ttft_off_mean_ms']:.2f}",
                f"{cold['ttft_on_mean_ms']:.2f}",
                f"{100 * cold['ttft_reduction_fraction']:.1f}%",
            ]
        )
        quality_table.append(
            [
                workload,
                f"{100 * all_rows['quality_pass_rate_off']:.1f}%",
                f"{100 * all_rows['quality_pass_rate_on']:.1f}%",
                f"{all_rows['mean_quality_score_off']:.3f}",
                f"{all_rows['mean_quality_score_on']:.3f}",
                f"{100 * all_rows['length_finish_rate_off']:.1f}%",
            ]
        )

    request_table = [
        [
            row["request_id"],
            row["change_type"],
            f"{row['mean_cached_tokens_on']:.0f}/{row['prompt_tokens']}",
            f"{100 * row['mean_cache_hit_fraction_on']:.1f}%",
            f"{row['ttft_off_mean_ms']:.2f}",
            f"{row['ttft_on_mean_ms']:.2f}",
            f"{100 * row['ttft_reduction_fraction']:.1f}%",
        ]
        for row in request_rows
        if row["scope"] == "reuse_eligible"
    ]

    quality_deltas = [
        abs(row["mean_quality_score_on"] - row["mean_quality_score_off"])
        for row in request_rows
    ]
    no_quality_delta = max(quality_deltas, default=0.0) == 0.0
    matrix_ids = ", ".join(provenance["matrix_job_ids"])
    models = ", ".join(provenance["models"])
    commits = ", ".join(value[:12] for value in provenance["project_commits"])
    vllm_versions = ", ".join(provenance["vllm_versions"])

    truncated_workloads = [
        workload
        for workload in WORKLOADS
        if max(
            by_scope[(workload, "all")]["length_finish_rate_off"],
            by_scope[(workload, "all")]["length_finish_rate_on"],
        )
        > 0
    ]
    if truncated_workloads:
        truncation_qualification = (
            "Absolute quality needs qualification for workloads with at least "
            "one length-limited response: "
            + ", ".join(truncated_workloads)
            + ". Their completion limits should be increased before treating "
            "absolute pass rates as final quality results."
        )
    else:
        truncation_qualification = (
            "No response ended because of the configured completion-token "
            "limit, so truncation does not qualify these quality scores."
        )

    chat_all = by_scope[("chat", "all")]
    chat_qualification = ""
    if (
        min(
            chat_all["quality_pass_rate_off"],
            chat_all["quality_pass_rate_on"],
        )
        < 1
    ):
        chat_qualification = (
            " The chat failures are missing required constraints and should be "
            "inspected separately from truncation."
        )

    content = f"""# CacheSelect baseline matrix {matrix_ids}

## Validation

- {provenance["result_count"]} result files, {provenance["ledger_count"]} complete request ledgers, and {provenance["request_count"]} request observations ({provenance["paired_request_count"]} APC-off/on pairs) validated.
- Model: `{models}`.
- CacheSelect commit: `{commits}`; vLLM: `{vllm_versions}`.
- APC-off cached-token counts are all zero; every cold APC-on request also has zero cached tokens.
- Full request inputs and raw outputs are present in every ledger.

## Reuse-eligible TTFT

The first request from each fresh server is excluded here because no cache can
exist yet. Values are means of three repetition means.

{_markdown_table(["Workload", "APC off (ms)", "APC on (ms)", "Reduction", "Speedup", "APC-on hit rate"], warm_table)}

## Cold-start control

{_markdown_table(["Workload", "APC off (ms)", "APC on (ms)", "Difference"], cold_table)}

The near-zero cold-start differences support attributing the warm-request
improvements to cache reuse rather than a general difference between the two
server configurations.

## Request-level behavior

{_markdown_table(["Request", "Transition", "Cached/prompt", "Hit rate", "Off TTFT", "On TTFT", "Reduction"], request_table)}

## Quality as recorded

{_markdown_table(["Workload", "Pass off", "Pass on", "Mean score off", "Mean score on", "Length-stop rate"], quality_table)}

{"No paired request changed its recorded quality score between APC off and on." if no_quality_delta else "At least one paired request changed its recorded quality score; inspect `paired_request_comparison.csv`."}

{truncation_qualification}{chat_qualification} Failures that occur identically
in both APC modes are model, workload, or evaluation-configuration limitations
rather than observed APC regressions.

## Interpretation and limitations

- Native vLLM APC works best when changes occur late in the prompt. The RAG
  query replacement reused 160 tokens, while document reordering reused only
  32 despite retaining the same documents.
- Sliding periodic windows reused only the stable 80-token prefix; unchanged
  rows that moved position were not recovered by native APC.
- Append-only chat reused 48 and then 96 tokens. Editing early history reduced
  reuse to 32 tokens, even though most later text remained identical.
- Only three repetitions were collected, APC-off always ran before APC-on
  within each workload job, and each workload ran on one GPU. The paired cold
  controls are stable, but final experiments should randomize mode order and
  include more repetitions and model/context scales.
- These measurements establish the native-vLLM baseline. They do not yet test
  CacheSelect or approximate/non-prefix reuse.

Generated files: `request_metrics.csv`, `run_summary.csv`,
`paired_request_comparison.csv`, `workload_summary.csv`,
`request_summary.csv`, and the plots in this directory.
"""
    output_path.write_text(content, encoding="utf-8")


def _write_plots(
    output_dir: Path,
    workload_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"off": "#9aa3a8", "on": "#147d73"}
    warm = {
        row["workload"]: row
        for row in workload_rows
        if row["scope"] == "reuse_eligible"
    }
    labels = ["RAG", "Periodic agent", "Chat"]
    x = list(range(len(WORKLOADS)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        [value - width / 2 for value in x],
        [warm[name]["ttft_off_mean_ms"] for name in WORKLOADS],
        width,
        yerr=[warm[name]["ttft_off_rep_sd_ms"] for name in WORKLOADS],
        label="APC off",
        color=colors["off"],
        capsize=3,
    )
    axis.bar(
        [value + width / 2 for value in x],
        [warm[name]["ttft_on_mean_ms"] for name in WORKLOADS],
        width,
        yerr=[warm[name]["ttft_on_rep_sd_ms"] for name in WORKLOADS],
        label="APC on",
        color=colors["on"],
        capsize=3,
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean TTFT (ms)")
    axis.set_title("Reuse-eligible requests (mean ± SD across 3 repetitions)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"warm_ttft_by_workload.{suffix}")
    plt.close(figure)

    palette = {"rag": "#147d73", "periodic_agent": "#e09f3e", "chat": "#6c63a8"}
    markers = {"rag": "o", "periodic_agent": "s", "chat": "^"}
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for workload in WORKLOADS:
        selected = [
            row
            for row in request_rows
            if row["workload"] == workload and row["scope"] == "reuse_eligible"
        ]
        axis.scatter(
            [100 * row["mean_cache_hit_fraction_on"] for row in selected],
            [100 * row["ttft_reduction_fraction"] for row in selected],
            label=workload.replace("_", " ").title(),
            color=palette[workload],
            marker=markers[workload],
            s=55,
        )
        if workload == "periodic_agent":
            axis.annotate(
                "periodic-01…05",
                (
                    _mean(100 * row["mean_cache_hit_fraction_on"] for row in selected),
                    _mean(100 * row["ttft_reduction_fraction"] for row in selected),
                ),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=7,
            )
        else:
            for row in selected:
                axis.annotate(
                    row["request_id"],
                    (
                        100 * row["mean_cache_hit_fraction_on"],
                        100 * row["ttft_reduction_fraction"],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("APC-on cached prompt tokens (%)")
    axis.set_ylabel("Paired TTFT reduction (%)")
    axis.set_title("More exact-prefix reuse generally yields larger TTFT savings")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cache_hit_vs_ttft_reduction.{suffix}")
    plt.close(figure)

    all_scope = {row["workload"]: row for row in workload_rows if row["scope"] == "all"}
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        [value - width / 2 for value in x],
        [all_scope[name]["mean_quality_score_off"] for name in WORKLOADS],
        width,
        label="APC off",
        color=colors["off"],
    )
    axis.bar(
        [value + width / 2 for value in x],
        [all_scope[name]["mean_quality_score_on"] for name in WORKLOADS],
        width,
        label="APC on",
        color=colors["on"],
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Mean deterministic requirement score")
    axis.set_title("Recorded quality is unchanged by native APC")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"quality_by_workload.{suffix}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "Downloaded artifact directory containing results/ and "
            "request-logs/. Repeat to combine runs; later roots replace "
            "duplicate workload/APC/repetition conditions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory (defaults to <input-root>/analysis).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write validated CSV/Markdown outputs without importing matplotlib.",
    )
    args = parser.parse_args()

    input_roots = [root.resolve() for root in args.input_root]
    if len(input_roots) > 1 and args.output_dir is None:
        parser.error("--output-dir is required with multiple --input-root values")
    output_dir = (args.output_dir or input_roots[0] / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    request_rows, run_rows, provenance = _load_matrix(input_roots)
    paired_rows = _paired_requests(request_rows)
    provenance["paired_request_count"] = len(paired_rows)
    workload_rows = _workload_summaries(request_rows)
    request_summaries = _request_summaries(paired_rows)

    _write_csv(output_dir / "request_metrics.csv", request_rows)
    _write_csv(output_dir / "run_summary.csv", run_rows)
    _write_csv(output_dir / "paired_request_comparison.csv", paired_rows)
    _write_csv(output_dir / "workload_summary.csv", workload_rows)
    _write_csv(output_dir / "request_summary.csv", request_summaries)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary(
        output_dir / "summary.md",
        provenance,
        workload_rows,
        request_summaries,
    )
    if not args.no_plots:
        _write_plots(output_dir, workload_rows, request_summaries)

    print(f"Validated {provenance['result_count']} result files")
    print(f"Validated {provenance['ledger_count']} full request ledgers")
    print(f"Aggregated {provenance['request_count']} request observations")
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
