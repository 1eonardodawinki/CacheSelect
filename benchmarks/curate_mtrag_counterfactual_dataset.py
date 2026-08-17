"""Merge automatic and reviewed MTRAG trials into one training dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.consolidate_mtrag_counterfactual import AUDIT_COLUMNS, TRAINING_COLUMNS
from benchmarks.prepare_mtrag_counterfactual_review import _load_ledger
from cacheselect.block_features import extract_candidate_block_features
from cacheselect.reuse_opportunity import analyze_reuse_opportunity


REVIEW_COLUMNS = (
    "trial_id",
    "adjudication",
    "review_id",
    "review_verdict",
    "review_reason",
)


# Read recorded prompt tokens from one completed request lifecycle event.
def _prompt_tokens(event: dict[str, Any], role: str) -> tuple[int, ...]:
    raw = (event.get("output") or {}).get("raw_response") or {}
    tokens = raw.get("prompt_token_ids")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(
            not isinstance(token, int) or isinstance(token, bool) for token in tokens
        )
    ):
        raise ValueError(f"{role} request has invalid prompt token IDs")
    return tuple(tokens)


# Reconstruct the selected block's cheap features from its donor and intervention.
def _reviewed_feature(
    audit_row: dict[str, Any],
    started: dict[str, dict],
    completed: dict[str, dict],
) -> tuple[dict[str, Any], str]:
    trial_id = audit_row.get("trial_id")
    roles = {
        event.get("metadata", {}).get("counterfactual_role"): event
        for event in started.values()
        if event.get("metadata", {}).get("counterfactual_trial_id") == trial_id
    }
    if "donor" not in roles or "intervention" not in roles:
        raise ValueError(f"trial {trial_id!r} lacks donor or intervention evidence")
    donor = completed[roles["donor"]["request_id"]]
    intervention = completed[roles["intervention"]["request_id"]]
    metrics = intervention.get("metrics", {}).get("server_metrics", {})
    plan = metrics.get("cacheselect_partial_reuse_plan")
    if not isinstance(plan, dict):
        raise ValueError(f"trial {trial_id!r} has no partial-reuse plan")
    block_size = plan.get("block_size")
    native_tokens = plan.get("native_cached_tokens")
    transition_id = plan.get("transition_id")
    if not isinstance(block_size, int) or not isinstance(native_tokens, int):
        raise ValueError(f"trial {trial_id!r} has invalid cache geometry")
    previous = _prompt_tokens(donor, "donor")
    current = _prompt_tokens(intervention, "intervention")
    opportunity = analyze_reuse_opportunity(
        previous,
        current,
        native_cached_tokens=native_tokens,
        block_size=block_size,
    )
    target = audit_row.get("block_index")
    matches = [
        feature
        for feature in extract_candidate_block_features(previous, current, opportunity)
        if feature.candidate_block_index == target and not feature.requires_repacking
    ]
    if len(matches) != 1 or not isinstance(transition_id, str):
        raise ValueError(f"trial {trial_id!r} has no unique aligned feature row")
    return asdict(matches[0]), transition_id


# Index trial IDs by the transition and block tested in the intervention request.
def _trial_ids_by_block(
    started: dict[str, dict], completed: dict[str, dict]
) -> dict[tuple[str, str], str]:
    result = {}
    for event in started.values():
        metadata = event.get("metadata") or {}
        if metadata.get("counterfactual_role") != "intervention":
            continue
        completed_event = completed[event["request_id"]]
        plan = (
            completed_event.get("metrics", {})
            .get("server_metrics", {})
            .get("cacheselect_partial_reuse_plan", {})
        )
        key = (
            str(plan.get("transition_id")),
            str(metadata.get("counterfactual_reuse_block_index")),
        )
        if key in result:
            raise ValueError("request ledger contains a duplicate block trial")
        result[key] = metadata.get("counterfactual_trial_id")
    return result


# Produce the final model-ready table and a hash-bound provenance summary.
def curate_mtrag_counterfactual_dataset(result_dir: Path) -> dict[str, Any]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    audit_path = result_dir / "manual-review-unblinded.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ledger_paths = tuple((result_dir / "request-logs").glob("*.jsonl"))
    if len(ledger_paths) != 1 or audit.get("abstentions") != 0:
        raise ValueError("curation requires one ledger and no unresolved reviews")
    ledger_path = ledger_paths[0]
    for name, path in (
        ("source_ledger_sha256", ledger_path),
        ("completed_blind_review_sha256", result_dir / "manual-review-blinded.json"),
        ("identity_key_sha256", result_dir / "manual-review-key.json"),
    ):
        if audit.get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"manual-review audit has a mismatched {name}")

    cases = {
        case["transition_id"]: (index, case)
        for index, case in enumerate(summary["cases"], 1)
    }
    started, completed = _load_ledger(ledger_path)
    trial_ids = _trial_ids_by_block(started, completed)
    source_path = result_dir / "mtrag-counterfactual-blocks.csv"
    with source_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != (*TRAINING_COLUMNS, *AUDIT_COLUMNS):
            raise ValueError("automatic MTRAG dataset has an unexpected schema")
        rows = [dict(row) for row in reader]
    for row in rows:
        row.update(
            trial_id=trial_ids[(row["transition_id"], row["candidate_block_index"])],
            adjudication="exact_output_match",
            review_id="",
            review_verdict="",
            review_reason="",
        )

    for reviewed in audit["rows"]:
        features, transition_id = _reviewed_feature(reviewed, started, completed)
        case_index, case = cases[transition_id]
        rows.append(
            {
                "trace_id": case["trace_id"],
                "transition_id": transition_id,
                "split": case["split"],
                "decision": reviewed["decision"],
                "label_source": "counterfactual_execution",
                "label_reason": f"Blinded semantic review: {reviewed['reason']}",
                **features,
                "mtrag_case_index": case_index,
                "mtrag_collection": case["collection"],
                "mtrag_conversation_id": case["current_task_id"].split("<::>", 1)[0],
                "mtrag_current_task_id": case["current_task_id"],
                "trial_id": reviewed["trial_id"],
                "adjudication": "blinded_semantic_review",
                "review_id": reviewed["review_id"],
                "review_verdict": reviewed["blind_verdict"],
                "review_reason": reviewed["reason"],
            }
        )

    keys = [(row["transition_id"], str(row["candidate_block_index"])) for row in rows]
    if len(keys) != len(set(keys)) or len(rows) != summary.get("trial_count"):
        raise ValueError("curated rows do not cover every unique pilot trial")
    rows.sort(
        key=lambda row: (
            int(row["mtrag_case_index"]),
            int(row["candidate_block_index"]),
        )
    )
    output_path = result_dir / "mtrag-curated-blocks.csv"
    columns = (*TRAINING_COLUMNS, *AUDIT_COLUMNS, *REVIEW_COLUMNS)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["decision"] for row in rows)
    report = {
        "schema_version": 1,
        "dataset": "mtrag-curated-counterfactual-blocks",
        "source_ledger_sha256": audit["source_ledger_sha256"],
        "manual_review_audit_sha256": hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
        "row_count": len(rows),
        "exact_match_rows": len(rows) - audit["reviewed_trials"],
        "semantic_review_rows": audit["reviewed_trials"],
        "reuse_labels": counts["reuse"],
        "repair_labels": counts["repair"],
        "output_dataset": str(output_path),
    }
    (result_dir / "mtrag-curated-summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


# Parse one result directory and write its curated dataset artifacts.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    report = curate_mtrag_counterfactual_dataset(args.result_dir)
    print(
        f"Curated {report['row_count']} labels: {report['reuse_labels']} reuse, "
        f"{report['repair_labels']} repair"
    )


if __name__ == "__main__":
    main()
