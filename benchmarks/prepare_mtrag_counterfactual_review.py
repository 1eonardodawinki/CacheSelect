"""Prepare a blinded semantic review of valid abstained MTRAG block trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from observability.request_recorder import validate_ledger


# Load a complete request ledger into request-ID indexed lifecycle events.
def _load_ledger(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    validation = validate_ledger(path)
    if not validation.is_complete or validation.failed:
        raise ValueError("MTRAG review requires a complete successful request ledger")
    started: dict[str, dict] = {}
    completed: dict[str, dict] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            request_id = event["request_id"]
            if event["event"] == "request_started":
                started[request_id] = event
            elif event["event"] == "request_completed":
                completed[request_id] = event
    return started, completed


# Build anonymous Answer A/B rows and a separate identity key for later unblinding.
def prepare_mtrag_counterfactual_review(result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ledger_paths = tuple((result_dir / "request-logs").glob("*.jsonl"))
    if len(ledger_paths) != 1:
        raise ValueError("MTRAG result must contain exactly one request ledger")
    trial_count = summary.get("trial_count")
    invalid_count = summary.get("invalid_trials")
    abstained_count = summary.get("abstained_trials")
    if not all(isinstance(value, int) for value in (trial_count, invalid_count, abstained_count)):
        raise ValueError("MTRAG result has invalid trial counts")
    invalid_references = set()
    rejected_trials = 0
    for case in summary.get("cases", []):
        if case.get("invalid_trials", 0):
            if case.get("invalid_trials") != case.get("trial_count") or not case.get(
                "discovery_id"
            ):
                raise ValueError("invalid MTRAG case is not wholly rejected")
            invalid_references.add(f"{case['discovery_id']}:edited")
            rejected_trials += case["invalid_trials"]
    if rejected_trials != invalid_count:
        raise ValueError("invalid trials do not match rejected reference cases")

    ledger_path = ledger_paths[0]
    ledger_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    started, completed = _load_ledger(ledger_path)
    by_benchmark_id = {
        event.get("metadata", {}).get("benchmark_request_id"): event
        for event in started.values()
    }
    trials: dict[str, dict[str, dict]] = {}
    for event in started.values():
        metadata = event.get("metadata") or {}
        role = metadata.get("counterfactual_role")
        trial_id = metadata.get("counterfactual_trial_id")
        if role not in {"reference", "intervention"}:
            continue
        if not isinstance(trial_id, str) or role in trials.setdefault(trial_id, {}):
            raise ValueError("MTRAG ledger has duplicate or invalid trial roles")
        trials[trial_id][role] = event
    for roles in trials.values():
        if "reference" not in roles and "intervention" in roles:
            reference_id = roles["intervention"].get("metadata", {}).get(
                "counterfactual_reference_request_id"
            )
            reference = by_benchmark_id.get(reference_id)
            if isinstance(reference, dict):
                roles["reference"] = reference
    if len(trials) != trial_count or any(len(roles) != 2 for roles in trials.values()):
        raise ValueError("MTRAG ledger does not contain every planned trial")

    stability_checks = {
        event.get("metadata", {}).get("counterfactual_reference_request_id"): event
        for event in started.values()
        if event.get("metadata", {}).get("counterfactual_role")
        == "stability_reference"
    }

    abstained = []
    for trial_id, roles in trials.items():
        reference_start = roles["reference"]
        intervention_start = roles["intervention"]
        reference_done = completed[reference_start["request_id"]]
        intervention_done = completed[intervention_start["request_id"]]
        reference_text = (reference_done.get("output") or {}).get("text")
        intervention_text = (intervention_done.get("output") or {}).get("text")
        reference_id = reference_start.get("metadata", {}).get(
            "benchmark_request_id"
        )
        if reference_id in invalid_references:
            continue
        stability_start = stability_checks.get(reference_id)
        stability_done = (
            completed.get(stability_start["request_id"])
            if isinstance(stability_start, dict)
            else None
        )
        stability_text = (
            (stability_done.get("output") or {}).get("text")
            if isinstance(stability_done, dict)
            else None
        )
        stability_choices = (
            ((stability_done.get("output") or {}).get("raw_response") or {}).get(
                "choices"
            )
            if isinstance(stability_done, dict)
            else None
        )
        stability_finish = (
            stability_choices[0].get("finish_reason") if stability_choices else None
        )
        unstable_reference = (
            stability_text is not None
            and (
                stability_text != reference_text
                or stability_finish not in {None, "stop"}
            )
        )
        if reference_text == intervention_text and not unstable_reference:
            continue
        evaluation = intervention_start.get("evaluation") or {}
        queries = [
            segment.get("content")
            for segment in evaluation.get("prompt_segments") or []
            if segment.get("kind") == "query"
        ]
        expected = (evaluation.get("ground_truth") or {}).get("expected_answer")
        block_index = (intervention_start.get("metadata") or {}).get(
            "counterfactual_reuse_block_index"
        )
        if (
            len(queries) != 1
            or not isinstance(queries[0], str)
            or not isinstance(expected, str)
            or not expected
            or not isinstance(reference_text, str)
            or not isinstance(intervention_text, str)
            or not isinstance(block_index, int)
        ):
            raise ValueError("MTRAG abstention lacks review evidence")
        abstained.append(
            (
                trial_id,
                block_index,
                queries[0],
                expected,
                reference_text,
                intervention_text,
                stability_text if unstable_reference else None,
            )
        )
    if len(abstained) != abstained_count:
        raise ValueError("non-exact ledger trials do not match recorded abstentions")

    review_rows = []
    key_rows = []
    for index, row in enumerate(
        sorted(abstained, key=lambda item: hashlib.sha256(item[0].encode()).digest()),
        start=1,
    ):
        (
            trial_id,
            block_index,
            question,
            expected,
            reference,
            intervention,
            stability,
        ) = row
        review_id = f"review-{index:03d}"
        reference_is_a = hashlib.sha256(
            f"{ledger_sha256}:{trial_id}".encode()
        ).digest()[0] % 2 == 0
        answer_a, answer_b = (
            (reference, intervention) if reference_is_a else (intervention, reference)
        )
        review_rows.append(
            {
                "review_id": review_id,
                "question": question,
                "expected_answer": expected,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "independent_normal_answer": stability,
                "verdict": "",
                "reason": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "trial_id": trial_id,
                "block_index": block_index,
                "reference_slot": "answer_a" if reference_is_a else "answer_b",
                "intervention_slot": "answer_b" if reference_is_a else "answer_a",
            }
        )

    shared = {"schema_version": 1, "source_ledger_sha256": ledger_sha256}
    review = {
        **shared,
        "review": "mtrag-counterfactual-blinded-semantic-review",
        "allowed_verdicts": ["equivalent", "answer_a_better", "answer_b_better", "unclear"],
        "rubric": "Judge correctness, completeness, relevance, and unsupported claims against the expected answer. Do not infer which answer used KV reuse. If an independent normal answer is present, choose unclear unless its meaning is compatible with the compared answers.",
        "rows": review_rows,
    }
    key = {
        **shared,
        "key": "mtrag-counterfactual-review-identity-key",
        "warning": "Do not inspect until every blinded verdict and reason is frozen.",
        "rows": key_rows,
    }
    review_path = result_dir / "manual-review-blinded.json"
    key_path = result_dir / "manual-review-key.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return {"review_path": str(review_path), "key_path": str(key_path), "rows": len(review_rows)}


# Parse one result directory and write its blinded review artifacts.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_mtrag_counterfactual_review(args.result_dir)
    print(f"Prepared {result['rows']} blinded reviews at {result['review_path']}")
    print(f"Keep the identity key sealed at {result['key_path']}")


if __name__ == "__main__":
    main()
