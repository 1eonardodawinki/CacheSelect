"""Unblind a completed MTRAG semantic review into block decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_VERDICTS = {
    "equivalent",
    "answer_a_better",
    "answer_b_better",
    "unclear",
}


# Read one JSON object and reject other top-level JSON values.
def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


# Convert one frozen blind verdict using only its separately stored identity row.
def _unblind_decision(verdict: str, key_row: dict[str, Any]) -> str:
    if verdict == "equivalent":
        return "reuse"
    if verdict == "unclear":
        return "abstain"
    better_slot = verdict.removesuffix("_better")
    return "reuse" if key_row.get("intervention_slot") == better_slot else "repair"


# Validate and unblind every reviewed trial while preserving its audit evidence.
def unblind_mtrag_counterfactual_review(result_dir: Path) -> dict[str, Any]:
    review_path = result_dir / "manual-review-blinded.json"
    key_path = result_dir / "manual-review-key.json"
    ledger_paths = tuple((result_dir / "request-logs").glob("*.jsonl"))
    if len(ledger_paths) != 1:
        raise ValueError("MTRAG result must contain exactly one request ledger")

    review = _read_object(review_path)
    key = _read_object(key_path)
    ledger_sha256 = hashlib.sha256(ledger_paths[0].read_bytes()).hexdigest()
    if not (
        review.get("source_ledger_sha256")
        == key.get("source_ledger_sha256")
        == ledger_sha256
    ):
        raise ValueError("review, key, and request ledger do not match")

    review_rows = review.get("rows")
    key_rows = key.get("rows")
    if not isinstance(review_rows, list) or not isinstance(key_rows, list):
        raise ValueError("review and key rows must be lists")
    keys_by_id = {row.get("review_id"): row for row in key_rows if isinstance(row, dict)}
    if len(keys_by_id) != len(key_rows):
        raise ValueError("identity key contains duplicate or invalid review IDs")

    decisions = []
    for row in review_rows:
        if not isinstance(row, dict):
            raise ValueError("review row must be an object")
        review_id = row.get("review_id")
        verdict = row.get("verdict")
        reason = row.get("reason")
        key_row = keys_by_id.get(review_id)
        if verdict not in ALLOWED_VERDICTS or not isinstance(reason, str) or not reason:
            raise ValueError(f"review {review_id!r} is incomplete")
        if not isinstance(key_row, dict):
            raise ValueError(f"review {review_id!r} has no identity key")
        decisions.append(
            {
                "review_id": review_id,
                "trial_id": key_row.get("trial_id"),
                "block_index": key_row.get("block_index"),
                "blind_verdict": verdict,
                "decision": _unblind_decision(verdict, key_row),
                "reason": reason,
            }
        )
    if len(decisions) != len(key_rows):
        raise ValueError("review and identity key contain different trials")

    counts = Counter(row["decision"] for row in decisions)
    audit = {
        "schema_version": 1,
        "audit": "mtrag-counterfactual-semantic-review",
        "source_ledger_sha256": ledger_sha256,
        "completed_blind_review_sha256": hashlib.sha256(
            review_path.read_bytes()
        ).hexdigest(),
        "identity_key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
        "reviewed_trials": len(decisions),
        "reuse_decisions": counts["reuse"],
        "repair_decisions": counts["repair"],
        "abstentions": counts["abstain"],
        "rows": decisions,
    }
    output_path = result_dir / "manual-review-unblinded.json"
    output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


# Parse one result directory and write its unblinded semantic audit.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = unblind_mtrag_counterfactual_review(args.result_dir)
    print(
        f"Unblinded {audit['reviewed_trials']} reviews: "
        f"{audit['reuse_decisions']} reuse, {audit['repair_decisions']} repair, "
        f"{audit['abstentions']} abstain"
    )


if __name__ == "__main__":
    main()
