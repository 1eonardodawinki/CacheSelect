"""Review identical blinded MTRAG answer pairs only once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.unblind_mtrag_counterfactual_review import ALLOWED_VERDICTS


def _key(row: dict) -> tuple:
    answer_a, answer_b = sorted((row["answer_a"], row["answer_b"]))
    return (
        row["question"],
        row["expected_answer"],
        answer_a,
        answer_b,
        row["independent_normal_answer"],
    )


def prepare_compact_review(result_dir: Path) -> dict:
    source_path = result_dir / "manual-review-blinded.json"
    source = json.loads(source_path.read_text())
    groups = {}
    for row in source["rows"]:
        groups.setdefault(_key(row), []).append(row["review_id"])

    rows = []
    for index, (key, review_ids) in enumerate(groups.items(), 1):
        question, expected, answer_a, answer_b, independent = key
        rows.append(
            {
                "group_id": f"group-{index:04d}",
                "occurrences": len(review_ids),
                "review_ids": review_ids,
                "question": question,
                "expected_answer": expected,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "independent_normal_answer": independent,
                "verdict": "",
                "reason": "",
            }
        )
    compact = {
        "schema_version": 1,
        "source_review_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "allowed_verdicts": sorted(ALLOWED_VERDICTS),
        "rubric": source["rubric"],
        "trial_count": len(source["rows"]),
        "group_count": len(rows),
        "rows": rows,
    }
    (result_dir / "manual-review-compact.json").write_text(
        json.dumps(compact, indent=2) + "\n"
    )
    return compact


def apply_compact_review(result_dir: Path) -> dict[str, int]:
    source_path = result_dir / "manual-review-blinded.json"
    compact_path = result_dir / "manual-review-compact.json"
    source = json.loads(source_path.read_text())
    compact = json.loads(compact_path.read_text())
    if compact["source_review_sha256"] != hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest():
        raise ValueError("compact review does not match its source")

    verdicts = {}
    for row in compact["rows"]:
        if row["verdict"] not in ALLOWED_VERDICTS or not row["reason"]:
            raise ValueError(f"{row['group_id']} is incomplete")
        verdicts[_key(row)] = (row["verdict"], row["reason"])

    counts = {verdict: 0 for verdict in ALLOWED_VERDICTS}
    for row in source["rows"]:
        key = _key(row)
        verdict, reason = verdicts[key]
        if verdict in {"answer_a_better", "answer_b_better"}:
            winner = verdict.removesuffix("_better")
            if row["answer_a"] != key[2]:
                winner = "answer_b" if winner == "answer_a" else "answer_a"
            verdict = f"{winner}_better"
        row.update(verdict=verdict, reason=reason)
        counts[verdict] += 1
    if len(verdicts) != compact["group_count"] or len(source["rows"]) != compact[
        "trial_count"
    ]:
        raise ValueError("compact review has incomplete coverage")
    source_path.write_text(json.dumps(source, indent=2) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        counts = apply_compact_review(args.result_dir)
        print(f"Applied {sum(counts.values())} blinded verdicts: {counts}")
    else:
        compact = prepare_compact_review(args.result_dir)
        print(f"Compacted {compact['trial_count']} trials into {compact['group_count']}")


if __name__ == "__main__":
    main()
