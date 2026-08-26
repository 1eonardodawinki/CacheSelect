"""Prepare a blinded correctness review of MTRAG reference answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


# Build a stable opaque identifier without exposing task or split identity.
def _review_id(seed: str, task_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
    return f"reference-{digest[:12]}"


# Convert reference artifacts into separate blinded review and identity files.
def prepare_mtrag_reference_review(
    paths: tuple[Path, ...],
    *,
    seed: str = "cacheselect-mtrag-reference-review-v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not paths or not seed:
        raise ValueError("reference review requires inputs and a seed")
    review_rows = []
    key_rows = []
    task_ids = set()
    source_hashes = {}
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        rows = artifact.get("rows")
        if (
            artifact.get("analysis") != "mtrag-reference-quality-calibration"
            or not isinstance(rows, list)
        ):
            raise ValueError(f"{path} is not an MTRAG reference artifact")
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("reference row must be an object")
            task_id = row.get("task_id")
            split = row.get("split")
            output = row.get("output_text")
            expected = row.get("expected_answer")
            collection = row.get("collection")
            quality = row.get("quality")
            if (
                not isinstance(task_id, str)
                or not task_id
                or task_id in task_ids
                or split not in {"train", "validation", "test"}
                or not isinstance(output, str)
                or not output
                or not isinstance(expected, str)
                or not expected
                or not isinstance(collection, str)
                or not isinstance(quality, dict)
            ):
                raise ValueError("reference row has invalid review fields")
            task_ids.add(task_id)
            review_id = _review_id(seed, task_id)
            metrics = quality.get("metrics") or {}
            review_rows.append(
                {
                    "review_id": review_id,
                    "model_answer": output,
                    "expected_answer": expected,
                    "token_recall": metrics.get("token_recall"),
                    "rouge_l_f1": metrics.get("rouge_l_f1"),
                    "verdict": "",
                    "reason": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "task_id": task_id,
                    "split": split,
                    "collection": collection,
                    "source_artifact": str(path),
                }
            )
    review_rows.sort(key=lambda row: row["review_id"])
    key_rows.sort(key=lambda row: row["review_id"])
    return (
        {
            "schema_version": 1,
            "review": "mtrag-reference-correctness-blinded",
            "review_policy": (
                "Compare the model answer with the expected answer for factual "
                "correctness, relevance, omissions, and unsupported claims."
            ),
            "allowed_verdicts": ["pass", "concern", "fail"],
            "row_count": len(review_rows),
            "rows": review_rows,
        },
        {
            "schema_version": 1,
            "key": "mtrag-reference-correctness-identity-key",
            "seed": seed,
            "source_artifact_sha256": source_hashes,
            "row_count": len(key_rows),
            "rows": key_rows,
            "warning": "Do not inspect until every blinded verdict is frozen.",
        },
    )


# Parse paths and save the blinded review separately from its identity key.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    args = parser.parse_args()
    review, key = prepare_mtrag_reference_review(tuple(args.input))
    for path, artifact in ((args.review_output, review), (args.key_output, key)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {review['row_count']} blinded reference answers")


if __name__ == "__main__":
    main()
