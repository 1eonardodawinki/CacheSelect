"""Turn reviewed Qwen3 MTRAG runs into references and a block plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.mtrag import MTRAG_SPLIT_SEED, mtrag_conversation_split
from benchmarks.mtrag_pilot import mtrag_testable_blocks
from benchmarks.reviewed_mtrag_counterfactual import (
    QWEN3_REVIEWED_REFERENCE_GATE,
    REVIEW_STATUS,
)


MODEL = "Qwen/Qwen3-14B"
SELECTION_SEED = "cacheselect-qwen3-14b-mtrag-counterfactual-v1"
REFERENCE_FILES = (
    "train-reference-calibration.json",
    "validation-reference-calibration.json",
    "test-reference-calibration.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _index(artifact: Mapping[str, Any], field: str) -> dict[str, dict]:
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise ValueError("review artifact rows must be a list")
    indexed = {row.get(field): row for row in rows if isinstance(row, dict)}
    if (
        len(indexed) != len(rows)
        or any(not isinstance(key, str) or not key for key in indexed)
        or artifact.get("row_count", len(rows)) != len(rows)
    ):
        raise ValueError(f"invalid or duplicate {field}")
    return indexed


# Resolve one blinded review back to its exact full-computation outputs.
def _accepted_review(directory: Path, model: str) -> tuple[str, int, list[dict], dict]:
    review_path = directory / "blinded-reference-review.json"
    verdict_path = directory / "assistant-verdicts-blinded.json"
    key_path = directory / "blinded-reference-key.json"
    review, verdicts, key = map(_read, (review_path, verdict_path, key_path))
    if (
        review.get("review") != "mtrag-reference-correctness-blinded"
        or verdicts.get("source_review") != review_path.name
        or key.get("key") != "mtrag-reference-correctness-identity-key"
    ):
        raise ValueError("MTRAG review provenance is invalid")
    blinded = _index(review, "review_id")
    judgments = _index(verdicts, "review_id")
    identities = _index(key, "review_id")
    if set(blinded) != set(judgments) or set(blinded) != set(identities):
        raise ValueError("review artifacts contain different rows")

    expected_hashes = {
        Path(name).name: digest
        for name, digest in (key.get("source_artifact_sha256") or {}).items()
    }
    if not expected_hashes or not set(expected_hashes) <= set(REFERENCE_FILES):
        raise ValueError("identity key references unexpected source artifacts")
    references, source_hash = {}, None
    for filename in sorted(expected_hashes):
        path = directory / filename
        artifact = _read(path)
        current_hash = artifact.get("source_sha256")
        if (
            artifact.get("analysis") != "mtrag-reference-quality-calibration"
            or artifact.get("model") != model
            or expected_hashes.get(filename) != _sha256(path)
            or not isinstance(current_hash, str)
            or (source_hash is not None and current_hash != source_hash)
        ):
            raise ValueError("reference artifact provenance is invalid")
        source_hash = current_hash
        references.update(
            {(filename, task_id): row for task_id, row in _index(artifact, "task_id").items()}
        )
    accepted = []
    for review_id in sorted(blinded):
        judgment, identity = judgments[review_id], identities[review_id]
        verdict = judgment.get("verdict")
        if verdict not in {"pass", "concern", "fail"} or not judgment.get("reason"):
            raise ValueError("reference verdict is incomplete")
        source_name = Path(str(identity.get("source_artifact"))).name
        reference = references.get((source_name, identity.get("task_id")))
        if not reference or (
            reference.get("split") != identity.get("split")
            or reference.get("collection") != identity.get("collection")
            or reference.get("output_text") != blinded[review_id].get("model_answer")
            or reference.get("expected_answer")
            != blinded[review_id].get("expected_answer")
            or reference.get("cached_tokens") != 0
            or reference.get("finish_reason") != "stop"
        ):
            raise ValueError("review does not match its full-compute reference")
        if verdict == "pass":
            output = reference["output_text"]
            accepted.append(
                {
                    "review_id": review_id,
                    "task_id": identity["task_id"],
                    "split": identity["split"],
                    "collection": identity["collection"],
                    "prompt_token_count": reference["prompt_token_count"],
                    "reference_output": output,
                    "reference_output_sha256": hashlib.sha256(
                        output.encode()
                    ).hexdigest(),
                    "expected_answer": reference["expected_answer"],
                    "source_reference_artifact": source_name,
                }
            )
    provenance = {
        "source_id": directory.name,
        "review_sha256": _sha256(review_path),
        "verdicts_sha256": _sha256(verdict_path),
        "key_sha256": _sha256(key_path),
        "reference_sha256": {
            name: _sha256(directory / name) for name in REFERENCE_FILES
        },
    }
    assert source_hash is not None
    return source_hash, len(judgments), accepted, provenance


def _rank(task_id: str, block: int) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{task_id}:{block}".encode()).hexdigest()


# Build both frozen artifacts in one pass, avoiding intermediate manifests.
def prepare_reviewed_mtrag_counterfactual(
    review_dirs: tuple[Path, ...],
    coverage_path: Path,
    *,
    model: str = MODEL,
    max_target_blocks: int | None = None,
    include_repacking: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not review_dirs
        or (
            max_target_blocks is not None
            and (
                isinstance(max_target_blocks, bool)
                or max_target_blocks < 1
            )
        )
    ):
        raise ValueError(
            "review directories and a valid target count are required"
        )
    tasks, sources, reviewed, source_hash = [], [], 0, None
    for directory in review_dirs:
        current_hash, current_count, current_tasks, provenance = _accepted_review(
            directory, model
        )
        if source_hash is not None and current_hash != source_hash:
            raise ValueError("review runs use different MTRAG sources")
        source_hash = current_hash
        reviewed += current_count
        tasks.extend(current_tasks)
        sources.append(provenance)
    task_ids = [row["task_id"] for row in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("review runs contain duplicate tasks")
    tasks.sort(key=lambda row: (row["split"], row["task_id"]))
    references = {
        "schema_version": 1,
        "manifest": "mtrag-counterfactual-reference-set",
        "status": REVIEW_STATUS,
        "model": model,
        "source_dataset_sha256": source_hash,
        "source_reviews": sorted(sources, key=lambda row: row["source_id"]),
        "reviewed_task_count": reviewed,
        "accepted_task_count": len(tasks),
        "accepted_by_split": dict(Counter(row["split"] for row in tasks)),
        "accepted_by_collection": dict(
            Counter(row["collection"] for row in tasks)
        ),
        "tasks": tasks,
    }

    coverage = _read(coverage_path)
    rows = coverage.get("transitions")
    block_size = coverage.get("block_size")
    if (
        coverage.get("analysis") != "mtrag-natural-block-coverage"
        or coverage.get("model") != model
        or coverage.get("source_sha256") != source_hash
        or not isinstance(rows, list)
        or not isinstance(block_size, int)
        or isinstance(block_size, bool)
    ):
        raise ValueError("coverage and reviewed references are incompatible")
    by_task = {
        row.get("current_task_id"): row for row in rows if isinstance(row, Mapping)
    }
    if len(by_task) != len(rows):
        raise ValueError("coverage contains duplicate or invalid tasks")

    selected = []
    for reference in tasks:
        task_id = reference["task_id"]
        row = by_task.get(task_id)
        opportunity = row.get("reuse_opportunity") if isinstance(row, Mapping) else None
        conversation_id = row.get("conversation_id") if isinstance(row, Mapping) else None
        if (
            not isinstance(opportunity, Mapping)
            or not isinstance(conversation_id, str)
            or row.get("collection") != reference["collection"]
            or mtrag_conversation_split(conversation_id).value != reference["split"]
            or opportunity.get("current_token_count")
            != reference["prompt_token_count"]
        ):
            raise ValueError("reviewed reference prompt geometry drifted")
        candidates, testable, excluded = mtrag_testable_blocks(
            row,
            block_size=block_size,
            include_repacking=include_repacking,
        )
        if not testable:
            raise ValueError("reviewed reference has no testable block")
        if max_target_blocks is None:
            targets = list(testable)
        else:
            ranked = sorted(testable, key=lambda block: _rank(task_id, block))
            targets = sorted(ranked[:max_target_blocks])
        selected.append(
            {
                "split": reference["split"],
                "conversation_id": conversation_id,
                "collection": reference["collection"],
                "previous_task_id": row["previous_task_id"],
                "current_task_id": task_id,
                "shared_document_ids": row["shared_document_ids"],
                "candidate_block_indices": list(candidates),
                "testable_block_indices": list(testable),
                "target_block_indices": targets,
                "excluded_output_block_index": excluded,
            }
        )
    selected.sort(key=lambda row: (row["split"], row["current_task_id"]))
    reference_bytes = (json.dumps(references, indent=2) + "\n").encode()
    plan = {
        "schema_version": 1,
        "selection": "mtrag-audited-counterfactual-pilot",
        "review_status": REVIEW_STATUS,
        "quality_calibration_id": QWEN3_REVIEWED_REFERENCE_GATE.calibration_id,
        "source_model": model,
        "source_prompt_template_version": coverage.get("prompt_template_version"),
        "source_dataset_sha256": source_hash,
        "source_coverage_sha256": _sha256(coverage_path),
        "source_reference_manifest_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "split_seed": MTRAG_SPLIT_SEED,
        "selection_seed": SELECTION_SEED,
        "block_size": block_size,
        "max_target_blocks": max_target_blocks,
        "repacking_enabled": include_repacking,
        "transition_count": len(selected),
        "transition_count_by_split": dict(
            Counter(row["split"] for row in selected)
        ),
        "total_testable_blocks": sum(
            len(row["testable_block_indices"]) for row in selected
        ),
        "total_target_blocks": sum(
            len(row["target_block_indices"]) for row in selected
        ),
        "transitions": selected,
    }
    return references, plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, action="append", required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-target-blocks", type=int)
    parser.add_argument("--include-repacking", action="store_true")
    parser.add_argument("--references-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    args = parser.parse_args()
    references, plan = prepare_reviewed_mtrag_counterfactual(
        tuple(args.review_dir),
        args.coverage,
        model=args.model,
        max_target_blocks=args.max_target_blocks,
        include_repacking=args.include_repacking,
    )
    for path, artifact in (
        (args.references_output, references),
        (args.plan_output, plan),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"Prepared {len(references['tasks'])} references and "
        f"{plan['total_target_blocks']} block trials"
    )


if __name__ == "__main__":
    main()
