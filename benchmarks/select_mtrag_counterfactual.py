"""Freeze an audited MTRAG counterfactual experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.mtrag_pilot import select_audited_mtrag_counterfactual_pilot
from benchmarks.mtrag_quality import load_mtrag_manual_quality_audit


MTRAG_SOURCE = "IBM/mt-rag-benchmark/mtrag-human/generation_tasks/RAG.jsonl"


# Parse selection inputs separately from the deterministic selector.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-collection", type=int, default=5)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-testable-blocks", type=int, default=64)
    parser.add_argument("--max-target-blocks", type=int, default=12)
    return parser.parse_args()


# Select approved transitions and attach immutable source provenance.
def build_audited_mtrag_manifest(
    *,
    coverage_path: Path,
    audit_path: Path,
    reference_artifact_path: Path,
    model: str,
    per_collection: int,
    max_prompt_tokens: int,
    max_testable_blocks: int,
    max_target_blocks: int,
) -> dict:
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    audit = load_mtrag_manual_quality_audit(
        audit_path,
        reference_artifact_path=reference_artifact_path,
        expected_model=model,
    )
    selected = select_audited_mtrag_counterfactual_pilot(
        coverage,
        approved_task_ids=audit.approved_task_ids,
        quality_calibration_id=audit.calibration_id,
        per_collection=per_collection,
        max_prompt_tokens=max_prompt_tokens,
        max_testable_blocks=max_testable_blocks,
        max_target_blocks=max_target_blocks,
    )
    return {
        **selected,
        "source": MTRAG_SOURCE,
        "source_revision": audit.source_dataset_revision,
        "source_sha256": audit.source_dataset_sha256,
        "source_coverage_sha256": hashlib.sha256(
            coverage_path.read_bytes()
        ).hexdigest(),
        "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "source_reference_artifact_sha256": hashlib.sha256(
            reference_artifact_path.read_bytes()
        ).hexdigest(),
    }


# Build and save one reproducible experiment manifest.
def main() -> None:
    args = _parse_args()
    manifest = build_audited_mtrag_manifest(
        coverage_path=args.coverage,
        audit_path=args.audit,
        reference_artifact_path=args.reference_artifact,
        model=args.model,
        per_collection=args.per_collection,
        max_prompt_tokens=args.max_prompt_tokens,
        max_testable_blocks=args.max_testable_blocks,
        max_target_blocks=args.max_target_blocks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected {manifest['transition_count']} transitions and "
        f"{manifest['total_target_blocks']} target blocks"
    )
    print(f"Saved manifest to {args.output}")


if __name__ == "__main__":
    main()
