"""Load reviewed Qwen3 MTRAG references and build their frozen cases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.mtrag import MtragTask
from benchmarks.mtrag_trace import (
    MtragCounterfactualCase,
    build_mtrag_counterfactual_cases,
)
from benchmarks.schema import ReferenceSimilarityGate


REVIEW_STATUS = "provisional_assistant_review_requires_researcher_audit"
QWEN3_REVIEWED_REFERENCE_GATE = ReferenceSimilarityGate(
    minimum_token_recall=0.15,
    minimum_rouge_l_f1=0.10,
    maximum_metric_drop=0.02,
    calibration_id="mtrag-qwen3-14b-reviewed-exact-v1",
)


@dataclass(frozen=True)
class ReviewedMtragReferenceSet:
    """Reviewed outputs plus the provenance needed by causal trials."""

    status: str
    source_dataset_sha256: str
    tasks_by_id: dict[str, dict[str, Any]]
    manifest_sha256: str

    @property
    def reference_outputs(self) -> dict[str, str]:
        return {
            task_id: row["reference_output"]
            for task_id, row in self.tasks_by_id.items()
        }


# Validate the exact outputs approved before any counterfactual run.
def load_reviewed_mtrag_reference_set(
    path: Path,
    *,
    expected_model: str,
) -> ReviewedMtragReferenceSet:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("tasks") if isinstance(manifest, dict) else None
    source_hash = manifest.get("source_dataset_sha256", "")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("manifest") != "mtrag-counterfactual-reference-set"
        or manifest.get("status") != REVIEW_STATUS
        or manifest.get("model") != expected_model
        or not isinstance(rows, list)
        or manifest.get("accepted_task_count") != len(rows)
        or not isinstance(source_hash, str)
        or len(source_hash) != 64
    ):
        raise ValueError("reviewed MTRAG reference manifest is invalid")

    tasks_by_id = {}
    for row in rows:
        task_id = row.get("task_id") if isinstance(row, dict) else None
        output = row.get("reference_output") if isinstance(row, dict) else None
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in tasks_by_id
            or row.get("split") not in {"train", "validation"}
            or not isinstance(row.get("collection"), str)
            or not isinstance(output, str)
            or not output
            or hashlib.sha256(output.encode()).hexdigest()
            != row.get("reference_output_sha256")
        ):
            raise ValueError("reviewed MTRAG reference row is invalid")
        tasks_by_id[task_id] = row
    return ReviewedMtragReferenceSet(
        manifest["status"],
        source_hash,
        tasks_by_id,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


# Add reference-specific checks, then reuse the generic MTRAG case builder.
def build_reviewed_mtrag_counterfactual_cases(
    tasks: Sequence[MtragTask],
    plan: Mapping[str, Any],
    references: ReviewedMtragReferenceSet,
    *,
    expected_model: str,
) -> tuple[MtragCounterfactualCase, ...]:
    rows = plan.get("transitions")
    if (
        plan.get("review_status") != references.status
        or plan.get("source_dataset_sha256") != references.source_dataset_sha256
        or plan.get("source_reference_manifest_sha256")
        != references.manifest_sha256
        or not isinstance(rows, list)
    ):
        raise ValueError("reviewed MTRAG block plan provenance is invalid")
    planned = {
        row.get("current_task_id"): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(planned) != set(references.tasks_by_id) or len(planned) != len(rows):
        raise ValueError("reviewed MTRAG plan and references contain different tasks")
    if any(
        row.get("split") != references.tasks_by_id[task_id].get("split")
        or row.get("collection")
        != references.tasks_by_id[task_id].get("collection")
        for task_id, row in planned.items()
    ):
        raise ValueError("reviewed MTRAG reference identity drifted")
    return build_mtrag_counterfactual_cases(
        tasks,
        plan,
        quality_gate=QWEN3_REVIEWED_REFERENCE_GATE,
        approved_task_ids=frozenset(references.tasks_by_id),
        expected_model=expected_model,
    )
