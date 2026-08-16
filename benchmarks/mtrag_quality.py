"""Load a manually audited MTRAG reference-quality gate."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.mtrag import mtrag_source_sha256
from benchmarks.schema import ReferenceSimilarityGate


@dataclass(frozen=True)
class MtragManualQualityAudit:
    """A frozen gate and the training task IDs approved to use it."""

    calibration_id: str
    quality_gate: ReferenceSimilarityGate
    approved_task_ids: frozenset[str]
    approved_reference_outputs: dict[str, str]
    source_dataset_revision: str
    source_dataset_sha256: str
    reviewed_task_count: int


# Validate one audit and return only the task IDs approved before interventions.
def load_mtrag_manual_quality_audit(
    path: Path,
    *,
    reference_artifact_path: Path,
    expected_model: str,
) -> MtragManualQualityAudit:
    audit: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    reference: dict[str, Any] = json.loads(
        reference_artifact_path.read_text(encoding="utf-8")
    )
    if (
        audit.get("schema_version") != 1
        or audit.get("audit") != "mtrag-reference-quality-manual-audit"
        or audit.get("split") != "train"
        or audit.get("model") != expected_model
        or reference.get("model") != expected_model
        or audit.get("source_run_id") != reference.get("run_id")
        or audit.get("source_artifact_sha256")
        != mtrag_source_sha256(reference_artifact_path)
    ):
        raise ValueError("MTRAG manual audit provenance is invalid")

    calibration_id = audit.get("calibration_id")
    source_revision = audit.get("source_dataset_revision")
    source_sha256 = audit.get("source_dataset_sha256")
    rows = audit.get("rows")
    summary = audit.get("summary")
    gate = audit.get("gate")
    reference_rows = reference.get("rows")
    if (
        not isinstance(calibration_id, str)
        or not calibration_id
        or not isinstance(source_revision, str)
        or not source_revision
        or not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
        or not isinstance(rows, list)
        or not isinstance(summary, dict)
        or not isinstance(gate, dict)
        or not isinstance(reference_rows, list)
    ):
        raise ValueError("MTRAG manual audit structure is invalid")

    verdicts: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("MTRAG manual audit row must be an object")
        task_id = row.get("task_id")
        verdict = row.get("verdict")
        reason = row.get("reason")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in verdicts
            or verdict not in {"pass", "concern", "fail"}
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError("MTRAG manual audit row is invalid")
        verdicts[task_id] = verdict

    reference_outputs: dict[str, str] = {}
    for row in reference_rows:
        if not isinstance(row, dict):
            raise ValueError("MTRAG reference row must be an object")
        task_id = row.get("task_id")
        output = row.get("output_text")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in reference_outputs
            or not isinstance(output, str)
            or not output
        ):
            raise ValueError("MTRAG reference row is invalid")
        reference_outputs[task_id] = output
    counts = Counter(verdicts.values())
    if (
        set(verdicts) != set(reference_outputs)
        or summary.get("reviewed") != len(verdicts)
        or any(summary.get(verdict) != counts[verdict] for verdict in counts)
    ):
        raise ValueError("MTRAG manual audit counts or task IDs are inconsistent")

    quality_gate = ReferenceSimilarityGate(
        minimum_token_recall=gate.get("minimum_token_recall"),
        minimum_rouge_l_f1=gate.get("minimum_rouge_l_f1"),
        maximum_metric_drop=gate.get("maximum_metric_drop"),
        calibration_id=calibration_id,
    )
    approved = frozenset(
        task_id for task_id, verdict in verdicts.items() if verdict == "pass"
    )
    if not approved:
        raise ValueError("MTRAG manual audit approved no reference tasks")
    approved_outputs = {
        task_id: reference_outputs[task_id] for task_id in sorted(approved)
    }
    return MtragManualQualityAudit(
        calibration_id,
        quality_gate,
        approved,
        approved_outputs,
        source_revision,
        source_sha256,
        len(verdicts),
    )
