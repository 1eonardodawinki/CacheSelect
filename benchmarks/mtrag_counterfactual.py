"""Run frozen MTRAG cases through the existing counterfactual workflow."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.counterfactual_workflow import run_counterfactual_dataset_workflow
from benchmarks.counterfactual_trial import CounterfactualReferenceQualityError
from benchmarks.mtrag_trace import MtragCounterfactualCase
from benchmarks.schema import save_trace
from observability.request_recorder import RequestRecorder


COUNT_FIELDS = (
    "trial_count",
    "valid_training_rows",
    "invalid_trials",
    "abstained_trials",
    "reference_drift_trials",
    "repair_labels",
    "reuse_labels",
)


# Execute every frozen case sequentially against one already-running vLLM server.
def run_mtrag_counterfactual_cases(
    cases: Sequence[MtragCounterfactualCase],
    *,
    reference_outputs: Mapping[str, str],
    output_dir: Path,
    url: str,
    model: str,
    max_completion_tokens: int,
    api_key: str | None,
    timeout_seconds: float,
    recorder: RequestRecorder,
    require_reference_output_match: bool = False,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("MTRAG pilot contains no cases")
    summaries = []
    totals: Counter[str] = Counter()
    planned_target_blocks = sum(len(case.target_block_indices) for case in cases)
    for index, case in enumerate(cases, start=1):
        if len(case.trace.transitions) != 1:
            raise ValueError("each MTRAG case must contain exactly one transition")
        transition = case.trace.transitions[0]
        try:
            approved_output = reference_outputs[transition.current_request_id]
        except KeyError as error:
            raise ValueError("MTRAG case has no approved reference output") from error

        case_dir = output_dir / f"case-{index:02d}"
        trace_path = case_dir / "trace.json"
        dataset_path = case_dir / "counterfactual-blocks.csv"
        summary_path = case_dir / "summary.json"
        case_dir.mkdir(parents=True, exist_ok=True)
        save_trace(case.trace, trace_path)
        common = {
            "collection": case.collection,
            "current_task_id": transition.current_request_id,
            "trace_path": str(trace_path),
        }
        try:
            result = run_counterfactual_dataset_workflow(
                trace=case.trace,
                transition_id=transition.transition_id,
                split=case.split,
                output_path=dataset_path,
                url=url,
                model=model,
                max_completion_tokens=max_completion_tokens,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                recorder=recorder,
                expected_block_size=case.block_size,
                expected_candidate_block_indices=case.expected_candidate_block_indices,
                expected_testable_block_indices=case.expected_testable_block_indices,
                selected_block_indices=case.target_block_indices,
                required_reference_output=approved_output,
                require_reference_output_match=require_reference_output_match,
                require_exact_output_match=True,
            )
        except CounterfactualReferenceQualityError as error:
            result = {
                "status": "skipped_reference_quality",
                "reason": str(error),
                **{field: 0 for field in COUNT_FIELDS},
                "skipped_target_blocks": len(case.target_block_indices),
            }
            totals["skipped_reference_cases"] += 1
            totals["skipped_target_blocks"] += len(case.target_block_indices)
        else:
            result = {"status": "completed", **result, "skipped_target_blocks": 0}
            totals["completed_cases"] += 1
            for field in COUNT_FIELDS:
                totals[field] += result[field]
        summary = {**common, **result}
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)

    return {
        "schema_version": 1,
        "experiment": "mtrag-counterfactual-pilot",
        "case_count": len(summaries),
        "completed_case_count": totals["completed_cases"],
        "skipped_reference_case_count": totals["skipped_reference_cases"],
        "planned_target_blocks": planned_target_blocks,
        "skipped_target_blocks": totals["skipped_target_blocks"],
        **{field: totals[field] for field in COUNT_FIELDS},
        "cases": summaries,
    }
