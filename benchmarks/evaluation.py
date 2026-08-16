"""Deterministic response scoring for controlled CacheSelect workloads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from benchmarks.reference_quality import score_reference_answer
from benchmarks.schema import RequestGroundTruth


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


# Select the configured exact-fact or natural-reference quality mode.
def score_response(
    response_text: str | None,
    ground_truth: RequestGroundTruth,
) -> dict[str, Any]:
    """Score a response with the request's explicit quality policy."""

    gate = ground_truth.reference_similarity_gate
    if ground_truth.requirements and gate is not None:
        raise ValueError("ground truth cannot combine requirements and reference gate")
    if gate is not None:
        metrics = score_reference_answer(response_text, ground_truth.expected_answer)
        passed = (
            metrics["token_recall"] >= gate.minimum_token_recall
            and metrics["rouge_l_f1"] >= gate.minimum_rouge_l_f1
        )
        return {
            "mode": "reference_similarity",
            "passed": passed,
            "score": min(metrics.values()),
            "metrics": metrics,
            "thresholds": {
                "minimum_token_recall": gate.minimum_token_recall,
                "minimum_rouge_l_f1": gate.minimum_rouge_l_f1,
                "maximum_metric_drop": gate.maximum_metric_drop,
            },
            "calibration_id": gate.calibration_id,
            "requirements_passed": 0,
            "requirements_total": 0,
            "requirements": [],
        }
    if not ground_truth.requirements:
        raise ValueError("ground truth has no configured quality gate")

    normalised = _normalise(response_text or "")
    requirements = []
    for requirement in ground_truth.requirements:
        matched_phrase = next(
            (
                phrase
                for phrase in requirement.accepted_phrases
                if _normalise(phrase) in normalised
            ),
            None,
        )
        requirements.append(
            {
                "requirement_id": requirement.requirement_id,
                "passed": matched_phrase is not None,
                "matched_phrase": matched_phrase,
            }
        )
    passed_count = sum(item["passed"] for item in requirements)
    total = len(requirements)
    return {
        "mode": "requirements",
        "passed": passed_count == total,
        "requirements_passed": passed_count,
        "requirements_total": total,
        "score": passed_count / total,
        "requirements": requirements,
    }


# Compare an intervention with full computation at the configured quality gate.
def compare_response_quality(
    reference_quality: Mapping[str, Any],
    intervention_quality: Mapping[str, Any],
    ground_truth: RequestGroundTruth,
) -> dict[str, Any]:
    gate = ground_truth.reference_similarity_gate
    expected_mode = "reference_similarity" if gate is not None else "requirements"
    for name, quality in (
        ("reference", reference_quality),
        ("intervention", intervention_quality),
    ):
        reported_mode = quality.get("mode")
        if reported_mode != expected_mode:
            raise ValueError(f"{name} quality used an unexpected scoring mode")

    reference_passed = reference_quality.get("passed") is True
    intervention_passed = intervention_quality.get("passed") is True
    if gate is None:
        return {
            "mode": expected_mode,
            "valid_reference": reference_passed,
            "intervention_passed": intervention_passed,
            "passed": reference_passed and intervention_passed,
            "metric_drops": {},
        }

    if not ground_truth.expected_answer:
        raise ValueError("reference quality comparison requires an expected answer")
    if any(
        quality.get("calibration_id") != gate.calibration_id
        for quality in (reference_quality, intervention_quality)
    ):
        raise ValueError("response quality used the wrong calibration")

    metric_drops = {}
    for metric in ("token_recall", "rouge_l_f1"):
        reference_value = (reference_quality.get("metrics") or {}).get(metric)
        intervention_value = (intervention_quality.get("metrics") or {}).get(metric)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
            for value in (reference_value, intervention_value)
        ):
            raise ValueError(f"quality comparison has invalid {metric}")
        metric_drops[metric] = reference_value - intervention_value

    within_drop = all(
        drop <= gate.maximum_metric_drop for drop in metric_drops.values()
    )
    return {
        "mode": expected_mode,
        "valid_reference": reference_passed,
        "intervention_passed": intervention_passed,
        "passed": reference_passed and intervention_passed and within_drop,
        "metric_drops": metric_drops,
        "maximum_metric_drop": gate.maximum_metric_drop,
    }
