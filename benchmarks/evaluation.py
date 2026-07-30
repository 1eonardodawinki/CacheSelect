"""Deterministic response scoring for controlled CacheSelect workloads."""

from __future__ import annotations

import re
from typing import Any

from benchmarks.schema import RequestGroundTruth


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def score_response(
    response_text: str | None,
    ground_truth: RequestGroundTruth,
) -> dict[str, Any]:
    """Score whether every required fact appears in a response."""

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
        "passed": passed_count == total,
        "requirements_passed": passed_count,
        "requirements_total": total,
        "score": passed_count / total if total else 1.0,
        "requirements": requirements,
    }
