"""Serializable benchmark-only request and transition structures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptSegment:
    """Benchmark ground truth for one logical part of a request."""

    segment_id: str
    role: str
    kind: str
    version: int
    content: str


@dataclass(frozen=True)
class AnswerRequirement:
    """One fact that must be present in a model response."""

    requirement_id: str
    accepted_phrases: list[str]


@dataclass(frozen=True)
class RequestGroundTruth:
    """Evaluation-only answer key that is never added to the API payload."""

    expected_answer: str
    requirements: list[AnswerRequirement]
    notes: str = ""


@dataclass(frozen=True)
class RequestSpec:
    """One OpenAI-compatible chat request before model-specific rendering."""

    request_id: str
    workload: str
    sequence_index: int
    messages: list[dict[str, Any]]
    segments: list[PromptSegment]
    ground_truth: RequestGroundTruth
    extra_body: dict[str, Any] = field(default_factory=dict)

    def api_payload(
        self,
        model: str,
        max_completion_tokens: int,
    ) -> dict[str, Any]:
        """Build the payload sent to the vLLM Chat Completions endpoint."""
        return {
            "model": model,
            "messages": self.messages,
            "request_id": self.request_id,
            "temperature": 0.0,
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
            "return_token_ids": True,
            "return_prompt_text": True,
            **self.extra_body,
        }


@dataclass(frozen=True)
class TransitionGroundTruth:
    """Known relationship between two synthetically controlled requests."""

    change_type: str
    changed_segment_ids: list[str]
    expected_native_behavior: str
    notes: str = ""


@dataclass(frozen=True)
class RequestTransition:
    """A directed transition between two requests in a workload trace."""

    transition_id: str
    previous_request_id: str
    current_request_id: str
    ground_truth: TransitionGroundTruth


@dataclass(frozen=True)
class WorkloadTrace:
    """An ordered request sequence and its adjacent transition labels."""

    trace_id: str
    workload: str
    description: str
    requests: list[RequestSpec]
    transitions: list[RequestTransition]


def save_trace(trace: WorkloadTrace, path: Path) -> None:
    """Write a trace as human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(trace), indent=2) + "\n")


def load_trace(path: Path) -> WorkloadTrace:
    """Load a trace produced by :func:`save_trace`."""
    data = json.loads(path.read_text())
    requests = [
        RequestSpec(
            request_id=request["request_id"],
            workload=request["workload"],
            sequence_index=request["sequence_index"],
            messages=request["messages"],
            segments=[
                PromptSegment(**segment) for segment in request["segments"]
            ],
            ground_truth=RequestGroundTruth(
                expected_answer=request["ground_truth"]["expected_answer"],
                requirements=[
                    AnswerRequirement(**requirement)
                    for requirement in request["ground_truth"]["requirements"]
                ],
                notes=request["ground_truth"].get("notes", ""),
            ),
            extra_body=request.get("extra_body", {}),
        )
        for request in data["requests"]
    ]
    transitions = [
        RequestTransition(
            transition_id=transition["transition_id"],
            previous_request_id=transition["previous_request_id"],
            current_request_id=transition["current_request_id"],
            ground_truth=TransitionGroundTruth(**transition["ground_truth"]),
        )
        for transition in data["transitions"]
    ]
    return WorkloadTrace(
        trace_id=data["trace_id"],
        workload=data["workload"],
        description=data["description"],
        requests=requests,
        transitions=transitions,
    )
