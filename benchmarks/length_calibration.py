"""Tokenizer-aware synthetic traces for prompt-length calibration."""

from __future__ import annotations

from collections.abc import Callable

from benchmarks.schema import (
    AnswerRequirement,
    PromptSegment,
    RequestGroundTruth,
    RequestSpec,
    RequestTransition,
    TransitionGroundTruth,
    WorkloadTrace,
)

EDIT_POSITIONS = ("early", "middle", "late")

CALIBRATION_SYSTEM = (
    "Answer using only the supplied synthetic record. Return the project code "
    "followed by [calibration_record]."
)
CALIBRATION_FACT = "The verified project code is NORTH-731."
CALIBRATION_QUESTION = "What is the verified project code?"
FILLER_WORDS = (
    "archive",
    "sensor",
    "record",
    "remained",
    "stable",
    "during",
    "routine",
    "inspection",
)

TokenCounter = Callable[[list[dict[str, str]]], int]


def _filler_words(count: int) -> list[str]:
    return [FILLER_WORDS[index % len(FILLER_WORDS)] for index in range(count)]


def _split_filler(count: int) -> tuple[str, str, str]:
    words = _filler_words(count)
    first_boundary = count // 3
    second_boundary = (2 * count) // 3
    return (
        " ".join(words[:first_boundary]),
        " ".join(words[first_boundary:second_boundary]),
        " ".join(words[second_boundary:]),
    )


def _record_parts(
    filler_word_count: int,
    *,
    edit_position: str | None,
) -> list[tuple[str, str, int]]:
    if edit_position is not None and edit_position not in EDIT_POSITIONS:
        raise ValueError(f"unsupported edit position: {edit_position}")

    early_filler, middle_filler, late_filler = _split_filler(filler_word_count)
    parts: list[tuple[str, str, int]] = []
    for position, filler in zip(
        EDIT_POSITIONS,
        (early_filler, middle_filler, late_filler),
        strict=True,
    ):
        marker = "B" if edit_position == position else "A"
        version = 2 if edit_position == position else 1
        parts.extend(
            [
                (
                    f"{position}_marker",
                    f"{position.title()} revision marker: {marker}.",
                    version,
                ),
                (f"{position}_filler", filler, 1),
            ]
        )
    return parts


def _messages(parts: list[tuple[str, str, int]]) -> list[dict[str, str]]:
    record = "\n\n".join(content for _, content, _ in parts)
    user_content = (
        "Synthetic record [calibration_record]:\n\n"
        f"{record}\n\n{CALIBRATION_FACT}\n\n"
        f"Question: {CALIBRATION_QUESTION}"
    )
    return [
        {"role": "system", "content": CALIBRATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _request(
    *,
    edit_position: str,
    edited: bool,
    filler_word_count: int,
) -> RequestSpec:
    applied_edit = edit_position if edited else None
    parts = _record_parts(filler_word_count, edit_position=applied_edit)
    request_kind = "edited" if edited else "base"
    return RequestSpec(
        request_id=f"calibration-{edit_position}-{request_kind}",
        workload="length_calibration",
        sequence_index=int(edited),
        messages=_messages(parts),
        segments=[
            PromptSegment(
                "system",
                "system",
                "instruction",
                1,
                CALIBRATION_SYSTEM,
            ),
            *[
                PromptSegment(
                    segment_id,
                    "user",
                    "revision_marker" if segment_id.endswith("marker") else "document",
                    version,
                    content,
                )
                for segment_id, content, version in parts
            ],
            PromptSegment(
                "fact",
                "user",
                "retrieved_fact",
                1,
                CALIBRATION_FACT,
            ),
            PromptSegment(
                "query",
                "user",
                "query",
                1,
                CALIBRATION_QUESTION,
            ),
        ],
        ground_truth=RequestGroundTruth(
            expected_answer="NORTH-731",
            requirements=[
                AnswerRequirement("project_code", ["north-731"]),
            ],
            notes=(
                "The revision marker is deliberately irrelevant to the answer. "
                "The calibration quality gate checks semantic fact retrieval, "
                "not citation-format compliance."
            ),
        ),
    )


def _prompt_counts(
    filler_word_count: int,
    *,
    edit_position: str,
    token_counter: TokenCounter,
) -> tuple[int, int]:
    base = _request(
        edit_position=edit_position,
        edited=False,
        filler_word_count=filler_word_count,
    )
    edited = _request(
        edit_position=edit_position,
        edited=True,
        filler_word_count=filler_word_count,
    )
    return token_counter(base.messages), token_counter(edited.messages)


def _choose_filler_word_count(
    target_prompt_tokens: int,
    *,
    edit_position: str,
    token_counter: TokenCounter,
) -> tuple[int, tuple[int, int]]:
    if target_prompt_tokens < 1:
        raise ValueError("target_prompt_tokens must be positive")

    empty_counts = _prompt_counts(
        0,
        edit_position=edit_position,
        token_counter=token_counter,
    )
    if max(empty_counts) > target_prompt_tokens:
        raise ValueError(
            f"target {target_prompt_tokens} is smaller than the base prompt "
            f"({max(empty_counts)} tokens)"
        )

    lower = 0
    upper = max(1, target_prompt_tokens)
    while (
        max(
            _prompt_counts(
                upper,
                edit_position=edit_position,
                token_counter=token_counter,
            )
        )
        < target_prompt_tokens
    ):
        upper *= 2
        if upper > target_prompt_tokens * 16:
            raise RuntimeError(
                "token counter did not grow with deterministic filler text"
            )

    while lower + 1 < upper:
        middle = (lower + upper) // 2
        counts = _prompt_counts(
            middle,
            edit_position=edit_position,
            token_counter=token_counter,
        )
        if max(counts) < target_prompt_tokens:
            lower = middle
        else:
            upper = middle

    candidates = range(max(0, lower - 8), upper + 9)
    best_count = min(
        candidates,
        key=lambda count: max(
            abs(tokens - target_prompt_tokens)
            for tokens in _prompt_counts(
                count,
                edit_position=edit_position,
                token_counter=token_counter,
            )
        ),
    )
    return best_count, _prompt_counts(
        best_count,
        edit_position=edit_position,
        token_counter=token_counter,
    )


def build_length_calibration_trace(
    *,
    target_prompt_tokens: int,
    edit_position: str,
    token_counter: TokenCounter,
    tokenizer_name: str,
    tolerance_tokens: int = 16,
) -> WorkloadTrace:
    """Build one cold-donor/edit pair close to a rendered token target."""
    if edit_position not in EDIT_POSITIONS:
        raise ValueError(f"unsupported edit position: {edit_position}")

    filler_word_count, measured_counts = _choose_filler_word_count(
        target_prompt_tokens,
        edit_position=edit_position,
        token_counter=token_counter,
    )
    if (
        max(abs(count - target_prompt_tokens) for count in measured_counts)
        > tolerance_tokens
    ):
        raise RuntimeError(
            f"could not approach {target_prompt_tokens} tokens within "
            f"{tolerance_tokens}: measured {measured_counts}"
        )

    base = _request(
        edit_position=edit_position,
        edited=False,
        filler_word_count=filler_word_count,
    )
    edited = _request(
        edit_position=edit_position,
        edited=True,
        filler_word_count=filler_word_count,
    )
    return WorkloadTrace(
        trace_id=f"length-calibration-{target_prompt_tokens}-{edit_position}-v1",
        workload="length_calibration",
        description=(
            f"Tokenizer-aware {edit_position} edit calibration targeting "
            f"{target_prompt_tokens} rendered prompt tokens; generated with "
            f"{tokenizer_name}; base/edit counts={measured_counts}."
        ),
        requests=[base, edited],
        transitions=[
            RequestTransition(
                transition_id=f"calibration-{edit_position}-base-to-edit",
                previous_request_id=base.request_id,
                current_request_id=edited.request_id,
                ground_truth=TransitionGroundTruth(
                    change_type=f"{edit_position}_document_edit",
                    changed_segment_ids=[f"{edit_position}_marker"],
                    expected_native_behavior=(
                        f"prefix_hit_until_{edit_position}_marker"
                    ),
                    notes=(
                        "Only one position-controlled marker changes; all later "
                        "tokens and the required answer remain identical."
                    ),
                ),
            )
        ],
    )
