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
BASE_PROJECT_CODE = "NORTH-731"
EDITED_PROJECT_CODE = "SOUTH-913"
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


# Repeat a stable vocabulary to make prompts longer without adding new facts.
def _filler_words(count: int) -> list[str]:
    return [FILLER_WORDS[index % len(FILLER_WORDS)] for index in range(count)]


# Divide filler evenly around the early, middle, and late edit locations.
def _split_filler(count: int) -> tuple[str, str, str]:
    words = _filler_words(count)
    first_boundary = count // 3
    second_boundary = (2 * count) // 3
    return (
        " ".join(words[:first_boundary]),
        " ".join(words[first_boundary:second_boundary]),
        " ".join(words[second_boundary:]),
    )


# Place either an irrelevant marker or an answer-bearing fact at the edit point.
def _record_parts(
    filler_word_count: int,
    *,
    edit_position: str,
    edited: bool,
    answer_sensitive: bool,
) -> list[tuple[str, str, int]]:
    if edit_position not in EDIT_POSITIONS:
        raise ValueError(f"unsupported edit position: {edit_position}")

    early_filler, middle_filler, late_filler = _split_filler(filler_word_count)
    parts: list[tuple[str, str, int]] = []
    for position, filler in zip(
        EDIT_POSITIONS,
        (early_filler, middle_filler, late_filler),
        strict=True,
    ):
        is_changed_segment = edit_position == position
        if answer_sensitive and is_changed_segment:
            code = EDITED_PROJECT_CODE if edited else BASE_PROJECT_CODE
            segment_id = f"{position}_fact"
            content = f"{position.title()} verified project code: {code}."
        else:
            marker = "B" if edited and is_changed_segment else "A"
            segment_id = f"{position}_marker"
            content = f"{position.title()} revision marker: {marker}."
        version = 2 if edited and is_changed_segment else 1
        parts.extend(
            [
                (segment_id, content, version),
                (f"{position}_filler", filler, 1),
            ]
        )
    return parts


# Render the synthetic record and question as a two-message chat request.
def _messages(
    parts: list[tuple[str, str, int]],
    *,
    answer_sensitive: bool,
) -> list[dict[str, str]]:
    record_sections = [content for _, content, _ in parts]
    if not answer_sensitive:
        record_sections.append(CALIBRATION_FACT)
    record = "\n\n".join(record_sections)
    user_content = (
        "Synthetic record [calibration_record]:\n\n"
        f"{record}\n\nQuestion: {CALIBRATION_QUESTION}"
    )
    return [
        {"role": "system", "content": CALIBRATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# Build one source or edited request with its private evaluation answer key.
def _request(
    *,
    edit_position: str,
    edited: bool,
    filler_word_count: int,
    answer_sensitive: bool,
) -> RequestSpec:
    parts = _record_parts(
        filler_word_count,
        edit_position=edit_position,
        edited=edited,
        answer_sensitive=answer_sensitive,
    )
    request_kind = "edited" if edited else "base"
    workload = "quality_stress" if answer_sensitive else "length_calibration"
    expected_code = (
        EDITED_PROJECT_CODE if answer_sensitive and edited else BASE_PROJECT_CODE
    )
    fixed_fact_segments = []
    if not answer_sensitive:
        fixed_fact_segments.append(
            PromptSegment(
                "fact",
                "user",
                "retrieved_fact",
                1,
                CALIBRATION_FACT,
            )
        )
    return RequestSpec(
        request_id=f"{workload}-{edit_position}-{request_kind}",
        workload=workload,
        sequence_index=int(edited),
        messages=_messages(parts, answer_sensitive=answer_sensitive),
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
                    "retrieved_fact"
                    if segment_id.endswith("fact")
                    else "revision_marker"
                    if segment_id.endswith("marker")
                    else "document",
                    version,
                    content,
                )
                for segment_id, content, version in parts
            ],
            *fixed_fact_segments,
            PromptSegment(
                "query",
                "user",
                "query",
                1,
                CALIBRATION_QUESTION,
            ),
        ],
        ground_truth=RequestGroundTruth(
            expected_answer=expected_code,
            requirements=[
                AnswerRequirement("project_code", [expected_code.casefold()]),
            ],
            notes=(
                "The selected fact changes the correct answer."
                if answer_sensitive
                else "The revision marker is deliberately irrelevant to the "
                "answer. The calibration quality gate checks semantic fact "
                "retrieval, not citation-format compliance."
            ),
        ),
    )


# Count both prompts because an edit can change their tokenizer lengths.
def _prompt_counts(
    filler_word_count: int,
    *,
    edit_position: str,
    token_counter: TokenCounter,
    answer_sensitive: bool,
) -> tuple[int, int]:
    base = _request(
        edit_position=edit_position,
        edited=False,
        filler_word_count=filler_word_count,
        answer_sensitive=answer_sensitive,
    )
    edited = _request(
        edit_position=edit_position,
        edited=True,
        filler_word_count=filler_word_count,
        answer_sensitive=answer_sensitive,
    )
    return token_counter(base.messages), token_counter(edited.messages)


# Search for a filler size that brings both rendered prompts near the target.
def _choose_filler_word_count(
    target_prompt_tokens: int,
    *,
    edit_position: str,
    token_counter: TokenCounter,
    answer_sensitive: bool,
) -> tuple[int, tuple[int, int]]:
    if target_prompt_tokens < 1:
        raise ValueError("target_prompt_tokens must be positive")

    empty_counts = _prompt_counts(
        0,
        edit_position=edit_position,
        token_counter=token_counter,
        answer_sensitive=answer_sensitive,
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
                answer_sensitive=answer_sensitive,
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
            answer_sensitive=answer_sensitive,
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
                answer_sensitive=answer_sensitive,
            )
        ),
    )
    return best_count, _prompt_counts(
        best_count,
        edit_position=edit_position,
        token_counter=token_counter,
        answer_sensitive=answer_sensitive,
    )


# Build one calibrated source/edit trace for performance or quality testing.
def build_length_calibration_trace(
    *,
    target_prompt_tokens: int,
    edit_position: str,
    token_counter: TokenCounter,
    tokenizer_name: str,
    tolerance_tokens: int = 16,
    answer_sensitive: bool = False,
) -> WorkloadTrace:
    """Build one cold-donor/edit pair close to a rendered token target."""
    if edit_position not in EDIT_POSITIONS:
        raise ValueError(f"unsupported edit position: {edit_position}")

    filler_word_count, measured_counts = _choose_filler_word_count(
        target_prompt_tokens,
        edit_position=edit_position,
        token_counter=token_counter,
        answer_sensitive=answer_sensitive,
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
        answer_sensitive=answer_sensitive,
    )
    edited = _request(
        edit_position=edit_position,
        edited=True,
        filler_word_count=filler_word_count,
        answer_sensitive=answer_sensitive,
    )
    trace_kind = "quality-stress" if answer_sensitive else "length-calibration"
    changed_segment_id = (
        f"{edit_position}_fact"
        if answer_sensitive
        else f"{edit_position}_marker"
    )
    return WorkloadTrace(
        trace_id=f"{trace_kind}-{target_prompt_tokens}-{edit_position}-v1",
        workload="quality_stress" if answer_sensitive else "length_calibration",
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
                    change_type=(
                        f"{edit_position}_answer_fact_edit"
                        if answer_sensitive
                        else f"{edit_position}_document_edit"
                    ),
                    changed_segment_ids=[changed_segment_id],
                    expected_native_behavior=(
                        f"prefix_hit_until_{changed_segment_id}"
                    ),
                    notes=(
                        "The position-controlled fact and required answer change."
                        if answer_sensitive
                        else "Only one position-controlled marker changes; all "
                        "later tokens and the required answer remain identical."
                    ),
                ),
            )
        ],
    )
