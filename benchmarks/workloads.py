"""Self-contained controlled workloads for CacheSelect evaluation."""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean

from benchmarks.schema import (
    AnswerRequirement,
    PromptSegment,
    RequestGroundTruth,
    RequestSpec,
    RequestTransition,
    TransitionGroundTruth,
    WorkloadTrace,
)


def _require(requirement_id: str, *accepted_phrases: str) -> AnswerRequirement:
    return AnswerRequirement(requirement_id, list(accepted_phrases))


def _transition(
    transition_id: str,
    previous: RequestSpec,
    current: RequestSpec,
    *,
    change_type: str,
    changed_segment_ids: list[str],
    expected_native_behavior: str,
    notes: str,
) -> RequestTransition:
    return RequestTransition(
        transition_id=transition_id,
        previous_request_id=previous.request_id,
        current_request_id=current.request_id,
        ground_truth=TransitionGroundTruth(
            change_type=change_type,
            changed_segment_ids=changed_segment_ids,
            expected_native_behavior=expected_native_behavior,
            notes=notes,
        ),
    )


# ---------------------------------------------------------------------------
# RAG: independently reusable documents are replaced and reordered.

RAG_SYSTEM = (
    "Answer using only the retrieved documents. Give the answer followed by "
    "the supporting document ID in square brackets. If the answer is absent, "
    "say INSUFFICIENT."
)

RAG_DOCUMENTS = {
    "doc_solar": (
        "[doc_solar] Northfield's solar array produced 18.4 GWh in 2024. "
        "Maintenance is normally scheduled in February."
    ),
    "doc_battery_2024": (
        "[doc_battery_2024] The 6 MW / 24 MWh battery was commissioned in June "
        "2024. Acceptance testing measured 88 percent round-trip efficiency."
    ),
    "doc_grid": (
        "[doc_grid] The eastern substation has a contracted export limit of "
        "15 MW. Eleven curtailment events occurred during 2024."
    ),
    "doc_battery_2025": (
        "[doc_battery_2025] A January 2025 firmware update changed the battery "
        "operating range. Follow-up testing measured 90 percent round-trip "
        "efficiency; its 6 MW / 24 MWh rating did not change."
    ),
}


def _rag_request(
    index: int,
    document_ids: list[str],
    question: str,
    *,
    query_version: int,
    expected_answer: str,
    requirements: list[AnswerRequirement],
) -> RequestSpec:
    documents = "\n\n".join(RAG_DOCUMENTS[doc_id] for doc_id in document_ids)
    return RequestSpec(
        request_id=f"rag-{index:02d}",
        workload="rag",
        sequence_index=index,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Retrieved documents:\n\n{documents}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        segments=[
            PromptSegment("system", "system", "instruction", 1, RAG_SYSTEM),
            *[
                PromptSegment(
                    doc_id,
                    "user",
                    "retrieved_document",
                    1,
                    RAG_DOCUMENTS[doc_id],
                )
                for doc_id in document_ids
            ],
            PromptSegment("query", "user", "query", query_version, question),
        ],
        ground_truth=RequestGroundTruth(
            expected_answer=expected_answer,
            requirements=requirements,
        ),
    )


def build_rag_trace() -> WorkloadTrace:
    """RAG replacement, reordering and query-change cases."""

    efficiency_question = "What round-trip efficiency did battery testing report?"
    requests = [
        _rag_request(
            0,
            ["doc_solar", "doc_battery_2024", "doc_grid"],
            efficiency_question,
            query_version=1,
            expected_answer="88 percent [doc_battery_2024]",
            requirements=[
                _require("efficiency", "88 percent", "88%"),
                _require("citation", "doc_battery_2024"),
            ],
        ),
        _rag_request(
            1,
            ["doc_solar", "doc_battery_2025", "doc_grid"],
            efficiency_question,
            query_version=1,
            expected_answer="90 percent [doc_battery_2025]",
            requirements=[
                _require("efficiency", "90 percent", "90%"),
                _require("citation", "doc_battery_2025"),
            ],
        ),
        _rag_request(
            2,
            ["doc_grid", "doc_solar", "doc_battery_2025"],
            efficiency_question,
            query_version=1,
            expected_answer="90 percent [doc_battery_2025]",
            requirements=[
                _require("efficiency", "90 percent", "90%"),
                _require("citation", "doc_battery_2025"),
            ],
        ),
        _rag_request(
            3,
            ["doc_grid", "doc_solar", "doc_battery_2025"],
            "What is the eastern substation's contracted export limit?",
            query_version=2,
            expected_answer="15 MW [doc_grid]",
            requirements=[
                _require("export_limit", "15 mw"),
                _require("citation", "doc_grid"),
            ],
        ),
    ]
    return WorkloadTrace(
        trace_id="rag-controlled-v2",
        workload="rag",
        description="Controlled RAG document replacement, reorder and query edit.",
        requests=requests,
        transitions=[
            _transition(
                "rag-00-to-01",
                requests[0],
                requests[1],
                change_type="document_replacement",
                changed_segment_ids=[
                    "doc_battery_2024",
                    "doc_battery_2025",
                ],
                expected_native_behavior="prefix_hit_until_replaced_document",
                notes="The answer changes because the retrieved battery report changes.",
            ),
            _transition(
                "rag-01-to-02",
                requests[1],
                requests[2],
                change_type="document_reorder",
                changed_segment_ids=[
                    "doc_grid",
                    "doc_solar",
                    "doc_battery_2025",
                ],
                expected_native_behavior="prefix_miss_at_first_reordered_document",
                notes="Document contents are identical but their positions change.",
            ),
            _transition(
                "rag-02-to-03",
                requests[2],
                requests[3],
                change_type="query_replacement",
                changed_segment_ids=["query"],
                expected_native_behavior="prefix_hit_through_all_documents",
                notes="Only the final query changes.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Chat: fixed conversation replay gives comparable prompts across policies.

CHAT_SYSTEM = (
    "Maintain the user's current trip constraints. When asked for a summary, "
    "state every active constraint concisely."
)


def _chat_ground_truth(
    expected_answer: str,
    *requirements: AnswerRequirement,
) -> RequestGroundTruth:
    return RequestGroundTruth(expected_answer, list(requirements))


def _chat_request(
    index: int,
    messages: list[dict[str, str]],
    ground_truth: RequestGroundTruth,
    *,
    edited_first_turn: bool = False,
) -> RequestSpec:
    segments = []
    for message_index, message in enumerate(messages):
        if message_index == 0:
            segment_id = "system"
            kind = "instruction"
            version = 1
        else:
            segment_id = f"turn_{message_index - 1:02d}"
            kind = "conversation_turn"
            version = 2 if edited_first_turn and message_index == 1 else 1
        segments.append(
            PromptSegment(
                segment_id,
                message["role"],
                kind,
                version,
                message["content"],
            )
        )
    return RequestSpec(
        request_id=f"chat-{index:02d}",
        workload="chat",
        sequence_index=index,
        messages=messages,
        segments=segments,
        ground_truth=ground_truth,
    )


def build_chat_trace() -> WorkloadTrace:
    """Append-only chat controls followed by an early history edit."""

    turns = [
        {
            "role": "user",
            "content": (
                "Plan a two-day Edinburgh visit. I am vegetarian. "
                "Summarise my constraints."
            ),
        },
        {
            "role": "assistant",
            "content": "Your trip is two days and all food must be vegetarian.",
        },
        {
            "role": "user",
            "content": (
                "I arrive at Waverley at 10:00 on Saturday. "
                "Summarise all constraints."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The trip is two days, food is vegetarian, and it starts at "
                "Waverley at 10:00 on Saturday."
            ),
        },
        {
            "role": "user",
            "content": (
                "Include the National Museum as an indoor option. "
                "Summarise all constraints."
            ),
        },
    ]
    message_sets = [
        [{"role": "system", "content": CHAT_SYSTEM}, turns[0]],
        [{"role": "system", "content": CHAT_SYSTEM}, *turns[:3]],
        [{"role": "system", "content": CHAT_SYSTEM}, *turns],
    ]
    edited_turns = [dict(turn) for turn in turns]
    edited_turns[0] = {
        "role": "user",
        "content": (
            "Plan a two-day Edinburgh visit. I am vegan. "
            "Summarise my constraints."
        ),
    }
    edited_turns[1] = {
        "role": "assistant",
        "content": "Your trip is two days and all food must be vegan.",
    }
    message_sets.append(
        [{"role": "system", "content": CHAT_SYSTEM}, *edited_turns]
    )

    requests = [
        _chat_request(
            0,
            message_sets[0],
            _chat_ground_truth(
                "Two days; vegetarian.",
                _require("duration", "two-day", "two day", "2-day", "2 day"),
                _require("diet", "vegetarian"),
            ),
        ),
        _chat_request(
            1,
            message_sets[1],
            _chat_ground_truth(
                "Two days; vegetarian; Waverley at 10:00 Saturday.",
                _require("duration", "two-day", "two day", "2-day", "2 day"),
                _require("diet", "vegetarian"),
                _require("arrival_place", "waverley"),
                _require("arrival_time", "10:00", "10 am", "10am"),
            ),
        ),
        _chat_request(
            2,
            message_sets[2],
            _chat_ground_truth(
                "Two days; vegetarian; Waverley 10:00 Saturday; National Museum.",
                _require("duration", "two-day", "two day", "2-day", "2 day"),
                _require("diet", "vegetarian"),
                _require("arrival_place", "waverley"),
                _require("arrival_time", "10:00", "10 am", "10am"),
                _require("indoor_option", "national museum"),
            ),
        ),
        _chat_request(
            3,
            message_sets[3],
            _chat_ground_truth(
                "Two days; vegan; Waverley 10:00 Saturday; National Museum.",
                _require("duration", "two-day", "two day", "2-day", "2 day"),
                _require("diet", "vegan"),
                _require("arrival_place", "waverley"),
                _require("arrival_time", "10:00", "10 am", "10am"),
                _require("indoor_option", "national museum"),
            ),
            edited_first_turn=True,
        ),
    ]
    return WorkloadTrace(
        trace_id="chat-controlled-v2",
        workload="chat",
        description="Fixed chat replay with append controls and an early edit.",
        requests=requests,
        transitions=[
            _transition(
                "chat-00-to-01",
                requests[0],
                requests[1],
                change_type="append_turn",
                changed_segment_ids=["turn_01", "turn_02"],
                expected_native_behavior="exact_prefix_reuse",
                notes="Native vLLM prefix caching should already handle this.",
            ),
            _transition(
                "chat-01-to-02",
                requests[1],
                requests[2],
                change_type="append_turn",
                changed_segment_ids=["turn_03", "turn_04"],
                expected_native_behavior="exact_prefix_reuse",
                notes="A second positive control for native prefix caching.",
            ),
            _transition(
                "chat-02-to-03",
                requests[2],
                requests[3],
                change_type="edit_history",
                changed_segment_ids=["turn_00", "turn_01"],
                expected_native_behavior="prefix_hit_until_edited_turn",
                notes="The early diet edit must propagate to the final answer.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Periodic agent: deterministic sliding telemetry windows.

PERIODIC_SYSTEM = (
    "You are a periodic infrastructure Data Analyst. Read the complete telemetry "
    "window. Return exactly one compact JSON object with keys mean_cpu, max_cpu, "
    "max_timestamp, and alert. Round mean_cpu to one decimal. alert is true when "
    "any CPU value is at least 80, otherwise false."
)

TELEMETRY = [
    ("2026-07-30T09:00:00Z", 42),
    ("2026-07-30T09:05:00Z", 44),
    ("2026-07-30T09:10:00Z", 43),
    ("2026-07-30T09:15:00Z", 45),
    ("2026-07-30T09:20:00Z", 47),
    ("2026-07-30T09:25:00Z", 46),
    ("2026-07-30T09:30:00Z", 48),
    ("2026-07-30T09:35:00Z", 87),
    ("2026-07-30T09:40:00Z", 49),
    ("2026-07-30T09:45:00Z", 50),
    ("2026-07-30T09:50:00Z", 51),
]


def _periodic_request(
    index: int,
    window: list[tuple[str, int]],
) -> RequestSpec:
    rows = [f"{timestamp},{cpu}" for timestamp, cpu in window]
    values = [cpu for _, cpu in window]
    mean_cpu = round(fmean(values), 1)
    max_cpu = max(values)
    max_timestamp = next(
        timestamp for timestamp, cpu in window if cpu == max_cpu
    )
    alert = max_cpu >= 80
    expected = (
        f'{{"mean_cpu":{mean_cpu:.1f},"max_cpu":{max_cpu},'
        f'"max_timestamp":"{max_timestamp}",'
        f'"alert":{str(alert).lower()}}}'
    )
    user_content = (
        "Telemetry CSV:\ntimestamp,cpu_percent\n"
        + "\n".join(rows)
        + "\n\nAnalyse this window now."
    )
    return RequestSpec(
        request_id=f"periodic-{index:02d}",
        workload="periodic_agent",
        sequence_index=index,
        messages=[
            {"role": "system", "content": PERIODIC_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        segments=[
            PromptSegment(
                "system",
                "system",
                "instruction",
                1,
                PERIODIC_SYSTEM,
            ),
            PromptSegment(
                "telemetry_schema",
                "user",
                "schema",
                1,
                "timestamp,cpu_percent",
            ),
            *[
                PromptSegment(
                    f"row_{timestamp}",
                    "user",
                    "telemetry_row",
                    1,
                    row,
                )
                for (timestamp, _), row in zip(window, rows, strict=True)
            ],
            PromptSegment(
                "query",
                "user",
                "query",
                1,
                "Analyse this window now.",
            ),
        ],
        ground_truth=RequestGroundTruth(
            expected_answer=expected,
            requirements=[
                _require("mean_cpu", f"{mean_cpu:.1f}"),
                _require("max_cpu", str(max_cpu)),
                _require("max_timestamp", max_timestamp),
                _require("alert", str(alert).lower()),
            ],
            notes="Values are computed directly from the synthetic CSV window.",
        ),
    )


def build_periodic_agent_trace(
    count: int = 6,
    window_size: int = 6,
) -> WorkloadTrace:
    """Build a periodic Data Analyst with one-row sliding windows."""

    if count < 2:
        raise ValueError("count must be at least 2")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if count + window_size - 1 > len(TELEMETRY):
        raise ValueError("count and window_size exceed available telemetry")

    requests = [
        _periodic_request(index, TELEMETRY[index : index + window_size])
        for index in range(count)
    ]
    transitions = []
    for index in range(1, len(requests)):
        departed_timestamp = TELEMETRY[index - 1][0]
        entered_timestamp = TELEMETRY[index + window_size - 1][0]
        transitions.append(
            _transition(
                f"periodic-{index - 1:02d}-to-{index:02d}",
                requests[index - 1],
                requests[index],
                change_type="sliding_window",
                changed_segment_ids=[
                    f"row_{departed_timestamp}",
                    f"row_{entered_timestamp}",
                ],
                expected_native_behavior="prefix_hit_only_through_stable_header",
                notes=(
                    "Most rows persist semantically, but shifting their token "
                    "positions defeats ordinary exact-prefix reuse."
                ),
            )
        )
    return WorkloadTrace(
        trace_id="periodic-data-analyst-v2",
        workload="periodic_agent",
        description="Periodic Data Analyst over deterministic sliding telemetry.",
        requests=requests,
        transitions=transitions,
    )


TRACE_BUILDERS: dict[str, Callable[[], WorkloadTrace]] = {
    "rag": build_rag_trace,
    "chat": build_chat_trace,
    "periodic_agent": build_periodic_agent_trace,
}
