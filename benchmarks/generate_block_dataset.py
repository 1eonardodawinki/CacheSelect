"""Generate a labelled candidate-block dataset from controlled dependencies."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from benchmarks.block_dataset import (
    DatasetSplit,
    SplitCandidateBlock,
    assign_trace_splits,
    build_trace_dependency_examples,
    save_block_dataset_csv,
)
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


@dataclass(frozen=True)
class PointerVariant:
    """One wording family whose selector changes between adjacent requests."""

    variant_id: str
    split: DatasetSplit
    left_key: str
    right_key: str
    left_value: str
    right_value: str
    pointer_template: str
    fact_template: str
    question: str


POINTER_VARIANTS = (
    PointerVariant(
        "version",
        DatasetSplit.TRAIN,
        "A",
        "B",
        "NORTH-731",
        "SOUTH-913",
        "The active version is {key}.",
        "Version {key} maps to code {value}.",
        "Return the code mapped to the active version.",
    ),
    PointerVariant(
        "record",
        DatasetSplit.TRAIN,
        "ALPHA",
        "BETA",
        "EMBER-214",
        "FROST-806",
        "Use record {key}.",
        "Record {key} contains code {value}.",
        "Return the code contained in the selected record.",
    ),
    PointerVariant(
        "route",
        DatasetSplit.TRAIN,
        "EAST",
        "WEST",
        "PORT-441",
        "PORT-772",
        "The selected route is {key}.",
        "Route {key} leads to destination code {value}.",
        "Return the destination code for the selected route.",
    ),
    PointerVariant(
        "tier",
        DatasetSplit.TRAIN,
        "GOLD",
        "SILVER",
        "LIMIT-900",
        "LIMIT-450",
        "Apply service tier {key}.",
        "Service tier {key} permits code {value}.",
        "Return the code permitted by the applied service tier.",
    ),
    PointerVariant(
        "profile",
        DatasetSplit.VALIDATION,
        "ORBIT",
        "NOVA",
        "MODE-318",
        "MODE-624",
        "The enabled profile is {key}.",
        "Profile {key} resolves to mode code {value}.",
        "Return the mode code for the enabled profile.",
    ),
    PointerVariant(
        "channel",
        DatasetSplit.TEST,
        "RED",
        "BLUE",
        "BAND-107",
        "BAND-509",
        "Listen to channel {key}.",
        "Channel {key} broadcasts band code {value}.",
        "Return the band code broadcast by the selected channel.",
    ),
)


# Produce deterministic neutral text that creates reusable negative blocks.
def _filler(word_count: int) -> str:
    return " ".join(
        FILLER_WORDS[index % len(FILLER_WORDS)] for index in range(word_count)
    )


# Place the changed pointer while retaining stable text on both sides.
def _filler_sides(word_count: int, position: str) -> tuple[str, str]:
    if position not in EDIT_POSITIONS:
        raise ValueError(f"unsupported edit position: {position}")
    before_fraction = {"early": 0.1, "middle": 0.5, "late": 0.8}[position]
    before_count = int(word_count * before_fraction)
    return _filler(before_count), _filler(word_count - before_count)


# Build one source/edit trace with a known distant semantic dependency.
def build_pointer_dependency_trace(
    *,
    variant: PointerVariant,
    filler_word_count: int,
    edit_position: str,
) -> WorkloadTrace:
    if filler_word_count < 1:
        raise ValueError("filler_word_count must be positive")
    before_filler, after_filler = _filler_sides(filler_word_count, edit_position)
    system = "Use only the synthetic mappings and return exactly one code."

    # Build either the donor request or its one-pointer edit.
    def request(*, edited: bool) -> RequestSpec:
        key = variant.right_key if edited else variant.left_key
        expected = variant.right_value if edited else variant.left_value
        pointer = variant.pointer_template.format(key=key)
        left_fact = variant.fact_template.format(
            key=variant.left_key,
            value=variant.left_value,
        )
        right_fact = variant.fact_template.format(
            key=variant.right_key,
            value=variant.right_value,
        )
        record = "\n\n".join(
            (before_filler, pointer, after_filler, left_fact, right_fact)
        )
        request_kind = "edited" if edited else "base"
        return RequestSpec(
            request_id=(
                f"dependency-{variant.variant_id}-{filler_word_count}-"
                f"{edit_position}-{request_kind}"
            ),
            workload="block_dependency",
            sequence_index=int(edited),
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Synthetic record:\n\n{record}\n\n{variant.question}",
                },
            ],
            segments=[
                PromptSegment("system", "system", "instruction", 1, system),
                PromptSegment("before_filler", "user", "document", 1, before_filler),
                PromptSegment("pointer", "user", "pointer", 2 if edited else 1, pointer),
                PromptSegment("after_filler", "user", "document", 1, after_filler),
                PromptSegment("left_fact", "user", "retrieved_fact", 1, left_fact),
                PromptSegment("right_fact", "user", "retrieved_fact", 1, right_fact),
                PromptSegment("query", "user", "query", 1, variant.question),
            ],
            ground_truth=RequestGroundTruth(
                expected_answer=expected,
                requirements=[AnswerRequirement("selected_code", [expected.casefold()])],
                notes="The changed selector chooses one of two unchanged mappings.",
            ),
        )

    base = request(edited=False)
    edited = request(edited=True)
    trace_id = (
        f"dependency-{variant.variant_id}-{filler_word_count}-{edit_position}-v1"
    )
    return WorkloadTrace(
        trace_id=trace_id,
        workload="block_dependency",
        description="A changed selector controls unchanged distant mapping facts.",
        requests=[base, edited],
        transitions=[
            RequestTransition(
                transition_id=f"{trace_id}-base-to-edit",
                previous_request_id=base.request_id,
                current_request_id=edited.request_id,
                ground_truth=TransitionGroundTruth(
                    change_type="pointer_replacement",
                    changed_segment_ids=["pointer"],
                    expected_native_behavior="prefix_hit_until_pointer",
                    notes="The benchmark author controls the selector-to-fact relation.",
                    dependent_segment_ids=["left_fact", "right_fact", "query"],
                ),
            )
        ],
    )


# Generate, family-split, and label the complete controlled dataset matrix.
def generate_pointer_block_dataset(
    *,
    tokenizer: object,
    filler_word_counts: Sequence[int],
    edit_positions: Sequence[str] = EDIT_POSITIONS,
    block_size: int = 16,
) -> tuple[SplitCandidateBlock, ...]:
    examples = []
    validation_trace_ids = set()
    test_trace_ids = set()
    for variant in POINTER_VARIANTS:
        for filler_word_count in filler_word_counts:
            for edit_position in edit_positions:
                trace = build_pointer_dependency_trace(
                    variant=variant,
                    filler_word_count=filler_word_count,
                    edit_position=edit_position,
                )
                trace_examples = build_trace_dependency_examples(
                    trace=trace,
                    tokenizer=tokenizer,
                    block_size=block_size,
                )
                if not trace_examples:
                    continue
                examples.extend(trace_examples)
                if variant.split == DatasetSplit.VALIDATION:
                    validation_trace_ids.add(trace.trace_id)
                elif variant.split == DatasetSplit.TEST:
                    test_trace_ids.add(trace.trace_id)
    return assign_trace_splits(
        examples,
        validation_trace_ids=validation_trace_ids,
        test_trace_ids=test_trace_ids,
    )


# Parse command-line settings and write one model-tokenized CSV dataset.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--filler-words",
        type=int,
        nargs="+",
        default=[64, 256, 1024],
    )
    parser.add_argument(
        "--edit-position",
        choices=EDIT_POSITIONS,
        action="append",
        dest="edit_positions",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = generate_pointer_block_dataset(
        tokenizer=tokenizer,
        filler_word_counts=args.filler_words,
        edit_positions=args.edit_positions or EDIT_POSITIONS,
        block_size=args.block_size,
    )
    save_block_dataset_csv(rows, args.output)
    counts = Counter((row.split.value, row.example.label.decision.value) for row in rows)
    print(f"Saved {len(rows)} candidate blocks to {args.output}")
    for (split, decision), count in sorted(counts.items()):
        print(f"{split} {decision}: {count}")


if __name__ == "__main__":
    main()
