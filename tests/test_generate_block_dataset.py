from unittest import TestCase

from benchmarks.block_dataset import DatasetSplit, RepairDecision
from benchmarks.generate_block_dataset import (
    POINTER_VARIANTS,
    build_pointer_dependency_trace,
    generate_pointer_block_dataset,
)


class WordTokenizer:
    # Create one vocabulary shared by every generated prompt.
    def __init__(self):
        self.vocabulary = {}

    # Render controlled messages with stable separators or return their IDs.
    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text = "\n".join(message["content"] for message in messages)
        return self._encode(text)[0] if tokenize else text

    # Return word IDs together with character offsets.
    def __call__(self, text, **kwargs):
        token_ids, offsets = self._encode(text)
        return {"input_ids": token_ids, "offset_mapping": offsets}

    # Encode whitespace-separated words without external model dependencies.
    def _encode(self, text):
        token_ids = []
        offsets = []
        cursor = 0
        for word in text.split():
            start = text.index(word, cursor)
            end = start + len(word)
            cursor = end
            token_ids.append(self.vocabulary.setdefault(word, len(self.vocabulary) + 1))
            offsets.append((start, end))
        return token_ids, offsets


class BlockDatasetGenerationTests(TestCase):
    # Verify one trace changes only its selector and records distant dependents.
    def test_builds_controlled_pointer_trace(self):
        trace = build_pointer_dependency_trace(
            variant=POINTER_VARIANTS[0],
            filler_word_count=32,
            edit_position="middle",
        )
        base, edited = trace.requests

        self.assertNotEqual(base.messages, edited.messages)
        self.assertEqual(base.ground_truth.expected_answer, "NORTH-731")
        self.assertEqual(edited.ground_truth.expected_answer, "SOUTH-913")
        self.assertEqual(
            trace.transitions[0].ground_truth.dependent_segment_ids,
            ["left_fact", "right_fact", "query"],
        )

    # Verify dependent mapping facts move relative to stable neutral text.
    def test_varies_dependency_layout(self):
        facts_early = build_pointer_dependency_trace(
            variant=POINTER_VARIANTS[0],
            filler_word_count=64,
            edit_position="early",
            dependency_layout="facts_early",
        ).requests[1]
        facts_late = build_pointer_dependency_trace(
            variant=POINTER_VARIANTS[0],
            filler_word_count=64,
            edit_position="early",
            dependency_layout="facts_late",
        ).requests[1]
        early_text = facts_early.messages[1]["content"]
        late_text = facts_late.messages[1]["content"]
        early_pointer = early_text.index("The active version")
        late_pointer = late_text.index("The active version")
        early_suffix_filler = early_text.index("archive", early_pointer)
        late_suffix_filler = late_text.index("archive", late_pointer)

        self.assertLess(early_text.index("Version A"), early_suffix_filler)
        self.assertGreater(late_text.index("Version A"), late_suffix_filler)

    # Verify held-out wording families and both decisions reach the dataset.
    def test_generates_family_level_splits(self):
        rows = generate_pointer_block_dataset(
            tokenizer=WordTokenizer(),
            filler_word_counts=[32],
            edit_positions=["middle"],
            dependency_layouts=["facts_middle"],
            block_size=4,
            include_context_features=True,
        )

        self.assertEqual(
            {row.split for row in rows},
            {DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST},
        )
        self.assertEqual(
            {row.example.label.decision for row in rows},
            {RepairDecision.REPAIR, RepairDecision.REUSE},
        )
        self.assertTrue(all(row.example.context_features is not None for row in rows))
        split_by_variant = {
            row.example.trace_id.split("-")[1]: row.split for row in rows
        }
        self.assertEqual(split_by_variant["profile"], DatasetSplit.VALIDATION)
        self.assertEqual(split_by_variant["channel"], DatasetSplit.TEST)
