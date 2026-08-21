import json
from unittest import TestCase

from benchmarks.mtrag import (
    MtragContext,
    MtragMessage,
    MtragTask,
    render_mtrag_messages,
)
from benchmarks.mtrag import mtrag_conversation_split
from benchmarks.mtrag_trace import (
    build_mtrag_counterfactual_cases,
    build_mtrag_request_spec,
    build_mtrag_transition_trace,
)
from benchmarks.schema import ReferenceSimilarityGate


# Build one compact multi-turn RAG task for converter tests.
def _task() -> MtragTask:
    return MtragTask(
        task_id="conversation-1<::>2",
        conversation_id="conversation-1",
        turn=2,
        collection="Cloud",
        contexts=(MtragContext("doc-1", "Guide", "London is the destination."),),
        messages=(
            MtragMessage("user", "Where should I travel?"),
            MtragMessage("assistant", "Let me check."),
            MtragMessage("user", "What does the guide say?"),
        ),
        target_text="The guide says London.",
    )


# Build the preceding task in the same natural conversation.
def _previous_task() -> MtragTask:
    return MtragTask(
        task_id="conversation-1<::>1",
        conversation_id="conversation-1",
        turn=1,
        collection="Cloud",
        contexts=(MtragContext("doc-1", "Guide", "London is the destination."),),
        messages=(MtragMessage("user", "Where should I travel?"),),
        target_text="The guide says London.",
    )


# Build the explicitly versioned gate used by the request answer key.
def _gate() -> ReferenceSimilarityGate:
    return ReferenceSimilarityGate(0.5, 0.4, 0.05, "mtrag-qwen-v1")


# Build one audited manifest that exactly describes the two fixture tasks.
def _manifest() -> dict:
    current = _task()
    return {
        "schema_version": 1,
        "selection": "mtrag-audited-counterfactual-pilot",
        "quality_calibration_id": _gate().calibration_id,
        "source_prompt_template_version": 1,
        "source_model": "test-model",
        "split_seed": "cacheselect-mtrag-v1",
        "block_size": 16,
        "max_target_blocks": 1,
        "transition_count": 1,
        "total_testable_blocks": 1,
        "total_target_blocks": 1,
        "transitions": [
            {
                "split": mtrag_conversation_split(current.conversation_id).value,
                "conversation_id": current.conversation_id,
                "collection": current.collection,
                "previous_task_id": _previous_task().task_id,
                "current_task_id": current.task_id,
                "shared_document_ids": ["doc-1"],
                "aligned_candidate_block_indices": [2, 3],
                "testable_block_indices": [2],
                "target_block_indices": [2],
                "excluded_output_block_index": 3,
            }
        ],
    }


class MtragTraceTests(TestCase):
    # Keep request rendering identical to the earlier CPU coverage analysis.
    def test_builds_exact_rendered_request(self):
        task = _task()

        request = build_mtrag_request_spec(task, quality_gate=_gate())

        self.assertEqual(request.messages, render_mtrag_messages(task))
        self.assertEqual(request.request_id, task.task_id)
        self.assertEqual(request.sequence_index, 2)
        self.assertEqual(
            request.extra_body,
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    # Preserve stable history/document identities as evaluation-only segments.
    def test_builds_ordered_audit_segments(self):
        request = build_mtrag_request_spec(_task(), quality_gate=_gate())

        self.assertEqual(
            [segment.segment_id for segment in request.segments],
            ["system", "message:0", "message:1", "document:doc-1", "message:2"],
        )
        self.assertEqual(request.segments[3].kind, "retrieved_document")
        self.assertEqual(request.segments[-1].version, 2)

    # Keep the reference answer in the ledger but out of the model API payload.
    def test_ground_truth_never_enters_payload(self):
        request = build_mtrag_request_spec(_task(), quality_gate=_gate())

        payload = request.api_payload("model", 128)
        serialized = json.dumps(payload)

        self.assertEqual(request.ground_truth.expected_answer, "The guide says London.")
        self.assertNotIn("The guide says London", serialized)
        self.assertNotIn("ground_truth", serialized)

    # Pair adjacent tasks without inventing synthetic dependency labels.
    def test_builds_natural_transition(self):
        trace = build_mtrag_transition_trace(
            _previous_task(),
            _task(),
            quality_gate=_gate(),
        )

        transition = trace.transitions[0]
        self.assertEqual(len(trace.requests), 2)
        self.assertEqual(transition.previous_request_id, "conversation-1<::>1")
        self.assertEqual(transition.current_request_id, "conversation-1<::>2")
        self.assertEqual(
            transition.ground_truth.change_type,
            "natural_adjacent_rag_turn",
        )
        self.assertEqual(transition.ground_truth.dependent_segment_ids, [])
        self.assertIn("message:0", transition.ground_truth.changed_segment_ids)

    # Reject task pairs whose relationship is not one adjacent session edit.
    def test_rejects_non_adjacent_transition(self):
        current = _task()
        skipped = MtragTask(
            task_id=current.task_id,
            conversation_id=current.conversation_id,
            turn=3,
            collection=current.collection,
            contexts=current.contexts,
            messages=current.messages,
            target_text=current.target_text,
        )

        with self.assertRaisesRegex(ValueError, "consecutive"):
            build_mtrag_transition_trace(
                _previous_task(),
                skipped,
                quality_gate=_gate(),
            )

    # Resolve a frozen plan independently of raw JSONL row ordering.
    def test_builds_manifest_validated_case(self):
        cases = build_mtrag_counterfactual_cases(
            [_task(), _previous_task()],
            _manifest(),
            quality_gate=_gate(),
            approved_task_ids=frozenset({_task().task_id}),
            expected_model="test-model",
        )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].expected_candidate_block_indices, (2, 3))
        self.assertEqual(cases[0].expected_testable_block_indices, (2,))
        self.assertEqual(cases[0].target_block_indices, (2,))

    # Reject a frozen case whose current answer was not manually approved.
    def test_rejects_unapproved_reference_task(self):
        with self.assertRaisesRegex(ValueError, "approval"):
            build_mtrag_counterfactual_cases(
                [_previous_task(), _task()],
                _manifest(),
                quality_gate=_gate(),
                approved_task_ids=frozenset({"different-task"}),
                expected_model="test-model",
            )
