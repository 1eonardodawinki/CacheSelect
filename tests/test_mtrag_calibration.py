from dataclasses import replace
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmarks.block_dataset import DatasetSplit
from benchmarks.mtrag_calibration import (
    MTRAG_UNCALIBRATED_GATE_ID,
    MtragReferenceCalibrationCase,
    build_mtrag_reference_calibration_cases,
    run_mtrag_reference_calibration,
)
from benchmarks.mtrag import (
    MtragContext,
    MtragMessage,
    MtragTask,
    mtrag_conversation_split,
)
from benchmarks.schema import ReferenceSimilarityGate
from benchmarks.workloads import build_rag_trace


# Build one case whose current request opts into the provisional quality gate.
def _case() -> MtragReferenceCalibrationCase:
    trace = build_rag_trace()
    gate = ReferenceSimilarityGate(0.0, 0.0, 1.0, MTRAG_UNCALIBRATED_GATE_ID)
    current = trace.requests[1]
    current = replace(
        current,
        ground_truth=replace(
            current.ground_truth,
            requirements=[],
            reference_similarity_gate=gate,
        ),
    )
    return MtragReferenceCalibrationCase(current, DatasetSplit.TRAIN, "Cloud")


# Build one natural task for compact-manifest resolution tests.
def _mtrag_task() -> MtragTask:
    return MtragTask(
        task_id="conversation-1<::>2",
        conversation_id="conversation-1",
        turn=2,
        collection="Cloud",
        contexts=(MtragContext("doc-1", "Guide", "London is documented."),),
        messages=(MtragMessage("user", "What is documented?"),),
        target_text="London is documented.",
    )


# Describe the same task without carrying irrelevant KV block arrays.
def _manifest() -> dict:
    task = _mtrag_task()
    return {
        "schema_version": 1,
        "selection": "mtrag-reference-quality-calibration",
        "source_prompt_template_version": 1,
        "source_model": "test-model",
        "split_seed": "cacheselect-mtrag-v1",
        "task_count": 1,
        "tasks": [
            {
                "task_id": task.task_id,
                "conversation_id": task.conversation_id,
                "collection": task.collection,
                "split": mtrag_conversation_split(task.conversation_id).value,
            }
        ],
    }


class MtragCalibrationTests(TestCase):
    # Resolve the frozen task ID into an exact request with a provisional gate.
    def test_builds_compact_calibration_cases(self):
        gate = ReferenceSimilarityGate(0.0, 0.0, 1.0, MTRAG_UNCALIBRATED_GATE_ID)

        cases = build_mtrag_reference_calibration_cases(
            (_mtrag_task(),),
            _manifest(),
            quality_gate=gate,
            expected_model="test-model",
        )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].request.request_id, _mtrag_task().task_id)
        self.assertEqual(
            cases[0].request.ground_truth.reference_similarity_gate,
            gate,
        )

    # Record one isolated full-compute answer without producing a causal label.
    def test_runs_uncached_reference_requests(self):
        observation = {
            "cached_tokens": 0,
            "runtime_policy": {"policy": "FULL_RECOMPUTE"},
            "server_metrics": {"cacheselect_compacted_batch_executed": False},
            "prompt_token_count": 128,
            "finish_reason": "stop",
            "output_text": "The answer is London.",
            "quality": {
                "mode": "reference_similarity",
                "passed": True,
                "metrics": {"token_recall": 0.8, "rouge_l_f1": 0.7},
            },
        }
        recorder = SimpleNamespace()

        with patch(
            "benchmarks.mtrag_calibration._observe_request",
            return_value=observation,
        ) as observe:
            result = run_mtrag_reference_calibration(
                (_case(),),
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=384,
                api_key=None,
                timeout_seconds=10.0,
                recorder=recorder,
            )

        self.assertEqual(result["request_count"], 1)
        self.assertEqual(result["gate_status"], "unfrozen")
        self.assertEqual(result["rows"][0]["cached_tokens"], 0)
        self.assertEqual(result["rows"][0]["finish_reason"], "stop")
        self.assertEqual(result["rows"][0]["quality"]["metrics"]["token_recall"], 0.8)
        call = observe.call_args
        self.assertIs(call.kwargs["recorder"], recorder)
        self.assertIn("uncached-reference", call.kwargs["cache_salt"])
