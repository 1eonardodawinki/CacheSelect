from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchmarks.schema import (
    ReferenceSimilarityGate,
    RequestGroundTruth,
    load_trace,
    save_trace,
)
from benchmarks.workloads import build_rag_trace


class ReferenceQualitySchemaTests(TestCase):
    # Preserve the calibrated natural-answer gate through trace serialization.
    def test_reference_gate_round_trips(self):
        trace = build_rag_trace()
        gate = ReferenceSimilarityGate(0.6, 0.5, 0.05, "mtrag-qwen-v1")
        original = trace.requests[0]
        ground_truth = RequestGroundTruth(
            expected_answer=original.ground_truth.expected_answer,
            requirements=[],
            notes=original.ground_truth.notes,
            reference_similarity_gate=gate,
        )
        trace = replace(
            trace,
            requests=[
                replace(original, ground_truth=ground_truth),
                *trace.requests[1:],
            ],
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            save_trace(trace, path)
            loaded = load_trace(path)

        self.assertEqual(
            loaded.requests[0].ground_truth.reference_similarity_gate,
            gate,
        )

    # Continue loading historical traces that predate reference-quality gates.
    def test_existing_trace_round_trips_without_a_gate(self):
        trace = build_rag_trace()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            save_trace(trace, path)
            loaded = load_trace(path)

        self.assertEqual(loaded, trace)

    # Fail before GPU execution when a quality policy is not meaningful.
    def test_rejects_invalid_reference_gate(self):
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            ReferenceSimilarityGate(1.1, 0.5, 0.05, "calibration")
        with self.assertRaisesRegex(ValueError, "calibration ID"):
            ReferenceSimilarityGate(0.5, 0.5, 0.05, "")

        # The type remains explicit for static users of RequestGroundTruth.
        ground_truth = RequestGroundTruth("answer", [], reference_similarity_gate=None)
        self.assertIsNone(ground_truth.reference_similarity_gate)
