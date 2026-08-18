# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from unittest.mock import Mock

import torch

from vllm.model_executor.layers.mamba.gdn.delta_cache import (
    GDNDeltaOperatorSidecar,
)
from vllm.model_executor.layers.mamba.gdn.delta_execution import (
    assess_gdn_delta_active_admission,
    build_gdn_delta_execution_plans,
    execute_gdn_delta_sequence_plan,
)


# Build one small CPU sidecar used by the sequential execution tests.
def _sidecar(*keys: bytes) -> GDNDeltaOperatorSidecar:
    sidecar = GDNDeltaOperatorSidecar(
        capacity=max(len(keys), 1),
        block_size=2,
        value_heads=1,
        key_width=1,
        value_width=1,
        dtype=torch.float32,
        device="cpu",
    )
    for key in keys:
        sidecar.store(
            key,
            torch.ones(1, 1, 1),
            torch.ones(2, 1, 1),
            torch.full((1, 1, 1), 10.0),
            torch.zeros(2, 1, 1),
        )
    return sidecar


class GDNDeltaExecutionPlanTests(unittest.TestCase):
    # Admit the exact single-prefill checkpoint path supported by active reuse.
    def test_admits_supported_active_batch(self) -> None:
        admission = assess_gdn_delta_active_admission(
            execution_mode="active",
            mamba_cache_mode="all",
            prefill_backend="triton",
            num_prefills=1,
            num_decodes=0,
            num_spec_decodes=0,
            reuse_candidates=(((3, b"three"),),),
        )

        self.assertTrue(admission.eligible)
        self.assertEqual(admission.reason, "eligible")

    # Keep shadow mode on the ordinary full-computation path.
    def test_rejects_shadow_mode_for_active_execution(self) -> None:
        admission = assess_gdn_delta_active_admission(
            execution_mode="shadow",
            mamba_cache_mode="all",
            prefill_backend="triton",
            num_prefills=1,
            num_decodes=0,
            num_spec_decodes=0,
            reuse_candidates=(((3, b"three"),),),
        )

        self.assertFalse(admission.eligible)
        self.assertEqual(admission.reason, "shadow_mode")

    # Reject a mixed decode/prefill batch until its state ordering is implemented.
    def test_rejects_mixed_decode_batch(self) -> None:
        admission = assess_gdn_delta_active_admission(
            execution_mode="active",
            mamba_cache_mode="all",
            prefill_backend="triton",
            num_prefills=1,
            num_decodes=1,
            num_spec_decodes=0,
            reuse_candidates=(((3, b"three"),),),
        )

        self.assertFalse(admission.eligible)
        self.assertEqual(admission.reason, "mixed_decode_unsupported")

    # Carry state through recompute, affine reuse, then recompute spans in order.
    def test_executes_mixed_plan_sequentially(self) -> None:
        (plan,) = build_gdn_delta_execution_plans(
            num_computed_tokens=(0,),
            num_scheduled_tokens=(6,),
            block_size=2,
            reuse_candidates=(((1, b"middle"),),),
        )

        # Model a simple normal recurrence whose state increases once per token.
        def recompute(start: int, end: int, state: torch.Tensor):
            span_outputs = []
            for _ in range(start, end):
                state = state + 1
                span_outputs.append(state.squeeze(-1))
            return state, torch.stack(span_outputs)

        result = execute_gdn_delta_sequence_plan(
            plan,
            initial_state=torch.zeros(1, 1, 1),
            sidecar=_sidecar(b"middle"),
            recompute_span=recompute,
        )

        assert result is not None
        torch.testing.assert_close(
            result.outputs.flatten(),
            torch.tensor([1.0, 2.0, 2.0, 2.0, 13.0, 14.0]),
        )
        torch.testing.assert_close(result.final_state, torch.full((1, 1, 1), 14.0))
        self.assertEqual(result.recomputed_tokens, 4)
        self.assertEqual(result.reused_tokens, 2)

    # Convert float32 affine outputs back to the model activation dtype.
    def test_casts_mixed_outputs_to_requested_dtype(self) -> None:
        (plan,) = build_gdn_delta_execution_plans(
            num_computed_tokens=(0,),
            num_scheduled_tokens=(4,),
            block_size=2,
            reuse_candidates=(((1, b"tail"),),),
        )

        # Return half-precision outputs while retaining a stable float32 state.
        def recompute(start: int, end: int, state: torch.Tensor):
            outputs = torch.ones(end - start, 1, 1, dtype=torch.float16)
            return state + (end - start), outputs

        result = execute_gdn_delta_sequence_plan(
            plan,
            initial_state=torch.zeros(1, 1, 1),
            sidecar=_sidecar(b"tail"),
            recompute_span=recompute,
            output_dtype=torch.float16,
        )

        assert result is not None
        self.assertEqual(result.outputs.dtype, torch.float16)
        self.assertEqual(result.final_state.dtype, torch.float32)

    # Report each span's resulting state so the caller can persist checkpoints.
    def test_reports_completed_span_states_in_order(self) -> None:
        (plan,) = build_gdn_delta_execution_plans(
            num_computed_tokens=(0,),
            num_scheduled_tokens=(6,),
            block_size=2,
            reuse_candidates=(((1, b"middle"),),),
        )
        completed = []

        # Increment the state once per normally recomputed token.
        def recompute(start: int, end: int, state: torch.Tensor):
            state = state + (end - start)
            return state, state.squeeze(-1).expand(end - start, -1, -1)

        result = execute_gdn_delta_sequence_plan(
            plan,
            initial_state=torch.zeros(1, 1, 1),
            sidecar=_sidecar(b"middle"),
            recompute_span=recompute,
            span_complete=lambda span, state: completed.append(
                (span.mode, span.end_token, float(state.item()))
            ),
        )

        assert result is not None
        self.assertEqual(
            completed,
            [
                ("recompute", 2, 2.0),
                ("affine_reuse", 4, 12.0),
                ("recompute", 6, 14.0),
            ],
        )

    # Fall back before invoking normal recurrence if one operator disappeared.
    def test_missing_operator_falls_back_before_execution(self) -> None:
        (plan,) = build_gdn_delta_execution_plans(
            num_computed_tokens=(0,),
            num_scheduled_tokens=(4,),
            block_size=2,
            reuse_candidates=(((1, b"missing"),),),
        )
        recompute = Mock()

        result = execute_gdn_delta_sequence_plan(
            plan,
            initial_state=torch.zeros(1, 1, 1),
            sidecar=_sidecar(b"other"),
            recompute_span=recompute,
        )

        self.assertIsNone(result)
        recompute.assert_not_called()

    # Cover a changed gap normally and reuse the following complete blocks.
    def test_partitions_recompute_and_reuse_spans(self) -> None:
        (plan,) = build_gdn_delta_execution_plans(
            num_computed_tokens=(32,),
            num_scheduled_tokens=(48,),
            block_size=16,
            reuse_candidates=(((3, b"three"), (4, b"four")),),
        )

        self.assertEqual(
            tuple((span.start_token, span.end_token, span.mode) for span in plan.spans),
            (
                (32, 48, "recompute"),
                (48, 64, "affine_reuse"),
                (64, 80, "affine_reuse"),
            ),
        )
        self.assertEqual(plan.reused_block_indices, (3, 4))
        self.assertEqual(plan.skipped_candidate_block_indices, ())
        self.assertEqual(plan.spans[1].source_contextual_hash, b"three")

    # Recompute partial boundary blocks and reuse only fully scheduled candidates.
    def test_skips_partially_scheduled_candidate_blocks(self) -> None:
        (plan,) = build_gdn_delta_execution_plans(
            num_computed_tokens=(40,),
            num_scheduled_tokens=(40,),
            block_size=16,
            reuse_candidates=(((2, b"partial"), (3, b"full")),),
        )

        self.assertEqual(
            tuple((span.start_token, span.end_token, span.mode) for span in plan.spans),
            (
                (40, 48, "recompute"),
                (48, 64, "affine_reuse"),
                (64, 80, "recompute"),
            ),
        )
        self.assertEqual(plan.reused_block_indices, (3,))
        self.assertEqual(plan.skipped_candidate_block_indices, (2,))

    # Preserve an explicit empty plan for a request with no scheduled tokens.
    def test_accepts_empty_scheduled_interval(self) -> None:
        (plan,) = build_gdn_delta_execution_plans(
            num_computed_tokens=(64,),
            num_scheduled_tokens=(0,),
            block_size=16,
            reuse_candidates=((),),
        )

        self.assertEqual(plan.scheduled_start_token, 64)
        self.assertEqual(plan.scheduled_end_token, 64)
        self.assertEqual(plan.spans, ())

    # Reject ambiguous mappings that name the same target block twice.
    def test_rejects_duplicate_target_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_gdn_delta_execution_plans(
                num_computed_tokens=(0,),
                num_scheduled_tokens=(32,),
                block_size=16,
                reuse_candidates=(((1, b"first"), (1, b"second")),),
            )


if __name__ == "__main__":
    unittest.main()
