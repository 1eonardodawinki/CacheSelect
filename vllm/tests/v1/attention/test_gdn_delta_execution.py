# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm.model_executor.layers.mamba.gdn.delta_execution import (
    build_gdn_delta_execution_plans,
)


class GDNDeltaExecutionPlanTests(unittest.TestCase):
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
