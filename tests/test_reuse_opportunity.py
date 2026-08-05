from unittest import TestCase

from benchmarks.analyze_locator_calibration import analyze_locator_result
from benchmarks.analyze_reuse_opportunities import analyze_benchmark_result
from cacheselect.reuse_opportunity import analyze_reuse_opportunity


class ReuseOpportunityTests(TestCase):
    def test_changed_middle_exposes_unchanged_suffix_blocks(self):
        previous = list(range(8)) + list(range(20, 24)) + list(range(40, 48))
        current = list(range(8)) + list(range(30, 34)) + list(range(40, 48))

        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=8,
            block_size=4,
        )

        self.assertEqual(opportunity.exact_common_prefix_tokens, 8)
        self.assertEqual(opportunity.common_suffix_tokens, 8)
        self.assertEqual(opportunity.monotonic_post_edit_matching_tokens, 8)
        self.assertEqual(opportunity.candidate_block_count, 2)
        self.assertEqual(opportunity.candidate_token_count, 8)
        self.assertEqual(opportunity.whole_source_block_count, 2)
        self.assertEqual(opportunity.repacking_required_block_count, 0)

    def test_location_independent_matching_detects_reordered_blocks(self):
        block_a = [1, 2, 3, 4]
        block_b = [5, 6, 7, 8]
        block_c = [9, 10, 11, 12]
        previous = block_a + block_b + block_c
        current = block_a + block_c + block_b

        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=4,
            block_size=4,
        )

        self.assertEqual(opportunity.candidate_block_count, 2)
        self.assertEqual(opportunity.moved_candidate_block_count, 2)
        self.assertEqual(opportunity.whole_source_block_count, 2)
        self.assertEqual(
            [block.current_start for block in opportunity.candidate_blocks],
            [4, 8],
        )

    def test_unaligned_source_is_marked_for_repacking(self):
        previous = [1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 23]
        current = [1, 2, 3, 4, 99, 10, 11, 12, 13, 20, 21, 22, 23]

        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=4,
            block_size=4,
        )

        self.assertEqual(opportunity.candidate_block_count, 1)
        candidate = opportunity.candidate_blocks[0]
        self.assertEqual(candidate.current_start, 8)
        self.assertEqual(candidate.previous_starts, (7,))
        self.assertTrue(candidate.requires_repacking)
        self.assertFalse(candidate.has_whole_source_block)

    def test_prefix_alignment_loss_is_reported(self):
        previous = list(range(12))
        current = list(range(10)) + [99, 100]

        opportunity = analyze_reuse_opportunity(
            previous,
            current,
            native_cached_tokens=8,
            block_size=4,
        )

        self.assertEqual(opportunity.exact_common_prefix_tokens, 10)
        self.assertEqual(opportunity.prefix_alignment_loss_tokens, 2)

    def test_invalid_native_hit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact common prefix"):
            analyze_reuse_opportunity(
                [1, 2, 3],
                [1, 9, 3],
                native_cached_tokens=2,
                block_size=1,
            )

        with self.assertRaisesRegex(ValueError, "block_size"):
            analyze_reuse_opportunity(
                [1],
                [1],
                native_cached_tokens=1,
                block_size=0,
            )


class BenchmarkReuseOpportunityTests(TestCase):
    def test_result_analysis_uses_runtime_native_candidate(self):
        result = {
            "trace_id": "trace",
            "workload": "rag",
            "model": "model",
            "observations": [
                {
                    "request_id": "before",
                    "prompt_token_ids": list(range(12)),
                    "cached_tokens": 0,
                    "runtime_policy": None,
                },
                {
                    "request_id": "after",
                    "prompt_token_ids": (
                        list(range(4)) + list(range(8, 12)) + list(range(4, 8))
                    ),
                    "cached_tokens": 4,
                    "runtime_policy": {"native_cached_tokens": 4},
                },
            ],
            "transitions": [
                {
                    "transition_id": "before-to-after",
                    "previous_request_id": "before",
                    "current_request_id": "after",
                    "ground_truth": {"change_type": "reorder"},
                    "current_cached_tokens": 4,
                }
            ],
        }

        report = analyze_benchmark_result(result, block_size=4)

        self.assertEqual(report["summary"]["candidate_block_count"], 2)
        self.assertEqual(
            report["summary"]["location_independent_candidate_tokens"],
            8,
        )
        self.assertEqual(
            report["transitions"][0]["opportunity"]["native_cached_tokens"],
            4,
        )

    def test_missing_observation_is_rejected(self):
        result = {
            "observations": [],
            "transitions": [
                {
                    "transition_id": "missing",
                    "previous_request_id": "before",
                    "current_request_id": "after",
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "referenced observation"):
            analyze_benchmark_result(result, block_size=4)


class LocatorCalibrationTests(TestCase):
    @staticmethod
    def _result(candidate_token_count: int = 8):
        source_tokens = list(range(16))
        edited_tokens = list(range(4)) + [40, 41, 42, 43] + list(range(8, 16))
        plan = {
            "transition_id": "source-to-edited",
            "source_request_id": "source",
            "target_request_id": "edited",
            "block_size": 4,
            "native_cached_tokens": 4,
            "reason": "aligned_candidates",
            "candidate_block_count": candidate_token_count // 4,
            "candidate_token_count": candidate_token_count,
            "resident_candidate_block_count": candidate_token_count // 4,
            "resident_candidate_token_count": candidate_token_count,
            "candidates": [
                {
                    "source_block_index": index,
                    "target_block_index": index,
                    "source_resident": True,
                    "requires_repair": True,
                }
                for index in range(2, 2 + candidate_token_count // 4)
            ],
        }
        return {
            "request_ledger_summary": {
                "started": 2,
                "completed": 2,
                "failed": 0,
            },
            "observations": [
                {
                    "request_id": "source",
                    "prompt_token_count": 16,
                    "prompt_token_ids": source_tokens,
                    "cached_tokens": 0,
                    "quality": {"passed": True},
                },
                {
                    "request_id": "edited",
                    "prompt_token_count": 16,
                    "prompt_token_ids": edited_tokens,
                    "cached_tokens": 4,
                    "quality": {"passed": True},
                    "server_metrics": {"time_to_first_token_ms": 10.0},
                    "runtime_policy": {
                        "policy": "VLLM_NATIVE_APC",
                        "native_cached_tokens": 4,
                        "partial_reuse_plan": plan,
                    },
                },
            ],
            "transitions": [
                {
                    "transition_id": "source-to-edited",
                    "previous_request_id": "source",
                    "current_request_id": "edited",
                    "current_cached_tokens": 4,
                }
            ],
        }

    def test_online_locator_matches_offline_aligned_candidates(self):
        row = analyze_locator_result(
            self._result(),
            target_prompt_tokens=16,
            edit_position="middle",
            block_size=4,
        )

        self.assertEqual(row["native_recomputed_tokens"], 12)
        self.assertEqual(row["offline_all_candidate_tokens"], 8)
        self.assertEqual(row["online_candidate_tokens"], 8)
        self.assertEqual(row["resident_candidate_tokens"], 8)
        self.assertAlmostEqual(
            row["online_candidate_share_of_native_recompute"],
            2 / 3,
        )

    def test_online_locator_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aligned candidate mismatch"):
            analyze_locator_result(
                self._result(candidate_token_count=4),
                target_prompt_tokens=16,
                edit_position="middle",
                block_size=4,
            )
