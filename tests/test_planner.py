import json
from unittest import TestCase

from cacheselect.planner import (
    DecisionReason,
    PolicyDecision,
    PrefixHeuristicPlanner,
    ReusePolicy,
)


class PrefixHeuristicPlannerTests(TestCase):
    def setUp(self):
        self.planner = PrefixHeuristicPlanner(minimum_native_prefix_tokens=4)

    def test_cold_request_recomputes(self):
        decision = self.planner.decide(None, [1, 2, 3])

        self.assertEqual(decision.policy, ReusePolicy.FULL_RECOMPUTE)
        self.assertEqual(decision.reason, DecisionReason.COLD_START)

    def test_exact_match_uses_native_cache(self):
        decision = self.planner.decide([1, 2, 3], [1, 2, 3])

        self.assertEqual(decision.policy, ReusePolicy.VLLM_NATIVE_APC)
        self.assertEqual(decision.reason, DecisionReason.EXACT_MATCH)

    def test_append_only_request_uses_native_cache(self):
        decision = self.planner.decide([1, 2], [1, 2, 3])

        self.assertEqual(decision.policy, ReusePolicy.VLLM_NATIVE_APC)
        self.assertEqual(decision.reason, DecisionReason.APPEND_ONLY)

    def test_non_prefix_edit_uses_native_cache_for_large_prefix(self):
        decision = self.planner.decide(
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 9, 6],
        )

        self.assertEqual(decision.policy, ReusePolicy.VLLM_NATIVE_APC)
        self.assertEqual(decision.reason, DecisionReason.REUSABLE_NATIVE_PREFIX)
        self.assertEqual(decision.features["common_prefix_tokens"], 4)

    def test_non_prefix_edit_recomputes_for_small_prefix(self):
        decision = self.planner.decide(
            [1, 2, 3, 4, 5],
            [1, 2, 9, 4, 5],
        )

        self.assertEqual(decision.policy, ReusePolicy.FULL_RECOMPUTE)
        self.assertEqual(decision.reason, DecisionReason.PREFIX_TOO_SMALL)

    def test_decision_is_json_safe_and_lists_future_action(self):
        decision = self.planner.decide([1, 2], [1, 2, 3])
        payload = decision.to_dict()

        self.assertEqual(payload["policy"], "VLLM_NATIVE_APC")
        self.assertNotIn("PARTIAL_KV_REUSE", payload["considered_policies"])
        json.dumps(payload)

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "confidence"):
            PolicyDecision(
                policy=ReusePolicy.FULL_RECOMPUTE,
                reason=DecisionReason.COLD_START,
                features={},
                considered_policies=(ReusePolicy.FULL_RECOMPUTE,),
                confidence=1.1,
            )

    def test_empty_current_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.planner.decide([1], [])
