import json
from unittest import TestCase

from cacheselect.planner import (
    DecisionReason,
    NativeAPCFallbackPlanner,
    PolicyDecision,
    ReusePolicy,
)
from cacheselect.tokenization import rendered_chat_token_ids


class NativeAPCFallbackPlannerTests(TestCase):
    def setUp(self):
        self.planner = NativeAPCFallbackPlanner()

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

    def test_non_prefix_edit_preserves_any_native_prefix(self):
        decision = self.planner.decide(
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 9, 6],
        )

        self.assertEqual(decision.policy, ReusePolicy.VLLM_NATIVE_APC)
        self.assertEqual(decision.reason, DecisionReason.REUSABLE_NATIVE_PREFIX)
        self.assertEqual(decision.features["common_prefix_tokens"], 4)

    def test_non_prefix_edit_recomputes_without_common_prefix(self):
        decision = self.planner.decide(
            [1, 2, 3],
            [9, 2, 3],
        )

        self.assertEqual(decision.policy, ReusePolicy.FULL_RECOMPUTE)
        self.assertEqual(decision.reason, DecisionReason.NO_COMMON_PREFIX)

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


class TokenizationTests(TestCase):
    class FakeTokenizer:
        def __init__(self, encoded):
            self.encoded = encoded

        def apply_chat_template(self, *args, **kwargs):
            return self.encoded

    def test_accepts_direct_and_mapping_tokenizer_outputs(self):
        messages = [{"role": "user", "content": "test"}]

        self.assertEqual(
            rendered_chat_token_ids(self.FakeTokenizer([1, 2, 3]), messages),
            [1, 2, 3],
        )
        self.assertEqual(
            rendered_chat_token_ids(
                self.FakeTokenizer({"input_ids": [4, 5, 6]}),
                messages,
            ),
            [4, 5, 6],
        )

    def test_accepts_single_item_batched_output(self):
        messages = [{"role": "user", "content": "test"}]

        self.assertEqual(
            rendered_chat_token_ids(self.FakeTokenizer([[1, 2, 3]]), messages),
            [1, 2, 3],
        )

    def test_rejects_multiple_rendered_prompts(self):
        messages = [{"role": "user", "content": "test"}]

        with self.assertRaisesRegex(ValueError, "received a batch"):
            rendered_chat_token_ids(
                self.FakeTokenizer([[1, 2], [3, 4]]),
                messages,
            )
