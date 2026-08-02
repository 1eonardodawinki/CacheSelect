# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.kv_reuse_planner import (
    KVReuseDecisionReason,
    KVReusePolicy,
    NativeAPCFallbackPlanner,
)

pytestmark = pytest.mark.cpu_test


def test_no_native_prefix_recomputes() -> None:
    planner = NativeAPCFallbackPlanner()

    decision = planner.decide(prompt_tokens=256, native_cached_tokens=0)

    assert decision.policy == KVReusePolicy.FULL_RECOMPUTE
    assert decision.reason == KVReuseDecisionReason.NO_NATIVE_PREFIX


def test_any_native_prefix_uses_apc() -> None:
    planner = NativeAPCFallbackPlanner()

    decision = planner.decide(prompt_tokens=256, native_cached_tokens=1)

    assert decision.policy == KVReusePolicy.VLLM_NATIVE_APC
    assert decision.reason == KVReuseDecisionReason.REUSABLE_NATIVE_PREFIX


def test_large_native_prefix_uses_apc() -> None:
    planner = NativeAPCFallbackPlanner()

    decision = planner.decide(prompt_tokens=256, native_cached_tokens=64)

    assert decision.policy == KVReusePolicy.VLLM_NATIVE_APC
    assert decision.reason == KVReuseDecisionReason.REUSABLE_NATIVE_PREFIX
    assert decision.native_cached_tokens == 64


@pytest.mark.parametrize(
    ("prompt_tokens", "native_cached_tokens"),
    [(0, 0), (8, -1), (8, 9)],
)
def test_invalid_token_counts_are_rejected(
    prompt_tokens: int,
    native_cached_tokens: int,
) -> None:
    planner = NativeAPCFallbackPlanner()

    with pytest.raises(ValueError):
        planner.decide(
            prompt_tokens=prompt_tokens,
            native_cached_tokens=native_cached_tokens,
        )
