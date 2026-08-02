# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime policy decisions for exact-prefix KV cache reuse."""

from dataclasses import dataclass
from enum import Enum


class KVReusePolicy(str, Enum):
    """Execution policies understood by the CacheSelect planner."""

    FULL_RECOMPUTE = "FULL_RECOMPUTE"
    VLLM_NATIVE_APC = "VLLM_NATIVE_APC"
    PARTIAL_KV_REUSE = "PARTIAL_KV_REUSE"


class KVReuseDecisionReason(str, Enum):
    """Stable explanation for a runtime policy decision."""

    NO_NATIVE_PREFIX = "no_native_prefix"
    REUSABLE_NATIVE_PREFIX = "reusable_native_prefix"


@dataclass(frozen=True)
class KVReuseDecision:
    """One decision made from vLLM's native prefix-cache lookup."""

    policy: KVReusePolicy
    reason: KVReuseDecisionReason
    prompt_tokens: int
    native_cached_tokens: int


class NativeAPCFallbackPlanner:
    """Preserve every exact prefix that vLLM can safely reuse.

    This planner defines the correctness-preserving fallback for future partial
    reuse policies. Until partial reuse is implemented, its selected execution
    is intentionally identical to native vLLM APC.
    """

    def decide(
        self,
        *,
        prompt_tokens: int,
        native_cached_tokens: int,
    ) -> KVReuseDecision:
        if prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= native_cached_tokens <= prompt_tokens:
            raise ValueError(
                "native_cached_tokens must be between zero and prompt_tokens"
            )

        if native_cached_tokens == 0:
            policy = KVReusePolicy.FULL_RECOMPUTE
            reason = KVReuseDecisionReason.NO_NATIVE_PREFIX
        else:
            policy = KVReusePolicy.VLLM_NATIVE_APC
            reason = KVReuseDecisionReason.REUSABLE_NATIVE_PREFIX

        return KVReuseDecision(
            policy=policy,
            reason=reason,
            prompt_tokens=prompt_tokens,
            native_cached_tokens=native_cached_tokens,
        )
