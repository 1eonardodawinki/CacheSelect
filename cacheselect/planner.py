"""Policy decisions for the CacheSelect runtime layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

from cacheselect.features import token_transition_features


class ReusePolicy(str, Enum):
    """Inference strategies that CacheSelect can select."""

    FULL_RECOMPUTE = "FULL_RECOMPUTE"
    VLLM_NATIVE_APC = "VLLM_NATIVE_APC"
    PARTIAL_KV_REUSE = "PARTIAL_KV_REUSE"


class DecisionReason(str, Enum):
    """Stable, machine-readable explanation for a policy choice."""

    COLD_START = "cold_start"
    EXACT_MATCH = "exact_match"
    APPEND_ONLY = "append_only"
    REUSABLE_NATIVE_PREFIX = "reusable_native_prefix"
    NO_COMMON_PREFIX = "no_common_prefix"


@dataclass(frozen=True)
class PolicyDecision:
    """One planner recommendation made before model execution."""

    policy: ReusePolicy
    reason: DecisionReason
    features: dict[str, Any]
    considered_policies: tuple[ReusePolicy, ...]
    confidence: float | None = None
    planner_name: str = "native-apc-fallback-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.policy not in self.considered_policies:
            raise ValueError("selected policy must be one of the considered policies")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for result files and ledgers."""
        return {
            "schema_version": self.schema_version,
            "planner_name": self.planner_name,
            "policy": self.policy.value,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "considered_policies": [
                policy.value for policy in self.considered_policies
            ],
            "features": self.features,
        }


class ReusePlanner(Protocol):
    """Interface shared by heuristic and future learned planners."""

    def decide(
        self,
        previous_prompt_tokens: Sequence[int] | None,
        current_prompt_tokens: Sequence[int],
    ) -> PolicyDecision:
        """Choose a policy using only information available before inference."""


class NativeAPCFallbackPlanner:
    """Describe the safe native-APC fallback for adjacent prompts.

    Every non-empty exact prefix is preserved. A future partial-reuse planner
    can improve on this policy, but must fall back to native APC rather than
    discarding a safe prefix hit.
    """

    considered_policies = (
        ReusePolicy.FULL_RECOMPUTE,
        ReusePolicy.VLLM_NATIVE_APC,
    )

    def decide(
        self,
        previous_prompt_tokens: Sequence[int] | None,
        current_prompt_tokens: Sequence[int],
    ) -> PolicyDecision:
        if not current_prompt_tokens:
            raise ValueError("current_prompt_tokens must not be empty")
        if previous_prompt_tokens is None:
            return PolicyDecision(
                policy=ReusePolicy.FULL_RECOMPUTE,
                reason=DecisionReason.COLD_START,
                features={
                    "previous_token_count": 0,
                    "current_token_count": len(current_prompt_tokens),
                },
                considered_policies=self.considered_policies,
            )

        features = token_transition_features(
            previous_prompt_tokens,
            current_prompt_tokens,
        )
        if features["exact_match"]:
            policy = ReusePolicy.VLLM_NATIVE_APC
            reason = DecisionReason.EXACT_MATCH
        elif features["previous_is_exact_prefix"]:
            policy = ReusePolicy.VLLM_NATIVE_APC
            reason = DecisionReason.APPEND_ONLY
        elif features["common_prefix_tokens"] > 0:
            policy = ReusePolicy.VLLM_NATIVE_APC
            reason = DecisionReason.REUSABLE_NATIVE_PREFIX
        else:
            policy = ReusePolicy.FULL_RECOMPUTE
            reason = DecisionReason.NO_COMMON_PREFIX

        return PolicyDecision(
            policy=policy,
            reason=reason,
            features=features,
            considered_policies=self.considered_policies,
        )
