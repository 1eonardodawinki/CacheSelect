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
    PREFIX_TOO_SMALL = "prefix_too_small"


@dataclass(frozen=True)
class PolicyDecision:
    """One planner recommendation made before model execution."""

    policy: ReusePolicy
    reason: DecisionReason
    features: dict[str, Any]
    considered_policies: tuple[ReusePolicy, ...]
    confidence: float | None = None
    planner_name: str = "prefix-heuristic-v1"
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


class PrefixHeuristicPlanner:
    """Initial transparent planner for full computation versus native APC.

    Exact append-only transitions use native APC. For non-prefix edits, native
    APC is recommended only when the exact prefix is large enough to be useful.
    The default 64-token threshold is deliberately configurable; it is an
    initial rule informed by the A16 calibration, not a learned optimum.
    """

    considered_policies = (
        ReusePolicy.FULL_RECOMPUTE,
        ReusePolicy.VLLM_NATIVE_APC,
    )

    def __init__(self, *, minimum_native_prefix_tokens: int = 64) -> None:
        if minimum_native_prefix_tokens < 1:
            raise ValueError("minimum_native_prefix_tokens must be positive")
        self.minimum_native_prefix_tokens = minimum_native_prefix_tokens

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
        elif features["common_prefix_tokens"] >= self.minimum_native_prefix_tokens:
            policy = ReusePolicy.VLLM_NATIVE_APC
            reason = DecisionReason.REUSABLE_NATIVE_PREFIX
        else:
            policy = ReusePolicy.FULL_RECOMPUTE
            reason = DecisionReason.PREFIX_TOO_SMALL

        return PolicyDecision(
            policy=policy,
            reason=reason,
            features=features,
            considered_policies=self.considered_policies,
        )
