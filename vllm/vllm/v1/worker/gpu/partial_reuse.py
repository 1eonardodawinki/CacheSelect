# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve CacheSelect plans against V2 runner block allocations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.partial_reuse import PartialReusePlan


@dataclass(frozen=True)
class ResolvedPartialReuseCandidate:
    source_block_index: int
    target_block_index: int
    source_block_id: int
    target_block_id: int
    source_resident: bool
    requires_repair: bool


def resolve_target_block_ids(
    plan: PartialReusePlan,
    target_block_ids: Sequence[Sequence[int]],
) -> tuple[ResolvedPartialReuseCandidate, ...]:
    """Resolve logical target positions to physical block IDs."""
    if len(target_block_ids) != 1:
        raise ValueError("partial reuse currently requires one KV cache group")

    target_group = target_block_ids[0]
    resolved = []
    for candidate in plan.candidates:
        target_index = candidate.target_block_index
        if target_index < 0 or target_index >= len(target_group):
            raise ValueError(
                f"target block index {target_index} is outside the request block table"
            )
        resolved.append(
            ResolvedPartialReuseCandidate(
                source_block_index=candidate.source_block_index,
                target_block_index=target_index,
                source_block_id=candidate.source_block_id,
                target_block_id=target_group[target_index],
                source_resident=candidate.source_resident,
                requires_repair=candidate.requires_repair,
            )
        )
    return tuple(resolved)
