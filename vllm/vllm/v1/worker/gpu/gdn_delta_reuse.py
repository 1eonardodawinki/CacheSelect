# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed GPU-side preflight for hybrid GDN delta reuse."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.model_executor.layers.mamba.gdn.delta_cache import (
    GDNDeltaCacheEntry,
    GDNDeltaOperatorSidecar,
)

if TYPE_CHECKING:
    from vllm.v1.core.gdn_delta_reuse import GDNDeltaReusePlan


@dataclass(frozen=True)
class ResolvedGDNDeltaLayer:
    """Resident block operators resolved for one GDN layer."""

    layer_name: str
    entries: tuple[GDNDeltaCacheEntry, ...]


@dataclass(frozen=True)
class GDNDeltaPreflightResult:
    """Auditable all-layer decision made before model execution."""

    eligible: bool
    reason: str
    candidate_count: int
    layers: tuple[ResolvedGDNDeltaLayer, ...] = ()


# Discover Qwen GDN modules structurally without importing a model implementation.
def collect_gdn_delta_sidecars(
    model: nn.Module,
) -> tuple[tuple[str, GDNDeltaOperatorSidecar | None], ...]:
    missing = object()
    layers = []
    for layer_name, module in model.named_modules():
        sidecar = getattr(module, "gdn_delta_operator_sidecar", missing)
        if sidecar is missing:
            continue
        if sidecar is not None and not isinstance(
            sidecar, GDNDeltaOperatorSidecar
        ):
            raise TypeError(
                f"{layer_name}.gdn_delta_operator_sidecar has an invalid type"
            )
        layers.append((layer_name, sidecar))
    return tuple(layers)


# Resolve every candidate on every layer, or return a recompute-only result.
def preflight_gdn_delta_reuse(
    plan: GDNDeltaReusePlan,
    layers: Sequence[tuple[str, GDNDeltaOperatorSidecar | None]],
) -> GDNDeltaPreflightResult:
    block_hashes = tuple(
        candidate.source_contextual_hash for candidate in plan.candidates
    )
    candidate_count = len(block_hashes)
    if candidate_count == 0:
        return GDNDeltaPreflightResult(False, "no_candidates", 0)
    if not layers:
        return GDNDeltaPreflightResult(False, "no_gdn_layers", candidate_count)

    for _, sidecar in layers:
        if sidecar is None:
            return GDNDeltaPreflightResult(
                False, "sidecar_unavailable", candidate_count
            )
        if sidecar.block_size != plan.block_size:
            return GDNDeltaPreflightResult(
                False, "block_size_mismatch", candidate_count
            )
        if not sidecar.contains_many(block_hashes):
            return GDNDeltaPreflightResult(
                False, "operator_not_resident", candidate_count
            )

    resolved_layers = []
    for layer_name, sidecar in layers:
        assert sidecar is not None
        entries = sidecar.lookup_many(block_hashes)
        # Residency was checked above; a miss now would indicate mutation.
        if entries is None:
            return GDNDeltaPreflightResult(
                False, "operator_evicted_during_preflight", candidate_count
            )
        resolved_layers.append(ResolvedGDNDeltaLayer(layer_name, entries))
    return GDNDeltaPreflightResult(
        True,
        "eligible",
        candidate_count,
        tuple(resolved_layers),
    )
