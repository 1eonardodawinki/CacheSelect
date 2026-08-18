# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded sidecar storage for cached GDN block-delta operators."""

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNDeltaCacheEntry:
    """Tensor views and slot metadata for one resident block operator."""

    slot: int
    transition: torch.Tensor
    output_responses: torch.Tensor


# Repeat grouped key/query heads across the value heads that consume them.
def _expand_gdn_grouped_heads(
    tensor: torch.Tensor,
    value_heads: int,
) -> torch.Tensor:
    key_heads = tensor.shape[1]
    if value_heads % key_heads != 0:
        raise ValueError("value-head count must be divisible by key-head count")
    return tensor.repeat_interleave(value_heads // key_heads, dim=1)


# Compose one unchanged GDN block into final-state and output response tensors.
def build_gdn_delta_operator(
    keys: torch.Tensor,
    queries: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the linear response to an incoming recurrent-state difference."""
    if keys.ndim != 3 or queries.shape != keys.shape:
        raise ValueError("keys and queries must have matching [T,H,K] shapes")
    token_count, _, key_width = keys.shape
    if log_decays.ndim != 2 or betas.shape != log_decays.shape:
        raise ValueError("log_decays and betas must have matching [T,HV] shapes")
    if log_decays.shape[0] != token_count:
        raise ValueError("coefficient and token counts must agree")

    value_heads = log_decays.shape[1]
    expanded_keys = _expand_gdn_grouped_heads(keys, value_heads).float()
    expanded_queries = _expand_gdn_grouped_heads(queries, value_heads).float()
    log_decays = log_decays.float()
    betas = betas.float()
    transition = torch.eye(
        key_width,
        dtype=torch.float32,
        device=keys.device,
    ).expand(value_heads, key_width, key_width).clone()
    responses = torch.empty(
        token_count,
        value_heads,
        key_width,
        dtype=torch.float32,
        device=keys.device,
    )
    output_scale = key_width**-0.5
    for token_index in range(token_count):
        key = expanded_keys[token_index]
        transition_read = torch.einsum("hkl,hl->hk", transition, key)
        transition = log_decays[token_index].exp()[:, None, None] * (
            transition
            - betas[token_index, :, None, None]
            * transition_read[:, :, None]
            * key[:, None, :]
        )
        responses[token_index] = (
            torch.einsum(
                "hkl,hl->hk",
                transition,
                expanded_queries[token_index],
            )
            * output_scale
        )
    return transition, responses


class GDNDeltaOperatorSidecar:
    """Fixed-capacity LRU storage kept outside ordinary Mamba cache pages."""

    # Allocate the bounded tensor storage and initialize its CPU lookup table.
    def __init__(
        self,
        *,
        capacity: int,
        block_size: int,
        value_heads: int,
        key_width: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        dimensions = (capacity, block_size, value_heads, key_width)
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("all GDN sidecar dimensions must be positive")
        if not dtype.is_floating_point:
            raise ValueError("GDN sidecar tensors require a floating-point dtype")

        self.capacity = capacity
        self.block_size = block_size
        self.value_heads = value_heads
        self.key_width = key_width
        self.transitions = torch.empty(
            capacity,
            value_heads,
            key_width,
            key_width,
            dtype=dtype,
            device=device,
        )
        self.output_responses = torch.empty(
            capacity,
            block_size,
            value_heads,
            key_width,
            dtype=dtype,
            device=device,
        )
        self._key_to_slot: OrderedDict[bytes, int] = OrderedDict()
        self._free_slots = list(range(capacity))

    # Report how many block operators currently occupy the bounded cache.
    def __len__(self) -> int:
        return len(self._key_to_slot)

    # Return resident keys in least-to-most-recently-used order for auditing.
    def resident_keys(self) -> tuple[bytes, ...]:
        return tuple(self._key_to_slot)

    # Insert or refresh one operator, evicting the least-recent key if needed.
    def store(
        self,
        block_hash: bytes,
        transition: torch.Tensor,
        output_responses: torch.Tensor,
    ) -> int:
        if not block_hash:
            raise ValueError("block_hash must be non-empty")
        expected_transition = (self.value_heads, self.key_width, self.key_width)
        expected_responses = (self.block_size, self.value_heads, self.key_width)
        if transition.shape != expected_transition:
            raise ValueError(
                f"transition must have shape {expected_transition}, "
                f"received {tuple(transition.shape)}"
            )
        if output_responses.shape != expected_responses:
            raise ValueError(
                f"output_responses must have shape {expected_responses}, "
                f"received {tuple(output_responses.shape)}"
            )

        slot = self._key_to_slot.pop(block_hash, None)
        if slot is None:
            if self._free_slots:
                slot = self._free_slots.pop(0)
            else:
                _, slot = self._key_to_slot.popitem(last=False)
        self.transitions[slot].copy_(transition)
        self.output_responses[slot].copy_(output_responses)
        self._key_to_slot[block_hash] = slot
        return slot

    # Resolve one block hash and refresh its LRU position without copying data.
    def lookup(self, block_hash: bytes) -> GDNDeltaCacheEntry | None:
        slot = self._key_to_slot.pop(block_hash, None)
        if slot is None:
            return None
        self._key_to_slot[block_hash] = slot
        return GDNDeltaCacheEntry(
            slot=slot,
            transition=self.transitions[slot],
            output_responses=self.output_responses[slot],
        )

    # Remove all logical entries while retaining the allocated GPU buffers.
    def clear(self) -> None:
        self._key_to_slot.clear()
        self._free_slots = list(range(self.capacity))
