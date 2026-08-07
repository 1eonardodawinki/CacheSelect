# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.partial_reuse import (
    EditProximityRepairSelector,
    FullBlockRepairSelector,
    PartialReuseCopyInstruction,
    PartialReuseRepairInstruction,
    ResolvedPartialReuseCandidate,
    build_full_block_repair_instructions,
    build_partial_reuse_copy_instructions,
    create_repair_selector,
    resolve_target_block_ids,
)


# Check that one logical target position resolves to its physical block ID.
def test_resolve_target_block_ids() -> None:
    candidate = SimpleNamespace(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        source_resident=True,
        requires_repair=True,
        block_displacement=2,
        nearest_changed_block_distance=1,
    )
    plan = SimpleNamespace(candidates=(candidate,))

    resolved = resolve_target_block_ids(plan, ([71, 12, 89, 34, 55, 63],))

    assert resolved == (
        ResolvedPartialReuseCandidate(
            source_block_index=3,
            target_block_index=5,
            source_block_id=42,
            target_block_id=63,
            source_resident=True,
            requires_repair=True,
            block_displacement=2,
            nearest_changed_block_distance=1,
        ),
    )


# Check that the resolver rejects a target position the request does not own.
def test_resolve_target_block_ids_rejects_missing_target() -> None:
    candidate = SimpleNamespace(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        source_resident=True,
        requires_repair=True,
    )
    plan = SimpleNamespace(candidates=(candidate,))

    with pytest.raises(ValueError, match="outside the request block table"):
        resolve_target_block_ids(plan, ([71, 12],))


# Check that resolved mappings become explicit but inert copy instructions.
def test_build_partial_reuse_copy_instructions() -> None:
    candidate = ResolvedPartialReuseCandidate(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        target_block_id=63,
        source_resident=True,
        requires_repair=True,
    )

    instructions = build_partial_reuse_copy_instructions((candidate,))

    assert instructions == (
        PartialReuseCopyInstruction(
            source_block_id=42,
            target_block_id=63,
            target_block_index=5,
            requires_repair=True,
        ),
    )


# Check that the fallback selector chooses every token in affected blocks.
def test_full_block_repair_selector() -> None:
    repaired_candidate = ResolvedPartialReuseCandidate(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        target_block_id=63,
        source_resident=True,
        requires_repair=True,
    )
    exact_candidate = ResolvedPartialReuseCandidate(
        source_block_index=4,
        target_block_index=6,
        source_block_id=43,
        target_block_id=64,
        source_resident=True,
        requires_repair=False,
    )

    selector = FullBlockRepairSelector()
    instructions = selector.select((repaired_candidate, exact_candidate), block_size=4)

    assert instructions == (
        PartialReuseRepairInstruction(
            source_block_id=42,
            target_block_id=63,
            target_block_index=5,
            target_token_indices=(20, 21, 22, 23),
        ),
    )


# Check that an invalid cache block size cannot create a repair range.
def test_build_full_block_repair_rejects_invalid_block_size() -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        build_full_block_repair_instructions((), block_size=0)


# Check that edit proximity repairs nearby and unknown blocks but skips far ones.
def test_edit_proximity_repair_selector() -> None:
    nearby_candidate = ResolvedPartialReuseCandidate(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        target_block_id=63,
        source_resident=True,
        requires_repair=True,
        nearest_changed_block_distance=1,
    )
    far_candidate = replace(
        nearby_candidate,
        target_block_index=6,
        target_block_id=64,
        nearest_changed_block_distance=2,
    )
    unknown_candidate = replace(
        nearby_candidate,
        target_block_index=7,
        target_block_id=65,
        nearest_changed_block_distance=None,
    )

    selector = EditProximityRepairSelector(max_block_distance=1)
    instructions = selector.select(
        (nearby_candidate, far_candidate, unknown_candidate), block_size=2
    )

    assert [instruction.target_block_id for instruction in instructions] == [63, 65]
    assert [instruction.target_token_indices for instruction in instructions] == [
        (10, 11),
        (14, 15),
    ]


# Check that an invalid edit radius cannot configure the experimental selector.
def test_edit_proximity_repair_selector_rejects_negative_radius() -> None:
    with pytest.raises(ValueError, match="max_block_distance must be non-negative"):
        EditProximityRepairSelector(max_block_distance=-1)


# Check that configuration names construct the expected repair policies.
def test_create_repair_selector() -> None:
    assert isinstance(
        create_repair_selector("full_block", edit_radius=3),
        FullBlockRepairSelector,
    )
    edit_selector = create_repair_selector("edit_proximity", edit_radius=3)
    assert isinstance(edit_selector, EditProximityRepairSelector)
    assert edit_selector.max_block_distance == 3


# Check that the selector factory rejects unknown policy names defensively.
def test_create_repair_selector_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown CacheSelect repair selector"):
        create_repair_selector("unknown", edit_radius=1)  # type: ignore[arg-type]
