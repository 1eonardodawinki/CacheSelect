# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.partial_reuse import (
    PartialReuseCopyInstruction,
    ResolvedPartialReuseCandidate,
    build_partial_reuse_copy_instructions,
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
