# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

from vllm.v1.core.partial_reuse import (
    AlignedBlockReuseLocator,
    PartialReuseCandidate,
    PartialReusePlan,
    SourceBlock,
    SourceRequestIndex,
)


class FakeCachedBlockMap:
    # Create a controllable residency index for locator tests.
    def __init__(self, resident: bool = True) -> None:
        self.resident = resident

    # Report whether the expected hash still owns the expected physical block.
    def contain(self, block_hash: object, block_id: int) -> bool:
        return self.resident


class FakeBlockPool:
    # Create a small pool whose reference counts behave like BlockPool.
    def __init__(self, block_id: int, resident: bool = True) -> None:
        self.blocks = [SimpleNamespace(block_id=index, ref_cnt=0) for index in range(8)]
        self.block = self.blocks[block_id]
        self.cached_block_hash_to_block = FakeCachedBlockMap(resident)

    # Pin blocks by adding one reference to each unique physical allocation.
    def touch(self, blocks) -> None:
        for block in blocks:
            block.ref_cnt += 1

    # Release the references acquired through touch().
    def free_blocks(self, blocks) -> None:
        for block in blocks:
            block.ref_cnt -= 1


# Build a locator and plan with two targets sharing one source block.
def make_locator_and_plan(resident: bool = True):
    block_id = 7
    block_hash = object()
    pool = FakeBlockPool(block_id, resident=resident)
    locator = AlignedBlockReuseLocator(pool, block_size=16)
    locator._sources["source"] = SourceRequestIndex(
        request_id="source",
        cache_salt=None,
        lora_adapter_id=None,
        blocks=(
            SourceBlock(
                block_index=3,
                token_ids=tuple(range(16)),
                block_id=block_id,
                block_hash=block_hash,
            ),
        ),
    )
    candidates = tuple(
        PartialReuseCandidate(
            source_block_index=3,
            target_block_index=target,
            source_block_id=block_id,
            source_resident=True,
        )
        for target in (5, 6)
    )
    plan = PartialReusePlan(
        transition_id="transition",
        source_request_id="source",
        target_request_id="target",
        block_size=16,
        native_cached_tokens=0,
        reason="aligned_candidates",
        candidates=candidates,
    )
    return locator, pool, plan


# Check that duplicate candidates acquire and release only one block reference.
def test_retain_and_release_unique_resident_sources() -> None:
    locator, pool, plan = make_locator_and_plan()

    retained = locator.retain_resident_sources(plan)

    assert retained == (pool.block,)
    assert pool.block.ref_cnt == 1

    locator.release_sources(retained)
    assert pool.block.ref_cnt == 0


# Check that a stale source mapping is not retained after revalidation fails.
def test_retain_resident_sources_skips_stale_block() -> None:
    locator, pool, plan = make_locator_and_plan(resident=False)

    retained = locator.retain_resident_sources(plan)

    assert retained == ()
    assert pool.block.ref_cnt == 0


# Check that worker metadata contains only mappings backed by retained blocks.
def test_plan_filters_candidates_to_retained_source_ids() -> None:
    _, _, plan = make_locator_and_plan()
    unretained_candidate = PartialReuseCandidate(
        source_block_index=4,
        target_block_index=7,
        source_block_id=6,
        source_resident=True,
    )
    plan = replace(plan, candidates=(*plan.candidates, unretained_candidate))

    retained_plan = plan.for_retained_source_ids({7})

    assert retained_plan is not None
    assert retained_plan.candidates == plan.candidates[:2]
    assert plan.for_retained_source_ids(set()) is None


# Check that insertion geometry measures candidates from the unmatched block.
def test_change_geometry_for_inserted_block() -> None:
    candidates = (
        PartialReuseCandidate(1, 2, 6, True),
        PartialReuseCandidate(2, 3, 7, True),
    )

    annotated = AlignedBlockReuseLocator._annotate_change_geometry(
        candidates, first_target_block=1, num_target_blocks=4
    )

    assert [candidate.block_displacement for candidate in annotated] == [1, 1]
    assert [
        candidate.nearest_changed_block_distance for candidate in annotated
    ] == [1, 2]


# Check that a deletion creates a virtual change boundary in the target layout.
def test_change_geometry_for_deleted_block() -> None:
    candidates = (
        PartialReuseCandidate(2, 1, 6, True),
        PartialReuseCandidate(3, 2, 7, True),
    )

    annotated = AlignedBlockReuseLocator._annotate_change_geometry(
        candidates, first_target_block=1, num_target_blocks=3
    )

    assert [candidate.block_displacement for candidate in annotated] == [-1, -1]
    assert [
        candidate.nearest_changed_block_distance for candidate in annotated
    ] == [0, 1]
