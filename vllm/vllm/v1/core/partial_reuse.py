# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shadow planning for block-aligned KV reuse beyond an exact prefix."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from vllm.v1.core.kv_cache_utils import resolve_block_hashes

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.request import Request


@dataclass(frozen=True)
class SourceBlock:
    block_index: int
    token_ids: tuple[int, ...]
    block_id: int
    block_hash: BlockHashWithGroupId
    contextual_hash: bytes | None = None


@dataclass(frozen=True)
class SourceRequestIndex:
    request_id: str
    cache_salt: str | None
    lora_adapter_id: int | None
    blocks: tuple[SourceBlock, ...]
    prompt_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PartialReuseCandidate:
    source_block_index: int
    target_block_index: int
    source_block_id: int
    source_resident: bool
    requires_repair: bool = True
    block_displacement: int = 0
    nearest_changed_block_distance: int | None = None
    source_contextual_hash: bytes | None = None
    source_block_offset: int = 0
    source_block_ids: tuple[int, ...] = ()

    @property
    def physical_source_block_ids(self) -> tuple[int, ...]:
        return self.source_block_ids or (self.source_block_id,)

    @property
    def requires_repacking(self) -> bool:
        return self.source_block_offset != 0

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "source_block_index": self.source_block_index,
            "target_block_index": self.target_block_index,
            "source_resident": self.source_resident,
            "requires_repair": self.requires_repair,
            "block_displacement": self.block_displacement,
            "nearest_changed_block_distance": self.nearest_changed_block_distance,
            "source_contextual_hash": (
                self.source_contextual_hash.hex()
                if self.source_contextual_hash is not None
                else None
            ),
            "source_block_offset": self.source_block_offset,
            "source_block_ids": list(self.physical_source_block_ids),
            "requires_repacking": self.requires_repacking,
        }


@dataclass(frozen=True)
class PartialReusePlan:
    transition_id: str | None
    source_request_id: str
    target_request_id: str
    block_size: int
    native_cached_tokens: int
    reason: str
    counterfactual_reuse_block_index: int | None = None
    candidates: tuple[PartialReuseCandidate, ...] = ()

    @property
    def candidate_block_count(self) -> int:
        return len(self.candidates)

    @property
    def candidate_token_count(self) -> int:
        return self.candidate_block_count * self.block_size

    @property
    def resident_candidate_block_count(self) -> int:
        return sum(candidate.source_resident for candidate in self.candidates)

    @property
    def resident_candidate_token_count(self) -> int:
        return self.resident_candidate_block_count * self.block_size

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [
            candidate.to_public_dict() for candidate in self.candidates
        ]
        payload["candidate_block_count"] = self.candidate_block_count
        payload["candidate_token_count"] = self.candidate_token_count
        payload["resident_candidate_block_count"] = (
            self.resident_candidate_block_count
        )
        payload["resident_candidate_token_count"] = (
            self.resident_candidate_token_count
        )
        return payload

    # Reduce this plan to candidates backed by source blocks pinned for the step.
    def for_retained_source_ids(
        self, source_block_ids: Collection[int]
    ) -> PartialReusePlan | None:
        retained_candidates = tuple(
            candidate
            for candidate in self.candidates
            if candidate.source_block_id in source_block_ids
        )
        if not retained_candidates:
            return None
        return replace(self, candidates=retained_candidates)


@dataclass(frozen=True)
class CacheSelectRepairMetrics:
    selector: str
    candidate_tokens: int
    repair_tokens: int
    skipped_repair_tokens: int
    copied_blocks: int = 0
    copied_tokens: int = 0
    execution_eligible: bool = False
    execution_reason: str = "not_evaluated"
    reused_batch_rows: int = 0
    compute_batch_rows: int = 0
    compute_span_count: int = 0
    compacted_batch_built: bool = False
    compacted_batch_executed: bool = False
    span_metadata_built: bool = False
    span_metadata_count: int = 0
    copy_time_ms: float = 0.0
    preparation_time_ms: float = 0.0
    forward_time_ms: float = 0.0


class AlignedBlockReuseLocator:
    """Locate content-identical source blocks without applying KV reuse."""

    def __init__(
        self,
        block_pool: BlockPool,
        block_size: int,
        max_source_requests: int = 1024,
        hash_block_size: int | None = None,
        allow_repacking: bool = False,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if max_source_requests < 1:
            raise ValueError("max_source_requests must be positive")
        self.block_pool = block_pool
        self.block_size = block_size
        self.hash_block_size = hash_block_size or block_size
        self.max_source_requests = max_source_requests
        self.allow_repacking = allow_repacking
        self._sources: OrderedDict[str, SourceRequestIndex] = OrderedDict()

    @staticmethod
    def _lora_adapter_id(request: Request) -> int | None:
        if request.lora_request is None:
            return None
        return request.lora_request.adapter_id

    def index(self, request: Request, blocks: Sequence[KVCacheBlock]) -> None:
        prompt_token_ids = request.prompt_token_ids
        if prompt_token_ids is None:
            return

        indexed_blocks: list[SourceBlock] = []
        num_full_blocks = len(prompt_token_ids) // self.block_size
        contextual_hashes = resolve_block_hashes(
            request.block_hashes,
            self.hash_block_size,
            self.block_size,
        )
        for block_index, block in enumerate(blocks[:num_full_blocks]):
            if block.is_null or block.block_hash is None:
                continue
            start = block_index * self.block_size
            indexed_blocks.append(
                SourceBlock(
                    block_index=block_index,
                    token_ids=tuple(
                        prompt_token_ids[start : start + self.block_size]
                    ),
                    block_id=block.block_id,
                    block_hash=block.block_hash,
                    contextual_hash=(
                        bytes(contextual_hashes[block_index])
                        if block_index < len(contextual_hashes)
                        else None
                    ),
                )
            )

        request_id = request.cacheselect_request_id or request.request_id
        self._sources[request_id] = SourceRequestIndex(
            request_id=request_id,
            cache_salt=request.cache_salt,
            lora_adapter_id=self._lora_adapter_id(request),
            blocks=tuple(indexed_blocks),
            prompt_token_ids=tuple(prompt_token_ids),
        )
        self._sources.move_to_end(request_id)
        while len(self._sources) > self.max_source_requests:
            self._sources.popitem(last=False)

    def locate(
        self,
        request: Request,
        native_cached_tokens: int,
    ) -> PartialReusePlan | None:
        source_request_id = request.cacheselect_source_request_id
        if source_request_id is None:
            return None

        source = self._sources.get(source_request_id)
        if source is None:
            return self._plan(request, native_cached_tokens, "source_not_indexed")
        if source.cache_salt != request.cache_salt:
            return self._plan(request, native_cached_tokens, "cache_salt_mismatch")
        if source.lora_adapter_id != self._lora_adapter_id(request):
            return self._plan(request, native_cached_tokens, "lora_mismatch")
        if request.prompt_token_ids is None:
            return self._plan(request, native_cached_tokens, "token_ids_unavailable")

        by_content = self._source_windows_by_content(source)

        prompt_token_ids = request.prompt_token_ids
        first_target_block = (
            native_cached_tokens + self.block_size - 1
        ) // self.block_size
        num_full_blocks = len(prompt_token_ids) // self.block_size
        candidates: list[PartialReuseCandidate] = []
        for target_block_index in range(first_target_block, num_full_blocks):
            start = target_block_index * self.block_size
            token_ids = tuple(prompt_token_ids[start : start + self.block_size])
            source_windows = by_content.get(token_ids)
            if not source_windows:
                continue
            source_start, source_blocks = self._select_nearest_source_window(
                source_windows,
                target_block_index,
            )
            source_block = source_blocks[0]
            source_offset = source_start % self.block_size
            candidates.append(
                PartialReuseCandidate(
                    source_block_index=source_block.block_index,
                    target_block_index=target_block_index,
                    source_block_id=source_block.block_id,
                    source_resident=all(
                        self._is_resident(block) for block in source_blocks
                    ),
                    source_contextual_hash=source_block.contextual_hash,
                    source_block_offset=source_offset,
                    source_block_ids=tuple(block.block_id for block in source_blocks),
                )
            )

        annotated_candidates = self._annotate_change_geometry(
            tuple(candidates), first_target_block, num_full_blocks
        )
        return self._plan(
            request,
            native_cached_tokens,
            reason=(
                "repacking_candidates"
                if any(candidate.requires_repacking for candidate in candidates)
                else "aligned_candidates"
                if candidates
                else "no_aligned_candidates"
            ),
            candidates=annotated_candidates,
        )

    # Index exact source windows while retaining their physical block mapping.
    def _source_windows_by_content(
        self, source: SourceRequestIndex
    ) -> dict[tuple[int, ...], list[tuple[int, tuple[SourceBlock, ...]]]]:
        blocks_by_index = {block.block_index: block for block in source.blocks}
        tokens = source.prompt_token_ids
        if not tokens:
            # Old indexes contain only aligned blocks and remain valid.
            windows: dict[
                tuple[int, ...], list[tuple[int, tuple[SourceBlock, ...]]]
            ] = {}
            for block in source.blocks:
                windows.setdefault(block.token_ids, []).append(
                    (block.block_index * self.block_size, (block,))
                )
            return windows
        step = 1 if self.allow_repacking else self.block_size
        windows: dict[tuple[int, ...], list[tuple[int, tuple[SourceBlock, ...]]]] = {}
        for start in range(0, len(tokens) - self.block_size + 1, step):
            first = start // self.block_size
            last = (start + self.block_size - 1) // self.block_size
            physical = tuple(
                blocks_by_index[index] for index in range(first, last + 1)
                if index in blocks_by_index
            )
            if len(physical) != last - first + 1:
                continue
            token_ids = tuple(tokens[start : start + self.block_size])
            windows.setdefault(token_ids, []).append((start, physical))
        return windows

    # Prefer resident aligned storage, then the occurrence nearest the target.
    def _select_nearest_source_window(
        self,
        windows: Sequence[tuple[int, tuple[SourceBlock, ...]]],
        target_block_index: int,
    ) -> tuple[int, tuple[SourceBlock, ...]]:
        if not windows:
            raise ValueError("source window choices cannot be empty")
        return min(
            windows,
            key=lambda item: (
                not all(self._is_resident(block) for block in item[1]),
                item[0] % self.block_size != 0,
                abs(item[0] - target_block_index * self.block_size),
                item[0],
            ),
        )

    # Prefer the resident identical block whose original position is closest.
    def _select_nearest_source_block(
        self,
        source_blocks: Sequence[SourceBlock],
        target_block_index: int,
    ) -> SourceBlock:
        resident_blocks = tuple(
            block for block in source_blocks if self._is_resident(block)
        )
        choices = resident_blocks or tuple(source_blocks)
        if not choices:
            raise ValueError("source block choices cannot be empty")
        # Stable index tie-breaking makes repeated-content plans reproducible.
        return min(
            choices,
            key=lambda block: (
                abs(block.block_index - target_block_index),
                block.block_index,
            ),
        )

    # Add cheap edit-proximity signals to each content-identical block match.
    @staticmethod
    def _annotate_change_geometry(
        candidates: tuple[PartialReuseCandidate, ...],
        first_target_block: int,
        num_target_blocks: int,
    ) -> tuple[PartialReuseCandidate, ...]:
        if not candidates:
            return ()

        matched_targets = {
            candidate.target_block_index for candidate in candidates
        }
        changed_targets = set(range(first_target_block, num_target_blocks))
        changed_targets.difference_update(matched_targets)

        previous_candidate: PartialReuseCandidate | None = None
        for candidate in candidates:
            if previous_candidate is None:
                source_discontinuity = (
                    candidate.source_block_index != candidate.target_block_index
                )
            else:
                targets_are_adjacent = (
                    previous_candidate.target_block_index + 1
                    == candidate.target_block_index
                )
                sources_are_adjacent = (
                    previous_candidate.source_block_index + 1
                    == candidate.source_block_index
                )
                source_discontinuity = targets_are_adjacent and not sources_are_adjacent

            # With no unmatched target block, a source-index discontinuity marks
            # the target-side boundary of a deletion or reordered block run.
            if (
                source_discontinuity
                and candidate.target_block_index - 1 not in changed_targets
            ):
                changed_targets.add(candidate.target_block_index)
            previous_candidate = candidate

        return tuple(
            replace(
                candidate,
                block_displacement=(
                    candidate.target_block_index - candidate.source_block_index
                ),
                nearest_changed_block_distance=(
                    min(
                        abs(candidate.target_block_index - changed_target)
                        for changed_target in changed_targets
                    )
                    if changed_targets
                    else None
                ),
            )
            for candidate in candidates
        )

    # Assemble one plan while preserving request-scoped experiment metadata.
    def _plan(
        self,
        request: Request,
        native_cached_tokens: int,
        reason: str,
        candidates: tuple[PartialReuseCandidate, ...] = (),
    ) -> PartialReusePlan:
        source_request_id = request.cacheselect_source_request_id
        assert source_request_id is not None
        return PartialReusePlan(
            transition_id=request.cacheselect_transition_id,
            source_request_id=source_request_id,
            target_request_id=request.cacheselect_request_id or request.request_id,
            block_size=self.block_size,
            native_cached_tokens=native_cached_tokens,
            reason=reason,
            counterfactual_reuse_block_index=(
                request.cacheselect_counterfactual_reuse_block_index
            ),
            candidates=candidates,
        )

    def _is_resident(self, block: SourceBlock) -> bool:
        return self.block_pool.cached_block_hash_to_block.contain(
            block.block_hash,
            block.block_id,
        )

    # Revalidate and pin each unique resident source block used by a plan.
    def retain_resident_sources(
        self, plan: PartialReusePlan
    ) -> tuple[KVCacheBlock, ...]:
        source = self._sources.get(plan.source_request_id)
        if source is None:
            return ()

        indexed_blocks = {
            (block.block_index, block.block_id): block for block in source.blocks
        }
        retained_by_id: dict[int, KVCacheBlock] = {}
        for candidate in plan.candidates:
            if not candidate.source_resident:
                continue
            source_block = indexed_blocks.get(
                (candidate.source_block_index, candidate.source_block_id)
            )
            if source_block is None or not self._is_resident(source_block):
                continue

            # BlockPool owns one canonical object per physical block ID. The
            # hash-and-ID residency check above prevents retaining a reassigned block.
            retained_by_id[source_block.block_id] = self.block_pool.blocks[
                source_block.block_id
            ]

        retained = tuple(retained_by_id.values())
        if retained:
            self.block_pool.touch(retained)
        return retained

    # Release references previously acquired by retain_resident_sources().
    def release_sources(self, blocks: Sequence[KVCacheBlock]) -> None:
        if blocks:
            self.block_pool.free_blocks(reversed(blocks))

    def clear(self) -> None:
        self._sources.clear()
