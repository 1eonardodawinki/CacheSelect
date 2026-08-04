# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shadow planning for block-aligned KV reuse beyond an exact prefix."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
from vllm.v1.request import Request


@dataclass(frozen=True)
class SourceBlock:
    block_index: int
    token_ids: tuple[int, ...]
    block_id: int
    block_hash: BlockHashWithGroupId


@dataclass(frozen=True)
class SourceRequestIndex:
    request_id: str
    cache_salt: str | None
    lora_adapter_id: int | None
    blocks: tuple[SourceBlock, ...]


@dataclass(frozen=True)
class PartialReuseCandidate:
    source_block_index: int
    target_block_index: int
    source_block_id: int
    source_resident: bool
    requires_repair: bool = True

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "source_block_index": self.source_block_index,
            "target_block_index": self.target_block_index,
            "source_resident": self.source_resident,
            "requires_repair": self.requires_repair,
        }


@dataclass(frozen=True)
class PartialReusePlan:
    transition_id: str | None
    source_request_id: str
    target_request_id: str
    block_size: int
    native_cached_tokens: int
    reason: str
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


class AlignedBlockReuseLocator:
    """Locate content-identical source blocks without applying KV reuse."""

    def __init__(
        self,
        block_pool: BlockPool,
        block_size: int,
        max_source_requests: int = 1024,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if max_source_requests < 1:
            raise ValueError("max_source_requests must be positive")
        self.block_pool = block_pool
        self.block_size = block_size
        self.max_source_requests = max_source_requests
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
                )
            )

        request_id = request.cacheselect_request_id or request.request_id
        self._sources[request_id] = SourceRequestIndex(
            request_id=request_id,
            cache_salt=request.cache_salt,
            lora_adapter_id=self._lora_adapter_id(request),
            blocks=tuple(indexed_blocks),
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

        by_content: dict[tuple[int, ...], list[SourceBlock]] = {}
        for block in source.blocks:
            by_content.setdefault(block.token_ids, []).append(block)

        prompt_token_ids = request.prompt_token_ids
        first_target_block = (
            native_cached_tokens + self.block_size - 1
        ) // self.block_size
        num_full_blocks = len(prompt_token_ids) // self.block_size
        candidates: list[PartialReuseCandidate] = []
        for target_block_index in range(first_target_block, num_full_blocks):
            start = target_block_index * self.block_size
            token_ids = tuple(prompt_token_ids[start : start + self.block_size])
            source_blocks = by_content.get(token_ids)
            if not source_blocks:
                continue
            resident_source = next(
                (block for block in source_blocks if self._is_resident(block)),
                None,
            )
            source_block = resident_source or source_blocks[0]
            candidates.append(
                PartialReuseCandidate(
                    source_block_index=source_block.block_index,
                    target_block_index=target_block_index,
                    source_block_id=source_block.block_id,
                    source_resident=resident_source is not None,
                )
            )

        return self._plan(
            request,
            native_cached_tokens,
            reason="aligned_candidates" if candidates else "no_aligned_candidates",
            candidates=tuple(candidates),
        )

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
            candidates=candidates,
        )

    def _is_resident(self, block: SourceBlock) -> bool:
        return self.block_pool.cached_block_hash_to_block.contain(
            block.block_hash,
            block.block_id,
        )

    def clear(self) -> None:
        self._sources.clear()
