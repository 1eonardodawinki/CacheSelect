# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-safe source indexing for experimental GDN delta reuse."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import resolve_block_hashes

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(frozen=True)
class GDNDeltaSourceBlock:
    """One full source block and its identity under the original prefix."""

    block_index: int
    token_ids: tuple[int, ...]
    contextual_hash: bytes


@dataclass(frozen=True)
class GDNDeltaSourceRequest:
    """Prompt blocks retained independently of physical KV-cache residency."""

    request_id: str
    cache_salt: str | None
    lora_adapter_id: int | None
    blocks: tuple[GDNDeltaSourceBlock, ...]


@dataclass(frozen=True)
class GDNDeltaReuseCandidate:
    """Map one unchanged target block to its operator's source context."""

    source_block_index: int
    target_block_index: int
    source_contextual_hash: bytes

    # Convert the opaque hash to hexadecimal for JSON-safe observability.
    def to_public_dict(self) -> dict[str, Any]:
        return {
            "source_block_index": self.source_block_index,
            "target_block_index": self.target_block_index,
            "source_contextual_hash": self.source_contextual_hash.hex(),
        }


@dataclass(frozen=True)
class GDNDeltaReusePlan:
    """Request-scoped block mappings for fail-closed GDN operator lookup."""

    source_request_id: str
    target_request_id: str
    block_size: int
    native_cached_tokens: int
    reason: str
    candidates: tuple[GDNDeltaReuseCandidate, ...] = ()

    # Produce a JSON-safe plan without exposing mutable runtime state.
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [
            candidate.to_public_dict() for candidate in self.candidates
        ]
        payload["candidate_block_count"] = len(self.candidates)
        return payload


class GDNDeltaSourceIndex:
    """Bounded CPU index used to find source-context GDN operators later."""

    # Initialize prompt/hash geometry and the bounded request LRU.
    def __init__(
        self,
        *,
        block_size: int,
        hash_block_size: int,
        max_source_requests: int = 1024,
        max_candidate_blocks: int | None = None,
    ) -> None:
        if block_size < 1 or hash_block_size < 1:
            raise ValueError("block sizes must be positive")
        if block_size % hash_block_size != 0:
            raise ValueError("block_size must be divisible by hash_block_size")
        if max_source_requests < 1:
            raise ValueError("max_source_requests must be positive")
        if max_candidate_blocks is not None and max_candidate_blocks < 1:
            raise ValueError("max_candidate_blocks must be positive")
        self.block_size = block_size
        self.hash_block_size = hash_block_size
        self.max_source_requests = max_source_requests
        self.max_candidate_blocks = max_candidate_blocks
        self._sources: OrderedDict[str, GDNDeltaSourceRequest] = OrderedDict()
        logger.info(
            "GDN delta diagnostic: source index enabled block_size=%d "
            "hash_block_size=%d max_candidate_blocks=%s",
            block_size,
            hash_block_size,
            max_candidate_blocks,
        )

    # Read the request's active LoRA identity without retaining the adapter.
    @staticmethod
    def _lora_adapter_id(request: Request) -> int | None:
        if request.lora_request is None:
            return None
        return request.lora_request.adapter_id

    # Index every complete prompt block under its original chained context hash.
    def index(self, request: Request) -> None:
        prompt_token_ids = request.prompt_token_ids
        if prompt_token_ids is None:
            return
        hashes = resolve_block_hashes(
            request.block_hashes,
            self.hash_block_size,
            self.block_size,
        )
        num_full_blocks = min(
            len(prompt_token_ids) // self.block_size,
            len(hashes),
        )
        blocks = tuple(
            GDNDeltaSourceBlock(
                block_index=block_index,
                token_ids=tuple(
                    prompt_token_ids[
                        block_index * self.block_size : (block_index + 1)
                        * self.block_size
                    ]
                ),
                contextual_hash=bytes(hashes[block_index]),
            )
            for block_index in range(num_full_blocks)
        )
        request_id = request.cacheselect_request_id or request.request_id
        self._sources[request_id] = GDNDeltaSourceRequest(
            request_id=request_id,
            cache_salt=request.cache_salt,
            lora_adapter_id=self._lora_adapter_id(request),
            blocks=blocks,
        )
        self._sources.move_to_end(request_id)
        while len(self._sources) > self.max_source_requests:
            self._sources.popitem(last=False)
        logger.info(
            "GDN delta diagnostic: indexed engine_request=%s source_request=%s "
            "prompt_tokens=%d blocks=%d hash_count=%d index_size=%d",
            request.request_id,
            request_id,
            len(prompt_token_ids),
            len(blocks),
            len(hashes),
            len(self._sources),
        )

    # Resolve and refresh one indexed source request.
    def get(self, request_id: str) -> GDNDeltaSourceRequest | None:
        source = self._sources.get(request_id)
        if source is not None:
            self._sources.move_to_end(request_id)
        return source

    # Find full target blocks whose token content exists in the named source.
    def locate(
        self,
        request: Request,
        native_cached_tokens: int,
    ) -> GDNDeltaReusePlan | None:
        source_request_id = request.cacheselect_source_request_id
        if source_request_id is None:
            if request.cacheselect_request_id is not None:
                logger.info(
                    "GDN delta diagnostic: locate skipped target_request=%s "
                    "because source metadata is absent",
                    request.cacheselect_request_id,
                )
            return None
        source = self.get(source_request_id)
        if source is None:
            logger.info(
                "GDN delta diagnostic: locate target_request=%s "
                "source_request=%s reason=source_not_indexed index_size=%d",
                request.cacheselect_request_id or request.request_id,
                source_request_id,
                len(self._sources),
            )
            return self._plan(request, native_cached_tokens, "source_not_indexed")
        if source.cache_salt != request.cache_salt:
            return self._plan(request, native_cached_tokens, "cache_salt_mismatch")
        if source.lora_adapter_id != self._lora_adapter_id(request):
            return self._plan(request, native_cached_tokens, "lora_mismatch")
        prompt_token_ids = request.prompt_token_ids
        if prompt_token_ids is None:
            return self._plan(request, native_cached_tokens, "token_ids_unavailable")

        candidate_source_blocks = source.blocks
        if self.max_candidate_blocks is not None:
            # Sidecars insert source operators in block order, retaining the tail.
            candidate_source_blocks = candidate_source_blocks[
                -self.max_candidate_blocks :
            ]
        by_content: dict[tuple[int, ...], list[GDNDeltaSourceBlock]] = {}
        for block in candidate_source_blocks:
            by_content.setdefault(block.token_ids, []).append(block)
        first_target_block = (
            native_cached_tokens + self.block_size - 1
        ) // self.block_size
        num_target_blocks = len(prompt_token_ids) // self.block_size
        candidates: list[GDNDeltaReuseCandidate] = []
        for target_block_index in range(first_target_block, num_target_blocks):
            start = target_block_index * self.block_size
            token_ids = tuple(prompt_token_ids[start : start + self.block_size])
            source_blocks = by_content.get(token_ids)
            if not source_blocks:
                continue
            source_block = min(
                source_blocks,
                key=lambda block: (
                    abs(block.block_index - target_block_index),
                    block.block_index,
                ),
            )
            candidates.append(
                GDNDeltaReuseCandidate(
                    source_block_index=source_block.block_index,
                    target_block_index=target_block_index,
                    source_contextual_hash=source_block.contextual_hash,
                )
            )
        plan = self._plan(
            request,
            native_cached_tokens,
            "aligned_candidates" if candidates else "no_aligned_candidates",
            tuple(candidates),
        )
        logger.info(
            "GDN delta diagnostic: located target_request=%s source_request=%s "
            "native_cached_tokens=%d candidates=%d reason=%s",
            plan.target_request_id,
            plan.source_request_id,
            native_cached_tokens,
            len(plan.candidates),
            plan.reason,
        )
        return plan

    # Assemble a request-scoped lookup plan for worker transport.
    def _plan(
        self,
        request: Request,
        native_cached_tokens: int,
        reason: str,
        candidates: tuple[GDNDeltaReuseCandidate, ...] = (),
    ) -> GDNDeltaReusePlan:
        source_request_id = request.cacheselect_source_request_id
        assert source_request_id is not None
        return GDNDeltaReusePlan(
            source_request_id=source_request_id,
            target_request_id=request.cacheselect_request_id or request.request_id,
            block_size=self.block_size,
            native_cached_tokens=native_cached_tokens,
            reason=reason,
            candidates=candidates,
        )

    # Remove all indexed prompt metadata without touching GPU sidecars.
    def clear(self) -> None:
        self._sources.clear()
