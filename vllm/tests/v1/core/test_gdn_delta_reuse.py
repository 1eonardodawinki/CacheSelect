# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.gdn_delta_reuse import GDNDeltaSourceIndex


# Build one minimal request carrying fine-grained chained prompt hashes.
def _request(request_id: str, token_offset: int = 0):
    return SimpleNamespace(
        request_id=request_id,
        cacheselect_request_id=request_id,
        prompt_token_ids=list(range(token_offset, token_offset + 12)),
        block_hashes=[f"{request_id}-{index}".encode() for index in range(6)],
        cache_salt="experiment",
        lora_request=None,
    )


# Resolve fine hashes to full GDN blocks and preserve their exact token content.
def test_indexes_contextual_hashes_at_gdn_block_size() -> None:
    index = GDNDeltaSourceIndex(block_size=4, hash_block_size=2)

    index.index(_request("source"))

    source = index.get("source")
    assert source is not None
    assert [block.block_index for block in source.blocks] == [0, 1, 2]
    assert [block.token_ids for block in source.blocks] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
    ]
    # Each coarse block uses the last chained hash inside that block.
    assert [block.contextual_hash for block in source.blocks] == [
        b"source-1",
        b"source-3",
        b"source-5",
    ]


# Keep the source metadata bounded using deterministic least-recent eviction.
def test_evicts_least_recent_source_request() -> None:
    index = GDNDeltaSourceIndex(
        block_size=4,
        hash_block_size=4,
        max_source_requests=2,
    )
    index.index(_request("first"))
    index.index(_request("second"))
    assert index.get("first") is not None

    index.index(_request("third"))

    assert index.get("second") is None
    assert index.get("first") is not None
    assert index.get("third") is not None


# Map an unchanged suffix back to the old contextual hashes after an insertion.
def test_locates_aligned_suffix_blocks_after_edit() -> None:
    index = GDNDeltaSourceIndex(block_size=4, hash_block_size=2)
    source = _request("source")
    index.index(source)
    target = SimpleNamespace(
        request_id="target",
        cacheselect_request_id="target",
        cacheselect_source_request_id="source",
        prompt_token_ids=[0, 1, 2, 3, 90, 91, 92, 93, *range(4, 12)],
        cache_salt="experiment",
        lora_request=None,
    )

    plan = index.locate(target, native_cached_tokens=4)

    assert plan is not None
    assert plan.reason == "aligned_candidates"
    assert [
        (candidate.source_block_index, candidate.target_block_index)
        for candidate in plan.candidates
    ] == [(1, 2), (2, 3)]
    assert [candidate.source_contextual_hash for candidate in plan.candidates] == [
        b"source-3",
        b"source-5",
    ]
    assert plan.to_dict()["candidate_block_count"] == 2


# Reject a cross-namespace source rather than looking up another tenant's operator.
def test_rejects_cache_salt_mismatch() -> None:
    index = GDNDeltaSourceIndex(block_size=4, hash_block_size=2)
    index.index(_request("source"))
    target = SimpleNamespace(
        request_id="target",
        cacheselect_request_id="target",
        cacheselect_source_request_id="source",
        prompt_token_ids=list(range(12)),
        cache_salt="different-experiment",
        lora_request=None,
    )

    plan = index.locate(target, native_cached_tokens=0)

    assert plan is not None
    assert plan.reason == "cache_salt_mismatch"
    assert plan.candidates == ()
