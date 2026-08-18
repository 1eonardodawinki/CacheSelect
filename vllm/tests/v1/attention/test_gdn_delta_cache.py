# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import torch

from vllm.model_executor.layers.mamba.gdn.delta_cache import (
    GDNDeltaOperatorSidecar,
    build_gdn_delta_operator,
    store_completed_gdn_delta_operators,
)


class GDNDeltaOperatorSidecarTests(unittest.TestCase):
    # Store only complete blocks while respecting each sequence's absolute offset.
    def test_stores_complete_prefill_blocks_by_contextual_hash(self) -> None:
        sidecar = GDNDeltaOperatorSidecar(
            capacity=5,
            block_size=2,
            value_heads=2,
            key_width=3,
            dtype=torch.float32,
            device="cpu",
        )
        keys = torch.randn(10, 1, 3)
        queries = torch.randn(10, 1, 3)
        log_decays = -torch.rand(10, 2)
        betas = torch.rand(10, 2)

        stored = store_completed_gdn_delta_operators(
            sidecar,
            keys=keys,
            queries=queries,
            log_decays=log_decays,
            betas=betas,
            query_start_locations=torch.tensor([0, 6, 10]),
            num_computed_tokens=torch.tensor([0, 2]),
            contextual_block_hashes=(
                (b"a0", b"a1", b"a2"),
                (b"b0", b"b1", b"b2"),
            ),
        )

        self.assertEqual(stored, ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)))
        self.assertEqual(
            sidecar.resident_keys(),
            (b"a0", b"a1", b"a2", b"b1", b"b2"),
        )

    # Skip a leading partial block and cache later blocks that are fully covered.
    def test_skips_partial_prefill_block(self) -> None:
        sidecar = GDNDeltaOperatorSidecar(
            capacity=2,
            block_size=2,
            value_heads=1,
            key_width=2,
            dtype=torch.float32,
            device="cpu",
        )
        keys = torch.randn(5, 1, 2)
        queries = torch.randn(5, 1, 2)

        stored = store_completed_gdn_delta_operators(
            sidecar,
            keys=keys,
            queries=queries,
            log_decays=-torch.rand(5, 1),
            betas=torch.rand(5, 1),
            query_start_locations=torch.tensor([0, 5]),
            num_computed_tokens=torch.tensor([1]),
            contextual_block_hashes=((b"partial", b"full-1", b"full-2"),),
        )

        self.assertEqual(stored, ((0, 1), (0, 2)))
        self.assertEqual(sidecar.resident_keys(), (b"full-1", b"full-2"))

    # Match the composed operator against direct token-by-token delta propagation.
    def test_block_operator_matches_direct_delta_recurrence(self) -> None:
        generator = torch.Generator().manual_seed(17)
        token_count, key_heads, value_heads = 5, 2, 4
        key_width, value_width = 3, 6
        keys = torch.randn(token_count, key_heads, key_width, generator=generator)
        queries = torch.randn(
            token_count, key_heads, key_width, generator=generator
        )
        log_decays = -torch.rand(token_count, value_heads, generator=generator)
        betas = torch.rand(token_count, value_heads, generator=generator)
        initial_delta = torch.randn(
            value_heads, value_width, key_width, generator=generator
        )

        transition, responses = build_gdn_delta_operator(
            keys, queries, log_decays, betas
        )
        composed_final = torch.einsum("hvk,hkl->hvl", initial_delta, transition)
        composed_outputs = torch.einsum("hvk,thk->thv", initial_delta, responses)

        expanded_keys = keys.repeat_interleave(value_heads // key_heads, dim=1)
        expanded_queries = queries.repeat_interleave(
            value_heads // key_heads, dim=1
        )
        direct_delta = initial_delta
        direct_outputs = []
        for token_index in range(token_count):
            key = expanded_keys[token_index]
            decayed = direct_delta * log_decays[token_index].exp()[:, None, None]
            read = torch.einsum("hvk,hk->hv", decayed, key)
            direct_delta = decayed - (
                betas[token_index, :, None, None]
                * read[:, :, None]
                * key[:, None, :]
            )
            direct_outputs.append(
                torch.einsum(
                    "hvk,hk->hv", direct_delta, expanded_queries[token_index]
                )
                * key_width**-0.5
            )

        torch.testing.assert_close(composed_final, direct_delta)
        torch.testing.assert_close(composed_outputs, torch.stack(direct_outputs))

    # Store and retrieve an operator without copying its resident tensor views.
    def test_stores_and_resolves_operator(self) -> None:
        cache = GDNDeltaOperatorSidecar(
            capacity=2,
            block_size=3,
            value_heads=2,
            key_width=4,
            dtype=torch.float32,
            device="cpu",
        )
        transition = torch.arange(32, dtype=torch.float32).view(2, 4, 4)
        responses = torch.arange(24, dtype=torch.float32).view(3, 2, 4)

        slot = cache.store(b"context:block-a", transition, responses)
        entry = cache.lookup(b"context:block-a")

        self.assertEqual(slot, 0)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.slot, slot)
        torch.testing.assert_close(entry.transition, transition)
        torch.testing.assert_close(entry.output_responses, responses)

    # Resolve every requested operator together and refresh their LRU order.
    def test_resolves_complete_operator_set_atomically(self) -> None:
        cache = GDNDeltaOperatorSidecar(
            capacity=2,
            block_size=1,
            value_heads=1,
            key_width=2,
            dtype=torch.float32,
            device="cpu",
        )
        transition = torch.eye(2).unsqueeze(0)
        responses = torch.ones(1, 1, 2)
        cache.store(b"a", transition, responses)
        cache.store(b"b", transition * 2, responses * 2)

        entries = cache.lookup_many((b"b", b"a"))

        self.assertIsNotNone(entries)
        assert entries is not None
        self.assertEqual(tuple(entry.slot for entry in entries), (1, 0))
        self.assertEqual(cache.resident_keys(), (b"b", b"a"))

    # Leave recency unchanged when any requested operator is absent.
    def test_atomic_lookup_miss_does_not_refresh_partial_hits(self) -> None:
        cache = GDNDeltaOperatorSidecar(
            capacity=2,
            block_size=1,
            value_heads=1,
            key_width=2,
            dtype=torch.float32,
            device="cpu",
        )
        transition = torch.eye(2).unsqueeze(0)
        responses = torch.ones(1, 1, 2)
        cache.store(b"a", transition, responses)
        cache.store(b"b", transition * 2, responses * 2)

        self.assertIsNone(cache.lookup_many((b"a", b"missing")))
        self.assertEqual(cache.resident_keys(), (b"a", b"b"))
        cache.store(b"c", transition * 3, responses * 3)
        self.assertEqual(cache.resident_keys(), (b"b", b"c"))

    # Evict the least-recent entry while preserving a recently accessed hash.
    def test_uses_bounded_lru_eviction(self) -> None:
        cache = GDNDeltaOperatorSidecar(
            capacity=2,
            block_size=1,
            value_heads=1,
            key_width=2,
            dtype=torch.float32,
            device="cpu",
        )
        transition = torch.eye(2).unsqueeze(0)
        responses = torch.ones(1, 1, 2)
        cache.store(b"a", transition, responses)
        cache.store(b"b", transition * 2, responses * 2)
        self.assertIsNotNone(cache.lookup(b"a"))

        reused_slot = cache.store(b"c", transition * 3, responses * 3)

        self.assertEqual(reused_slot, 1)
        self.assertIsNone(cache.lookup(b"b"))
        self.assertEqual(cache.resident_keys(), (b"a", b"c"))
        self.assertEqual(len(cache), 2)

    # Refresh an existing hash in place and reset metadata without reallocating.
    def test_refreshes_and_clears_entries(self) -> None:
        cache = GDNDeltaOperatorSidecar(
            capacity=1,
            block_size=1,
            value_heads=1,
            key_width=1,
            dtype=torch.float32,
            device="cpu",
        )
        slot = cache.store(b"same", torch.ones(1, 1, 1), torch.ones(1, 1, 1))
        refreshed_slot = cache.store(
            b"same", torch.full((1, 1, 1), 4.0), torch.full((1, 1, 1), 5.0)
        )

        self.assertEqual(refreshed_slot, slot)
        entry = cache.lookup(b"same")
        assert entry is not None
        self.assertEqual(entry.transition.item(), 4.0)
        self.assertEqual(entry.output_responses.item(), 5.0)
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.lookup(b"same"))


if __name__ == "__main__":
    unittest.main()
