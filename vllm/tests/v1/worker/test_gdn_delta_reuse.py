# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.gdn.delta_cache import (
    GDNDeltaBlockShadowResult,
    GDNDeltaOperatorSidecar,
    GDNDeltaShadowComparison,
)
from vllm.v1.worker.gpu.gdn_delta_reuse import (
    GDNDeltaPreflightResult,
    build_gdn_delta_reuse_candidates,
    collect_gdn_delta_sidecars,
    preflight_gdn_delta_reuse,
    summarize_gdn_delta_shadow_results,
)


class _FakeGDNLayer(torch.nn.Module):
    # Expose the same sidecar attribute used by Qwen GDN layers.
    def __init__(self, sidecar: GDNDeltaOperatorSidecar | None) -> None:
        super().__init__()
        self.gdn_delta_operator_sidecar = sidecar


# Build one small CPU sidecar with the requested operator keys.
def _make_sidecar(*keys: bytes, block_size: int = 2) -> GDNDeltaOperatorSidecar:
    sidecar = GDNDeltaOperatorSidecar(
        capacity=max(len(keys), 1),
        block_size=block_size,
        value_heads=1,
        key_width=2,
        value_width=2,
        dtype=torch.float32,
        device="cpu",
    )
    transition = torch.eye(2).unsqueeze(0)
    responses = torch.ones(block_size, 1, 2)
    state_bias = torch.zeros(1, 2, 2)
    output_biases = torch.zeros(block_size, 1, 2)
    for index, key in enumerate(keys, start=1):
        sidecar.store(
            key,
            transition * index,
            responses * index,
            state_bias,
            output_biases,
        )
    return sidecar


# Build the minimal plan interface consumed by the pure preflight helper.
def _make_plan(*keys: bytes, block_size: int = 2):
    return SimpleNamespace(
        block_size=block_size,
        candidates=tuple(
            SimpleNamespace(source_contextual_hash=key) for key in keys
        ),
    )


class GDNDeltaReusePreflightTests(unittest.TestCase):
    # Map layer-local shadow rows back to request IDs and summarize worst errors.
    def test_summarizes_shadow_results_per_request(self) -> None:
        model = torch.nn.Module()
        layer = _FakeGDNLayer(None)
        layer.last_gdn_delta_shadow_results = (
            GDNDeltaBlockShadowResult(
                sequence_index=1,
                target_block_index=7,
                source_contextual_hash=b"source-seven",
                reason="compared",
                comparison=GDNDeltaShadowComparison(0.25, 0.5, 0.125, 0.25),
            ),
        )
        model.add_module("gdn", layer)

        summaries = summarize_gdn_delta_shadow_results(
            model,
            ("decode-request", "edited-request"),
        )

        summary = summaries["edited-request"]
        self.assertEqual(summary["shadow_result_count"], 1)
        self.assertEqual(summary["shadow_compared_count"], 1)
        self.assertEqual(summary["shadow_max_output_relative_l2"], 0.25)
        self.assertEqual(summary["shadow_max_final_state_relative_l2"], 0.125)
        self.assertEqual(summary["shadow_results"][0]["layer_name"], "gdn")

    # Forward block mappings only for requests whose all-layer preflight succeeded.
    def test_builds_candidates_only_for_eligible_requests(self) -> None:
        plans = {
            "eligible": SimpleNamespace(
                candidates=(
                    SimpleNamespace(
                        target_block_index=7,
                        source_contextual_hash=b"source-seven",
                    ),
                )
            ),
            "rejected": SimpleNamespace(
                candidates=(
                    SimpleNamespace(
                        target_block_index=9,
                        source_contextual_hash=b"source-nine",
                    ),
                )
            ),
        }
        results = {
            "eligible": GDNDeltaPreflightResult(True, "eligible", 1),
            "rejected": GDNDeltaPreflightResult(
                False, "operator_not_resident", 1
            ),
        }

        candidates = build_gdn_delta_reuse_candidates(
            ("eligible", "rejected", "ordinary"), plans, results
        )

        self.assertEqual(candidates, (((7, b"source-seven"),), (), ()))

    # Discover only modules that expose the Qwen GDN sidecar contract.
    def test_discovers_gdn_sidecars_from_model(self) -> None:
        model = torch.nn.Module()
        model.add_module("ordinary", torch.nn.Linear(2, 2))
        sidecar = _make_sidecar(b"a")
        model.add_module("gdn", _FakeGDNLayer(sidecar))
        model.add_module("empty_gdn", _FakeGDNLayer(None))

        layers = collect_gdn_delta_sidecars(model)

        self.assertEqual(layers, (("gdn", sidecar), ("empty_gdn", None)))

    # Resolve the same ordered candidate set across every GDN layer.
    def test_accepts_only_complete_all_layer_residency(self) -> None:
        result = preflight_gdn_delta_reuse(
            _make_plan(b"a", b"b"),
            (
                ("layer.0", _make_sidecar(b"a", b"b")),
                ("layer.1", _make_sidecar(b"a", b"b")),
            ),
        )

        self.assertTrue(result.eligible)
        self.assertEqual(result.reason, "eligible")
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(
            tuple(layer.layer_name for layer in result.layers),
            ("layer.0", "layer.1"),
        )
        self.assertTrue(all(len(layer.entries) == 2 for layer in result.layers))

    # Reject the entire attempt when one layer lacks one planned operator.
    def test_rejects_partial_layer_residency_without_touching_lru(self) -> None:
        complete = _make_sidecar(b"a", b"b")
        incomplete = _make_sidecar(b"a", b"other")

        result = preflight_gdn_delta_reuse(
            _make_plan(b"a", b"b"),
            (("layer.0", complete), ("layer.1", incomplete)),
        )

        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "operator_not_resident")
        self.assertEqual(complete.resident_keys(), (b"a", b"b"))
        self.assertEqual(incomplete.resident_keys(), (b"a", b"other"))

    # Reject incompatible layer geometry instead of reading wrong tensor rows.
    def test_rejects_block_size_mismatch(self) -> None:
        result = preflight_gdn_delta_reuse(
            _make_plan(b"a", block_size=2),
            (("layer.0", _make_sidecar(b"a", block_size=4)),),
        )

        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "block_size_mismatch")


if __name__ == "__main__":
    unittest.main()
