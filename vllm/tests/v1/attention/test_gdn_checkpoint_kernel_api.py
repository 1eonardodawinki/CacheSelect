# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the inference-only GDN checkpoint-state API."""

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops import chunk as chunk_module


# Verify that the narrow inference API returns states normally hidden by FLA.
def test_checkpoint_api_returns_intermediate_states(monkeypatch):
    """The wrapper should request and expose the kernel's chunk states."""
    query = torch.zeros((1, 2, 1, 4), dtype=torch.bfloat16)
    final_state = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    chunk_states = torch.full((1, 2, 1, 4, 4), 2, dtype=torch.bfloat16)
    captured: dict[str, object] = {}

    # Replace the CUDA kernel so this API contract stays testable on CPU.
    def fake_forward(**kwargs):
        captured.update(kwargs)
        output = kwargs["q"] + 3
        return None, output, None, final_state, None, chunk_states, None

    monkeypatch.setattr(chunk_module, "chunk_gated_delta_rule_fwd", fake_forward)

    with torch.inference_mode():
        output, returned_final, returned_chunks = (
            chunk_module.chunk_gated_delta_rule_inference_with_states(
                q=query,
                k=query,
                v=query,
                g=torch.zeros((1, 2, 1), dtype=torch.bfloat16),
                beta=torch.zeros((1, 2, 1), dtype=torch.bfloat16),
            )
        )

    assert captured["output_intermediate_states"] is True
    assert torch.equal(output, query + 3)
    assert returned_final is final_state
    assert returned_chunks is chunk_states


# Verify that training code cannot accidentally retain the large state tensor.
def test_checkpoint_api_rejects_grad_enabled_execution():
    """Intermediate checkpoint extraction must remain inference-only."""
    query = torch.zeros((1, 2, 1, 4), dtype=torch.bfloat16)

    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        chunk_module.chunk_gated_delta_rule_inference_with_states(
            q=query,
            k=query,
            v=query,
            g=torch.zeros((1, 2, 1), dtype=torch.bfloat16),
            beta=torch.zeros((1, 2, 1), dtype=torch.bfloat16),
        )
