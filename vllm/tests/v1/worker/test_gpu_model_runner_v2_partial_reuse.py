# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.partial_reuse import (
    PartialReuseComputeSpan,
    PartialReuseSpanExecutionStep,
)


# Check that one span receives its own attention and KV-write context.
def test_execute_cacheselect_span_step(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_context: dict[str, Any] = {}
    layer_slot_mappings = {"layer.0": torch.tensor([100, 101])}

    # Replace layer mapping expansion with a small observable test double.
    def fake_build_slot_mappings_by_layer(
        slot_mappings: torch.Tensor,
        kv_cache_config: Any,
    ) -> dict[str, torch.Tensor]:
        captured_context["raw_slot_mappings"] = slot_mappings
        captured_context["kv_cache_config"] = kv_cache_config
        return layer_slot_mappings

    # Capture the forward context that would surround the real model call.
    @contextmanager
    def fake_set_forward_context(
        attention_metadata: Any,
        vllm_config: Any,
        **kwargs: Any,
    ) -> Iterator[None]:
        captured_context["attention_metadata"] = attention_metadata
        captured_context["vllm_config"] = vllm_config
        captured_context.update(kwargs)
        yield

    monkeypatch.setattr(
        mrv2,
        "build_slot_mappings_by_layer",
        fake_build_slot_mappings_by_layer,
    )
    monkeypatch.setattr(mrv2, "set_forward_context", fake_set_forward_context)

    model_output = torch.tensor([[1.0, 2.0]])
    model = Mock(return_value=model_output)
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.model = model
    runner.vllm_config = SimpleNamespace(name="test-config")
    runner.kv_cache_config = SimpleNamespace(name="test-cache")
    model_inputs = {
        "input_ids": torch.tensor([10, 11]),
        "positions": torch.tensor([2, 3]),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    raw_slot_mappings = torch.tensor([[100, 101]])
    attention_metadata = {"layer.0": SimpleNamespace(name="test-attention")}
    step = PartialReuseSpanExecutionStep(
        span=PartialReuseComputeSpan(start_row=0, end_row=2),
        model_inputs=model_inputs,
        attention_metadata=attention_metadata,
        slot_mappings=raw_slot_mappings,
    )

    output = runner._execute_cacheselect_span_step(step)

    assert output is model_output
    assert model.call_count == 1
    assert model.call_args.kwargs.keys() == model_inputs.keys()
    for name, value in model_inputs.items():
        assert model.call_args.kwargs[name] is value
    assert captured_context["raw_slot_mappings"] is raw_slot_mappings
    assert captured_context["kv_cache_config"] is runner.kv_cache_config
    assert captured_context["attention_metadata"] is attention_metadata
    assert captured_context["vllm_config"] is runner.vllm_config
    assert captured_context["num_tokens"] == 2
    assert captured_context["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
    assert captured_context["slot_mapping"] is layer_slot_mappings
    assert captured_context["skip_compiled"] is True


# Check that malformed span slot mappings are rejected before model execution.
def test_execute_cacheselect_span_step_rejects_incomplete_slot_mapping() -> None:
    model = Mock()
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.model = model
    step = PartialReuseSpanExecutionStep(
        span=PartialReuseComputeSpan(start_row=0, end_row=2),
        model_inputs={},
        attention_metadata={},
        slot_mappings=torch.tensor([[100]]),
    )

    with pytest.raises(
        ValueError,
        match="span slot mappings must cover every model input row",
    ):
        runner._execute_cacheselect_span_step(step)

    model.assert_not_called()
