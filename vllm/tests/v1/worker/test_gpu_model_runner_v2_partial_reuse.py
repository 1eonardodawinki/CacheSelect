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
from vllm.v1.core.partial_reuse import CacheSelectRepairMetrics
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.partial_reuse import (
    PartialReuseBatchDecision,
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


# Check that the runner sends all prepared spans to its callback in order.
def test_execute_cacheselect_span_steps_in_order() -> None:
    first_step = PartialReuseSpanExecutionStep(
        span=PartialReuseComputeSpan(start_row=0, end_row=2),
        model_inputs={},
        attention_metadata={},
        slot_mappings=torch.tensor([[100, 101]]),
    )
    second_step = PartialReuseSpanExecutionStep(
        span=PartialReuseComputeSpan(start_row=4, end_row=6),
        model_inputs={},
        attention_metadata={},
        slot_mappings=torch.tensor([[104, 105]]),
    )
    callback = Mock(side_effect=("first-output", "second-output"))
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.partial_reuse_span_execution_steps = (first_step, second_step)
    runner._execute_cacheselect_span_step = callback

    outputs = runner._execute_cacheselect_span_steps()

    assert outputs == ("first-output", "second-output")
    assert callback.call_count == 2
    assert callback.call_args_list[0].args[0] is first_step
    assert callback.call_args_list[1].args[0] is second_step


# Check that ordered span outputs are restored to full request row positions.
def test_execute_and_stitch_cacheselect_spans() -> None:
    first_step = PartialReuseSpanExecutionStep(
        span=PartialReuseComputeSpan(start_row=0, end_row=2),
        model_inputs={},
        attention_metadata={},
        slot_mappings=torch.tensor([[100, 101]]),
    )
    second_step = PartialReuseSpanExecutionStep(
        span=PartialReuseComputeSpan(start_row=4, end_row=6),
        model_inputs={},
        attention_metadata={},
        slot_mappings=torch.tensor([[104, 105]]),
    )
    span_outputs = (
        torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
        torch.tensor([[5.0, 5.5], [6.0, 6.5]]),
    )
    execute_spans = Mock(return_value=span_outputs)
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.partial_reuse_span_execution_steps = (first_step, second_step)
    runner._execute_cacheselect_span_steps = execute_spans

    stitched = runner._execute_and_stitch_cacheselect_spans(total_rows=6)

    execute_spans.assert_called_once_with()
    assert stitched.tolist() == [
        [1.0, 1.5],
        [2.0, 2.5],
        [0.0, 0.0],
        [0.0, 0.0],
        [5.0, 5.5],
        [6.0, 6.5],
    ]


# Check that an eligible request executes spans instead of the full fallback.
def test_execute_selected_cacheselect_forward_uses_spans() -> None:
    step = PartialReuseSpanExecutionStep(
        span=PartialReuseComputeSpan(start_row=0, end_row=2),
        model_inputs={},
        attention_metadata={},
        slot_mappings=torch.tensor([[100, 101]]),
    )
    expected_output = torch.tensor([[1.0, 2.0]])
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.partial_reuse_batch_decision = PartialReuseBatchDecision(
        True, "eligible", request_id="rag"
    )
    runner.partial_reuse_span_execution_steps = (step,)
    runner.pending_cacheselect_repair_metrics = {
        "rag": CacheSelectRepairMetrics(
            selector="edit_proximity",
            candidate_tokens=2,
            repair_tokens=0,
            skipped_repair_tokens=2,
            execution_eligible=True,
            execution_reason="eligible",
            reused_batch_rows=2,
            compute_batch_rows=2,
            compute_span_count=1,
            compacted_batch_built=True,
            span_metadata_built=True,
            span_metadata_count=1,
        )
    }
    runner.kv_connector = Mock()
    runner._execute_and_stitch_cacheselect_spans = Mock(
        return_value=expected_output
    )
    execute_full = Mock()
    scheduler_output = SimpleNamespace(name="test-schedule")

    output = runner._execute_selected_cacheselect_forward(
        scheduler_output,
        total_rows=2,
        dummy_run=False,
        execute_full=execute_full,
    )

    assert output is expected_output
    assert runner.partial_reuse_forward_path == "spans"
    runner.kv_connector.pre_forward.assert_called_once_with(scheduler_output)
    runner._execute_and_stitch_cacheselect_spans.assert_called_once_with(2)
    assert runner.pending_cacheselect_repair_metrics[
        "rag"
    ].compacted_batch_executed
    execute_full.assert_not_called()


# Check that an ineligible request preserves vLLM's full forward unchanged.
def test_execute_selected_cacheselect_forward_preserves_fallback() -> None:
    expected_output = torch.tensor([[3.0, 4.0]])
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.partial_reuse_batch_decision = PartialReuseBatchDecision(
        False, "execution_disabled"
    )
    runner.partial_reuse_span_execution_steps = ()
    runner.kv_connector = Mock()
    runner._execute_and_stitch_cacheselect_spans = Mock()
    execute_full = Mock(return_value=expected_output)

    output = runner._execute_selected_cacheselect_forward(
        SimpleNamespace(name="test-schedule"),
        total_rows=2,
        dummy_run=False,
        execute_full=execute_full,
    )

    assert output is expected_output
    assert runner.partial_reuse_forward_path == "full"
    execute_full.assert_called_once_with()
    runner.kv_connector.pre_forward.assert_not_called()
    runner._execute_and_stitch_cacheselect_spans.assert_not_called()


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
