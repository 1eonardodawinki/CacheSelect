# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput


def _create_new_requests_data(prompt_embeds: torch.Tensor | None) -> NewRequestData:
    return NewRequestData(
        req_id="test_req",
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=prompt_embeds,
    )


def test_repr_with_none() -> None:
    """Test repr when prompt_embeds is None."""
    new_requests_data = _create_new_requests_data(None)

    assert "prompt_embeds_shape=None" in repr(new_requests_data)
    assert "prompt_embeds_shape=None" in new_requests_data.anon_repr()


def test_repr_with_multi_element_tensor() -> None:
    """Test repr when prompt_embeds is a multi-element tensor."""
    prompt_embeds = torch.randn(10, 768)
    new_requests_data = _create_new_requests_data(prompt_embeds)

    assert "prompt_embeds_shape=torch.Size([10, 768])" in repr(new_requests_data)
    assert "prompt_embeds_shape=torch.Size([10, 768])" in new_requests_data.anon_repr()


# Check that only the explicitly approved plan enters the worker payload.
def test_from_request_uses_explicit_partial_reuse_plan() -> None:
    plan = object()
    request = SimpleNamespace(
        request_id="test_req",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        num_computed_tokens=0,
        partial_reuse_plan=plan,
    )

    request_data = NewRequestData.from_request(
        request,
        block_ids=([],),
        partial_reuse_plan=plan,
    )

    assert request_data.partial_reuse_plan is plan
    unapproved_request_data = NewRequestData.from_request(request, block_ids=([],))
    assert unapproved_request_data.partial_reuse_plan is None


# Check that scheduler-approved contextual hashes survive the worker payload.
def test_from_request_forwards_contextual_block_hashes() -> None:
    request = SimpleNamespace(
        request_id="test_req",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        num_computed_tokens=0,
    )

    request_data = NewRequestData.from_request(
        request,
        block_ids=([],),
        contextual_block_hashes=(b"first", b"second"),
    )

    assert request_data.contextual_block_hashes == (b"first", b"second")


def test_scheduler_output_exposes_partial_reuse_plans() -> None:
    plan = object()
    request_with_plan = _create_new_requests_data(None)
    request_with_plan.partial_reuse_plan = plan
    request_without_plan = _create_new_requests_data(None)
    request_without_plan.req_id = "request_without_plan"
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [request_with_plan, request_without_plan]

    assert output.partial_reuse_plans == {"test_req": plan}
