# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.mlp_repair_model import MLPRepairModel
from vllm.v1.worker.gpu.partial_reuse import (
    EditProximityRepairSelector,
    FullBlockRepairSelector,
    MLPRepairSelector,
    PartialReuseBatchDecision,
    PartialReuseCompactedBatch,
    PartialReuseComputeSpan,
    PartialReuseCopyInstruction,
    PartialReuseRepairInstruction,
    PartialReuseSpanAttentionInputs,
    PartialReuseSpanExecutionStep,
    PartialReuseSpanInputs,
    ResolvedPartialReuseCandidate,
    assess_partial_reuse_batch,
    build_counterfactual_repair_instructions,
    build_full_block_repair_instructions,
    build_kv_cache_block_copies,
    build_partial_reuse_compacted_batch,
    build_partial_reuse_compute_rows,
    build_partial_reuse_compute_spans,
    build_partial_reuse_copy_instructions,
    build_partial_reuse_span_attention_inputs,
    build_partial_reuse_span_attention_metadata,
    build_partial_reuse_span_execution_steps,
    build_partial_reuse_span_input_sequence,
    build_partial_reuse_span_inputs,
    build_partial_reuse_span_model_input_sequence,
    build_partial_reuse_span_model_inputs,
    build_reused_token_indices,
    compact_partial_reuse_model_inputs,
    compact_partial_reuse_query_start_locations,
    compact_partial_reuse_slot_mappings,
    correct_qwen3_kv_positions_inplace,
    create_repair_selector,
    execute_partial_reuse_span_steps,
    map_reused_tokens_to_batch_rows,
    record_batch_execution_decision,
    record_compacted_batch_construction,
    record_compacted_batch_execution,
    record_copy_execution,
    record_gpu_execution_times,
    record_preparation_time,
    record_span_attention_metadata_construction,
    repack_kv_cache_blocks_inplace,
    resolve_target_block_ids,
    select_partial_reuse_forward_path,
    stitch_partial_reuse_span_outputs,
    summarize_repair_selection,
)


# Check that one logical target position resolves to its physical block ID.
def test_resolve_target_block_ids() -> None:
    candidate = SimpleNamespace(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        source_resident=True,
        requires_repair=True,
        block_displacement=2,
        nearest_changed_block_distance=1,
    )
    plan = SimpleNamespace(candidates=(candidate,))

    resolved = resolve_target_block_ids(plan, ([71, 12, 89, 34, 55, 63],))

    assert resolved == (
        ResolvedPartialReuseCandidate(
            source_block_index=3,
            target_block_index=5,
            source_block_id=42,
            target_block_id=63,
            source_resident=True,
            requires_repair=True,
            block_displacement=2,
            nearest_changed_block_distance=1,
        ),
    )


# Check that the resolver rejects a target position the request does not own.
def test_resolve_target_block_ids_rejects_missing_target() -> None:
    candidate = SimpleNamespace(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        source_resident=True,
        requires_repair=True,
    )
    plan = SimpleNamespace(candidates=(candidate,))

    with pytest.raises(ValueError, match="outside the request block table"):
        resolve_target_block_ids(plan, ([71, 12],))


# Check that chunked prefill defers mappings for target blocks not allocated yet.
def test_resolve_target_block_ids_defers_unallocated_chunk() -> None:
    candidate = SimpleNamespace(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        source_resident=True,
        requires_repair=True,
        block_displacement=2,
        nearest_changed_block_distance=1,
    )
    plan = SimpleNamespace(candidates=(candidate,))

    resolved = resolve_target_block_ids(
        plan,
        ([71, 12],),
        allow_unallocated=True,
    )

    assert resolved == ()


# Check that resolved mappings become explicit worker copy instructions.
def test_build_partial_reuse_copy_instructions() -> None:
    candidate = ResolvedPartialReuseCandidate(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        target_block_id=63,
        source_resident=True,
        requires_repair=True,
    )

    instructions = build_partial_reuse_copy_instructions((candidate,))

    assert instructions == (
        PartialReuseCopyInstruction(
            source_block_id=42,
            target_block_id=63,
            target_block_index=5,
            requires_repair=True,
            source_block_index=3,
        ),
    )


# Check that worker instructions become vLLM physical block-copy pairs.
def test_build_kv_cache_block_copies() -> None:
    instructions = (
        PartialReuseCopyInstruction(
            source_block_id=42,
            target_block_id=63,
            target_block_index=5,
            requires_repair=True,
        ),
        PartialReuseCopyInstruction(
            source_block_id=43,
            target_block_id=64,
            target_block_index=6,
            requires_repair=False,
        ),
    )

    assert build_kv_cache_block_copies(instructions) == ((42, 63), (43, 64))


# Check that two source-page slices assemble one complete target cache block.
def test_repack_kv_cache_blocks_inplace() -> None:
    caches = []
    originals = []
    for layer in range(2):
        cache = torch.arange(4 * 2 * 4).reshape(4, 2, 4, 1) + layer * 100
        caches.append(cache)
        originals.append(cache.clone())
    instruction = PartialReuseCopyInstruction(
        source_block_id=0,
        source_block_ids=(0, 1),
        source_block_offset=2,
        target_block_id=2,
        target_block_index=3,
        requires_repair=False,
    )

    repack_kv_cache_blocks_inplace(caches, (instruction,), block_size=4)

    for cache, original in zip(caches, originals, strict=True):
        expected = torch.cat(
            (original[0, :, 2:, :], original[1, :, :2, :]), dim=1
        )
        assert torch.equal(cache[2], expected)
        assert torch.equal(cache[:2], original[:2])


# Check that moved Qwen3 keys are re-rotated while values remain unchanged.
def test_correct_qwen3_kv_positions_inplace() -> None:
    cache = torch.zeros(4, 1, 2, 8)
    cache[3, :, :, :4] = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )
    cache[3, :, :, 4:] = 9
    original_keys = cache[3, :, :, :4].clone()
    original_values = cache[3, :, :, 4:].clone()
    instruction = PartialReuseCopyInstruction(
        source_block_id=1,
        target_block_id=3,
        target_block_index=3,
        requires_repair=False,
        source_block_index=1,
        source_block_offset=1,
    )

    correct_qwen3_kv_positions_inplace(
        (cache,), (instruction,), block_size=2, head_size=4, rope_theta=100
    )

    angle = torch.tensor([3.0, 0.3])
    first, second = original_keys.chunk(2, dim=-1)
    expected = torch.cat(
        (
            first * angle.cos() - second * angle.sin(),
            second * angle.cos() + first * angle.sin(),
        ),
        dim=-1,
    )
    assert torch.allclose(cache[3, :, :, :4], expected)
    assert torch.equal(cache[3, :, :, 4:], original_values)


# Check that a repacked instruction cannot enter the whole-page copy path.
def test_block_copy_rejects_repacking_instruction() -> None:
    instruction = PartialReuseCopyInstruction(
        1, 3, 2, False, source_block_ids=(1, 2), source_block_offset=1
    )

    with pytest.raises(ValueError, match="require KV repacking"):
        build_kv_cache_block_copies((instruction,))


# Check that the fallback selector chooses every token in affected blocks.
def test_full_block_repair_selector() -> None:
    repaired_candidate = ResolvedPartialReuseCandidate(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        target_block_id=63,
        source_resident=True,
        requires_repair=True,
    )
    exact_candidate = ResolvedPartialReuseCandidate(
        source_block_index=4,
        target_block_index=6,
        source_block_id=43,
        target_block_id=64,
        source_resident=True,
        requires_repair=False,
    )

    selector = FullBlockRepairSelector()
    instructions = selector.select((repaired_candidate, exact_candidate), block_size=4)

    assert instructions == (
        PartialReuseRepairInstruction(
            source_block_id=42,
            target_block_id=63,
            target_block_index=5,
            target_token_indices=(20, 21, 22, 23),
        ),
    )


# Check that an invalid cache block size cannot create a repair range.
def test_build_full_block_repair_rejects_invalid_block_size() -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        build_full_block_repair_instructions((), block_size=0)


# Check one absolute target index is reused while every peer is repaired.
def test_build_counterfactual_repair_instructions() -> None:
    candidate = ResolvedPartialReuseCandidate(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        target_block_id=63,
        source_resident=True,
        requires_repair=True,
    )
    candidates = (
        candidate,
        replace(
            candidate,
            source_block_index=6,
            target_block_index=8,
            source_block_id=43,
            target_block_id=64,
        ),
        replace(
            candidate,
            source_block_index=9,
            target_block_index=11,
            source_block_id=44,
            target_block_id=65,
        ),
    )

    repairs = build_counterfactual_repair_instructions(
        candidates, block_size=4, reused_target_block_index=8
    )
    copies = build_partial_reuse_copy_instructions(candidates)

    assert [repair.target_block_index for repair in repairs] == [5, 11]
    assert build_reused_token_indices(copies, repairs, block_size=4) == (
        32,
        33,
        34,
        35,
    )
    with pytest.raises(ValueError, match="exactly one resolved candidate"):
        build_counterfactual_repair_instructions(
            candidates, block_size=4, reused_target_block_index=7
        )


# Check that edit proximity repairs nearby and unknown blocks but skips far ones.
def test_edit_proximity_repair_selector() -> None:
    nearby_candidate = ResolvedPartialReuseCandidate(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        target_block_id=63,
        source_resident=True,
        requires_repair=True,
        nearest_changed_block_distance=1,
    )
    far_candidate = replace(
        nearby_candidate,
        target_block_index=6,
        target_block_id=64,
        nearest_changed_block_distance=2,
    )
    unknown_candidate = replace(
        nearby_candidate,
        target_block_index=7,
        target_block_id=65,
        nearest_changed_block_distance=None,
    )

    selector = EditProximityRepairSelector(max_block_distance=1)
    instructions = selector.select(
        (nearby_candidate, far_candidate, unknown_candidate), block_size=2
    )

    assert [instruction.target_block_id for instruction in instructions] == [63, 65]
    assert [instruction.target_token_indices for instruction in instructions] == [
        (10, 11),
        (14, 15),
    ]
    copy_instructions = build_partial_reuse_copy_instructions(
        (nearby_candidate, far_candidate, unknown_candidate)
    )
    assert build_reused_token_indices(
        copy_instructions,
        instructions,
        block_size=2,
    ) == (12, 13)


# Check prompt positions map into the correct request slice of a flat batch.
def test_map_reused_tokens_to_batch_rows() -> None:
    rows = map_reused_tokens_to_batch_rows(
        req_ids=("decode", "rag"),
        query_start_locations=(0, 4, 148),
        num_computed_tokens=(200, 32),
        num_scheduled_tokens=(4, 144),
        reused_token_indices={
            "rag": (*range(96, 128), 200),
        },
    )

    assert rows == {"rag": tuple(range(68, 100))}


# Check that the narrow first execution scope accepts a simple prefill batch.
def test_assess_partial_reuse_batch_accepts_supported_prefill() -> None:
    decision = assess_partial_reuse_batch(
        execution_enabled=True,
        req_ids=("rag",),
        is_prefilling=(True,),
        reused_batch_rows={"rag": tuple(range(64, 96))},
        required_output_rows=(143,),
        prompt_logprobs=False,
        single_gpu=True,
        eager_execution=True,
        supported_kv_layout=True,
        speculative_decoding=False,
        multimodal_model=False,
        encoder_decoder_model=False,
        pooling_model=False,
    )

    assert decision == PartialReuseBatchDecision(
        True,
        "eligible",
        request_id="rag",
        reused_batch_rows=tuple(range(64, 96)),
    )


# Check that an otherwise valid reuse plan falls back for a mixed request batch.
def test_assess_partial_reuse_batch_rejects_batched_requests() -> None:
    decision = assess_partial_reuse_batch(
        execution_enabled=True,
        req_ids=("decode", "rag"),
        is_prefilling=(False, True),
        reused_batch_rows={"rag": tuple(range(64, 96))},
        required_output_rows=(3, 147),
        prompt_logprobs=False,
        single_gpu=True,
        eager_execution=True,
        supported_kv_layout=True,
        speculative_decoding=False,
        multimodal_model=False,
        encoder_decoder_model=False,
        pooling_model=False,
    )

    assert decision == PartialReuseBatchDecision(
        False,
        "batched_requests_unsupported",
        request_id="rag",
        reused_batch_rows=tuple(range(64, 96)),
    )


# Check that compiled execution falls back before dynamic spans are attempted.
def test_assess_partial_reuse_batch_rejects_non_eager_execution() -> None:
    decision = assess_partial_reuse_batch(
        execution_enabled=True,
        req_ids=("rag",),
        is_prefilling=(True,),
        reused_batch_rows={"rag": tuple(range(64, 96))},
        required_output_rows=(143,),
        prompt_logprobs=False,
        single_gpu=True,
        eager_execution=False,
        supported_kv_layout=True,
        speculative_decoding=False,
        multimodal_model=False,
        encoder_decoder_model=False,
        pooling_model=False,
    )

    assert decision == PartialReuseBatchDecision(
        False,
        "non_eager_execution_unsupported",
        request_id="rag",
        reused_batch_rows=tuple(range(64, 96)),
    )


# Check that the sampled output row is removed from the selected reuse rows.
def test_assess_partial_reuse_batch_protects_reused_output_row() -> None:
    decision = assess_partial_reuse_batch(
        execution_enabled=True,
        req_ids=("rag",),
        is_prefilling=(True,),
        reused_batch_rows={"rag": tuple(range(64, 96))},
        required_output_rows=(95,),
        prompt_logprobs=False,
        single_gpu=True,
        eager_execution=True,
        supported_kv_layout=True,
        speculative_decoding=False,
        multimodal_model=False,
        encoder_decoder_model=False,
        pooling_model=False,
    )

    assert decision == PartialReuseBatchDecision(
        True,
        "eligible",
        request_id="rag",
        reused_batch_rows=tuple(range(64, 95)),
    )


# Check that prompt-token scoring falls back until reused rows have hidden states.
def test_assess_partial_reuse_batch_rejects_prompt_logprobs() -> None:
    decision = assess_partial_reuse_batch(
        execution_enabled=True,
        req_ids=("rag",),
        is_prefilling=(True,),
        reused_batch_rows={"rag": tuple(range(64, 96))},
        required_output_rows=(143,),
        prompt_logprobs=True,
        single_gpu=True,
        eager_execution=True,
        supported_kv_layout=True,
        speculative_decoding=False,
        multimodal_model=False,
        encoder_decoder_model=False,
        pooling_model=False,
    )

    assert decision == PartialReuseBatchDecision(
        False,
        "prompt_logprobs_unsupported",
        request_id="rag",
        reused_batch_rows=tuple(range(64, 96)),
    )


# Check that a safe eligible request selects span-based execution.
def test_select_partial_reuse_forward_path_uses_spans() -> None:
    decision = PartialReuseBatchDecision(True, "eligible", request_id="rag")
    step = PartialReuseSpanExecutionStep(
        span=PartialReuseComputeSpan(start_row=0, end_row=2),
        model_inputs={},
        attention_metadata={},
        slot_mappings=torch.tensor([[100, 101]]),
    )

    path = select_partial_reuse_forward_path(
        decision,
        (step,),
        dummy_run=False,
    )

    assert path == "spans"


# Check that an ineligible request stays on vLLM's normal full forward.
def test_select_partial_reuse_forward_path_preserves_fallback() -> None:
    decision = PartialReuseBatchDecision(False, "execution_disabled")

    path = select_partial_reuse_forward_path(
        decision,
        (),
        dummy_run=False,
    )

    assert path == "full"


# Check that profiling and warm-up runs never use request-specific span plans.
def test_select_partial_reuse_forward_path_preserves_dummy_run() -> None:
    decision = PartialReuseBatchDecision(True, "eligible", request_id="rag")

    path = select_partial_reuse_forward_path(
        decision,
        (),
        dummy_run=True,
    )

    assert path == "full"


# Check that an eligible decision cannot silently run without prepared spans.
def test_select_partial_reuse_forward_path_requires_steps() -> None:
    decision = PartialReuseBatchDecision(True, "eligible", request_id="rag")

    with pytest.raises(
        ValueError,
        match="eligible partial reuse requires execution steps",
    ):
        select_partial_reuse_forward_path(
            decision,
            (),
            dummy_run=False,
        )


# Check that eligible reused rows are omitted from the future compute batch.
def test_build_partial_reuse_compute_rows_omits_reused_rows() -> None:
    decision = PartialReuseBatchDecision(
        True,
        "eligible",
        request_id="rag",
        reused_batch_rows=tuple(range(64, 96)),
    )

    compute_rows = build_partial_reuse_compute_rows(decision, num_tokens=144)

    assert compute_rows == (*range(64), *range(96, 144))


# Check that a fallback decision leaves the full input batch unchanged.
def test_build_partial_reuse_compute_rows_preserves_fallback_batch() -> None:
    decision = PartialReuseBatchDecision(False, "execution_disabled")

    assert build_partial_reuse_compute_rows(decision, num_tokens=4) == (0, 1, 2, 3)


# Check that an eligible decision cannot reference rows outside its input batch.
def test_build_partial_reuse_compute_rows_rejects_invalid_rows() -> None:
    decision = PartialReuseBatchDecision(
        True,
        "eligible",
        request_id="rag",
        reused_batch_rows=(2, 4),
    )

    with pytest.raises(ValueError, match="outside the input batch"):
        build_partial_reuse_compute_rows(decision, num_tokens=4)


# Check that one reusable middle region creates two safe compute spans.
def test_build_partial_reuse_compute_spans() -> None:
    spans = build_partial_reuse_compute_spans(
        compute_rows=(*range(64), *range(96, 144)),
        num_rows=144,
    )

    assert spans == (
        PartialReuseComputeSpan(start_row=0, end_row=64),
        PartialReuseComputeSpan(start_row=96, end_row=144),
    )


# Check that compaction keeps matching token IDs and absolute positions.
def test_compact_partial_reuse_model_inputs() -> None:
    input_ids = torch.tensor([10, 11, 12, 13, 14, 15])
    positions = torch.tensor([20, 21, 22, 23, 24, 25])

    compacted_ids, compacted_positions = compact_partial_reuse_model_inputs(
        input_ids,
        positions,
        compute_rows=(0, 1, 4, 5),
    )

    assert compacted_ids.tolist() == [10, 11, 14, 15]
    assert compacted_positions.tolist() == [20, 21, 24, 25]
    assert input_ids.tolist() == [10, 11, 12, 13, 14, 15]
    assert positions.tolist() == [20, 21, 22, 23, 24, 25]


# Check that KV destinations are compacted along the matching token rows.
def test_compact_partial_reuse_slot_mappings() -> None:
    slot_mappings = torch.tensor(
        [
            [100, 101, 102, 103, 104, 105],
            [200, 201, 202, 203, 204, 205],
        ]
    )

    compacted = compact_partial_reuse_slot_mappings(
        slot_mappings,
        compute_rows=(0, 1, 4, 5),
    )

    assert compacted.tolist() == [
        [100, 101, 104, 105],
        [200, 201, 204, 205],
    ]
    assert slot_mappings.shape == (2, 6)


# Check that packed attention boundaries retain each request's row count.
def test_compact_partial_reuse_query_start_locations() -> None:
    compacted = compact_partial_reuse_query_start_locations(
        query_start_locations=(0, 3, 6),
        compute_rows=(0, 2, 3, 5),
    )

    assert compacted == (0, 2, 4)


# Check that one compacted plan keeps every model input aligned.
def test_build_partial_reuse_compacted_batch() -> None:
    compacted = build_partial_reuse_compacted_batch(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        positions=torch.tensor([20, 21, 22, 23, 24, 25]),
        slot_mappings=torch.tensor([[100, 101, 102, 103, 104, 105]]),
        block_tables=(torch.tensor([[7, 8, 9], [10, 11, 12]]),),
        query_start_locations=(0, 3, 6),
        compute_rows=(0, 2, 3, 5),
        initial_computed_tokens=20,
    )

    assert isinstance(compacted, PartialReuseCompactedBatch)
    assert compacted.compute_rows == (0, 2, 3, 5)
    assert compacted.compute_spans == (
        PartialReuseComputeSpan(start_row=0, end_row=1),
        PartialReuseComputeSpan(start_row=2, end_row=4),
        PartialReuseComputeSpan(start_row=5, end_row=6),
    )
    assert [span.sequence_length for span in compacted.span_inputs] == [21, 24, 26]
    assert [
        model_inputs["positions"].tolist()
        for model_inputs in compacted.span_model_inputs
    ] == [[20], [22, 23], [25]]
    assert [
        item.max_seq_len for item in compacted.span_attention_inputs
    ] == [21, 24, 26]
    assert compacted.input_ids.tolist() == [10, 12, 13, 15]
    assert compacted.positions.tolist() == [20, 22, 23, 25]
    assert compacted.slot_mappings.tolist() == [[100, 102, 103, 105]]
    assert compacted.query_start_locations == (0, 2, 4)


# Check that the first span follows the native prefix and keeps aligned inputs.
def test_build_partial_reuse_span_inputs() -> None:
    span_inputs = build_partial_reuse_span_inputs(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7]),
        slot_mappings=torch.tensor([[100, 101, 102, 103, 104, 105]]),
        span=PartialReuseComputeSpan(start_row=0, end_row=2),
        initial_computed_tokens=2,
    )

    assert isinstance(span_inputs, PartialReuseSpanInputs)
    assert span_inputs.input_ids.tolist() == [10, 11]
    assert span_inputs.positions.tolist() == [2, 3]
    assert span_inputs.slot_mappings.tolist() == [[100, 101]]
    assert span_inputs.query_start_locations == (0, 2)
    assert span_inputs.sequence_length == 4


# Check that the second span sees the computed and reused context before it.
def test_build_partial_reuse_span_input_sequence() -> None:
    span_inputs = build_partial_reuse_span_input_sequence(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7]),
        slot_mappings=torch.tensor([[100, 101, 102, 103, 104, 105]]),
        spans=(
            PartialReuseComputeSpan(start_row=0, end_row=2),
            PartialReuseComputeSpan(start_row=4, end_row=6),
        ),
        initial_computed_tokens=2,
    )

    first_span, second_span = span_inputs
    assert first_span.input_ids.tolist() == [10, 11]
    assert first_span.sequence_length == 4
    assert second_span.input_ids.tolist() == [14, 15]
    assert second_span.positions.tolist() == [6, 7]
    assert second_span.slot_mappings.tolist() == [[104, 105]]
    assert second_span.query_start_locations == (0, 2)
    assert second_span.sequence_length == 8


# Check that one span becomes complete inputs for vLLM's attention builder.
def test_build_partial_reuse_span_attention_inputs() -> None:
    span_inputs = build_partial_reuse_span_inputs(
        input_ids=torch.tensor([10, 11, 12, 13]),
        positions=torch.tensor([2, 3, 4, 5]),
        slot_mappings=torch.tensor([[100, 101, 102, 103]]),
        span=PartialReuseComputeSpan(start_row=0, end_row=2),
        initial_computed_tokens=2,
    )

    attention_inputs = build_partial_reuse_span_attention_inputs(
        span_inputs,
        block_tables=(torch.tensor([[7, 8, 9], [10, 11, 12]]),),
    )

    assert isinstance(attention_inputs, PartialReuseSpanAttentionInputs)
    assert attention_inputs.num_tokens == 2
    assert attention_inputs.query_start_loc_cpu.tolist() == [0, 2]
    assert attention_inputs.query_start_loc_gpu.tolist() == [0, 2]
    assert attention_inputs.seq_lens.tolist() == [4]
    assert attention_inputs.max_query_len == 2
    assert attention_inputs.max_seq_len == 4
    assert attention_inputs.block_tables[0].tolist() == [[7, 8, 9]]
    assert attention_inputs.slot_mappings.tolist() == [[100, 101]]
    assert attention_inputs.positions.tolist() == [2, 3]


# Check that one span becomes the keyword arguments expected by a decoder model.
def test_build_partial_reuse_span_model_inputs() -> None:
    span_inputs = build_partial_reuse_span_inputs(
        input_ids=torch.tensor([10, 11, 12, 13]),
        positions=torch.tensor([2, 3, 4, 5]),
        slot_mappings=torch.tensor([[100, 101, 102, 103]]),
        span=PartialReuseComputeSpan(start_row=1, end_row=3),
        initial_computed_tokens=2,
    )

    model_inputs = build_partial_reuse_span_model_inputs(span_inputs)

    assert set(model_inputs) == {
        "input_ids",
        "positions",
        "inputs_embeds",
        "intermediate_tensors",
    }
    assert model_inputs["input_ids"].tolist() == [11, 12]
    assert model_inputs["positions"].tolist() == [3, 4]
    assert model_inputs["inputs_embeds"] is None
    assert model_inputs["intermediate_tensors"] is None


# Check that multiple model calls preserve their causal span order.
def test_build_partial_reuse_span_model_input_sequence() -> None:
    span_inputs = build_partial_reuse_span_input_sequence(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7]),
        slot_mappings=torch.tensor([[100, 101, 102, 103, 104, 105]]),
        spans=(
            PartialReuseComputeSpan(start_row=0, end_row=2),
            PartialReuseComputeSpan(start_row=4, end_row=6),
        ),
        initial_computed_tokens=2,
    )

    model_inputs = build_partial_reuse_span_model_input_sequence(span_inputs)

    assert [item["input_ids"].tolist() for item in model_inputs] == [
        [10, 11],
        [14, 15],
    ]
    assert [item["positions"].tolist() for item in model_inputs] == [
        [2, 3],
        [6, 7],
    ]


# Check that advisory metadata calls vLLM's builder once for each span.
def test_build_partial_reuse_span_attention_metadata(monkeypatch) -> None:
    calls = []

    # Capture builder inputs without requiring a configured attention backend.
    def fake_build_attn_metadata(**kwargs):
        calls.append(kwargs)
        return {"span_tokens": kwargs["num_tokens"]}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.partial_reuse.build_attn_metadata",
        fake_build_attn_metadata,
    )
    span_inputs = build_partial_reuse_span_input_sequence(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7]),
        slot_mappings=torch.tensor([[100, 101, 102, 103, 104, 105]]),
        spans=(
            PartialReuseComputeSpan(start_row=0, end_row=2),
            PartialReuseComputeSpan(start_row=4, end_row=6),
        ),
        initial_computed_tokens=2,
    )
    attention_inputs = tuple(
        build_partial_reuse_span_attention_inputs(
            item,
            block_tables=(torch.tensor([[7, 8, 9]]),),
        )
        for item in span_inputs
    )

    metadata = build_partial_reuse_span_attention_metadata(
        attention_inputs,
        attn_groups=[],
        kv_cache_config=SimpleNamespace(),
    )

    assert metadata == ({"span_tokens": 2}, {"span_tokens": 2})
    assert [call["max_seq_len"] for call in calls] == [4, 8]
    assert [call["seq_lens"].tolist() for call in calls] == [[4], [8]]


# Check that model inputs and attention metadata remain paired by span.
def test_build_partial_reuse_span_execution_steps() -> None:
    compacted = build_partial_reuse_compacted_batch(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7]),
        slot_mappings=torch.tensor([[100, 101, 102, 103, 104, 105]]),
        block_tables=(torch.tensor([[7, 8, 9]]),),
        query_start_locations=(0, 6),
        compute_rows=(0, 1, 4, 5),
        initial_computed_tokens=2,
    )
    metadata = ({"span": "first"}, {"span": "second"})

    steps = build_partial_reuse_span_execution_steps(compacted, metadata)

    assert all(isinstance(step, PartialReuseSpanExecutionStep) for step in steps)
    assert [(step.span.start_row, step.span.end_row) for step in steps] == [
        (0, 2),
        (4, 6),
    ]
    assert [step.model_inputs["input_ids"].tolist() for step in steps] == [
        [10, 11],
        [14, 15],
    ]
    assert [step.attention_metadata["span"] for step in steps] == [
        "first",
        "second",
    ]
    assert [step.slot_mappings.tolist() for step in steps] == [
        [[100, 101]],
        [[104, 105]],
    ]


# Check that execution planning rejects missing span metadata.
def test_build_partial_reuse_span_execution_steps_requires_matching_counts() -> None:
    compacted = build_partial_reuse_compacted_batch(
        input_ids=torch.tensor([10, 11, 12, 13]),
        positions=torch.tensor([2, 3, 4, 5]),
        slot_mappings=torch.tensor([[100, 101, 102, 103]]),
        block_tables=(torch.tensor([[7, 8, 9]]),),
        query_start_locations=(0, 4),
        compute_rows=(0, 1, 3),
        initial_computed_tokens=2,
    )

    with pytest.raises(
        ValueError,
        match="every compute span requires attention metadata",
    ):
        build_partial_reuse_span_execution_steps(compacted, ({"span": "first"},))


# Check that the executor calls each span once in causal order.
def test_execute_partial_reuse_span_steps() -> None:
    compacted = build_partial_reuse_compacted_batch(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7]),
        slot_mappings=torch.tensor([[100, 101, 102, 103, 104, 105]]),
        block_tables=(torch.tensor([[7, 8, 9]]),),
        query_start_locations=(0, 6),
        compute_rows=(0, 1, 4, 5),
        initial_computed_tokens=2,
    )
    steps = build_partial_reuse_span_execution_steps(
        compacted,
        ({"span": "first"}, {"span": "second"}),
    )
    called_spans = []

    # Mimic one model call while recording the order received by the executor.
    def fake_execute(step: PartialReuseSpanExecutionStep) -> torch.Tensor:
        called_spans.append((step.span.start_row, step.span.end_row))
        return step.model_inputs["input_ids"] * 2

    outputs = execute_partial_reuse_span_steps(steps, fake_execute)

    assert called_spans == [(0, 2), (4, 6)]
    assert [output.tolist() for output in outputs] == [
        [20, 22],
        [28, 30],
    ]


# Check that the executor rejects steps supplied in reverse causal order.
def test_execute_partial_reuse_span_steps_rejects_reordered_steps() -> None:
    compacted = build_partial_reuse_compacted_batch(
        input_ids=torch.tensor([10, 11, 12, 13]),
        positions=torch.tensor([2, 3, 4, 5]),
        slot_mappings=torch.tensor([[100, 101, 102, 103]]),
        block_tables=(torch.tensor([[7, 8, 9]]),),
        query_start_locations=(0, 4),
        compute_rows=(0, 1, 3),
        initial_computed_tokens=2,
    )
    steps = build_partial_reuse_span_execution_steps(
        compacted,
        ({"span": "first"}, {"span": "second"}),
    )

    # Return a step unchanged if ordering validation permits callback execution.
    def return_step(
        step: PartialReuseSpanExecutionStep,
    ) -> PartialReuseSpanExecutionStep:
        return step

    with pytest.raises(
        ValueError,
        match="execution steps must be ordered and non-overlapping",
    ):
        execute_partial_reuse_span_steps(tuple(reversed(steps)), return_step)


# Check that separated hidden states return to their original prompt rows.
def test_stitch_partial_reuse_span_outputs() -> None:
    compacted = build_partial_reuse_compacted_batch(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7]),
        slot_mappings=torch.tensor([[100, 101, 102, 103, 104, 105]]),
        block_tables=(torch.tensor([[7, 8, 9]]),),
        query_start_locations=(0, 6),
        compute_rows=(0, 1, 4, 5),
        initial_computed_tokens=2,
    )
    steps = build_partial_reuse_span_execution_steps(
        compacted,
        ({"span": "first"}, {"span": "second"}),
    )
    outputs = (
        torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
        torch.tensor([[5.0, 5.5], [6.0, 6.5]]),
    )

    stitched = stitch_partial_reuse_span_outputs(steps, outputs, total_rows=6)

    assert stitched.tolist() == [
        [1.0, 1.5],
        [2.0, 2.5],
        [0.0, 0.0],
        [0.0, 0.0],
        [5.0, 5.5],
        [6.0, 6.5],
    ]


# Check that a span cannot return a different number of hidden-state rows.
def test_stitch_partial_reuse_span_outputs_rejects_wrong_row_count() -> None:
    compacted = build_partial_reuse_compacted_batch(
        input_ids=torch.tensor([10, 11]),
        positions=torch.tensor([2, 3]),
        slot_mappings=torch.tensor([[100, 101]]),
        block_tables=(torch.tensor([[7, 8, 9]]),),
        query_start_locations=(0, 2),
        compute_rows=(0, 1),
        initial_computed_tokens=2,
    )
    steps = build_partial_reuse_span_execution_steps(
        compacted,
        ({"span": "only"},),
    )

    with pytest.raises(
        ValueError,
        match="span output rows must match the execution span",
    ):
        stitch_partial_reuse_span_outputs(
            steps,
            (torch.tensor([[1.0, 1.5]]),),
            total_rows=2,
        )


# Check that an invalid edit radius cannot configure the experimental selector.
def test_edit_proximity_repair_selector_rejects_negative_radius() -> None:
    with pytest.raises(ValueError, match="max_block_distance must be non-negative"):
        EditProximityRepairSelector(max_block_distance=-1)


# Check learned repair decisions and conservative missing-feature fallback.
def test_mlp_repair_selector() -> None:
    model = MLPRepairModel(
        {
            "schema_version": 1,
            "feature_names": ["signal"],
            "selected_repair_threshold": 0.5,
            "standardizer": {"mean": [0.0], "scale": [1.0]},
            "layers": [
                {
                    "weights": [[1.0]],
                    "bias": [0.0],
                    "activation": "logistic",
                }
            ],
        }
    )
    candidates = tuple(
        ResolvedPartialReuseCandidate(
            source_block_index=index,
            target_block_index=index,
            source_block_id=40 + index,
            target_block_id=50 + index,
            source_resident=True,
            selector_features=features,
        )
        for index, features in enumerate(((("signal", -2.0),), (("signal", 2.0),), ()))
    )

    repairs = MLPRepairSelector(model).select(candidates, block_size=4)

    assert {repair.target_block_index for repair in repairs} == {1, 2}


# Check that configuration names construct the expected repair policies.
def test_create_repair_selector() -> None:
    assert isinstance(
        create_repair_selector("full_block", edit_radius=3),
        FullBlockRepairSelector,
    )
    edit_selector = create_repair_selector("edit_proximity", edit_radius=3)
    assert isinstance(edit_selector, EditProximityRepairSelector)
    assert edit_selector.max_block_distance == 3


# Check that the selector factory rejects unknown policy names defensively.
def test_create_repair_selector_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown CacheSelect repair selector"):
        create_repair_selector("unknown", edit_radius=1)  # type: ignore[arg-type]


# Check that repair summaries distinguish selected from skipped repair tokens.
def test_summarize_repair_selection() -> None:
    candidate = ResolvedPartialReuseCandidate(
        source_block_index=3,
        target_block_index=5,
        source_block_id=42,
        target_block_id=63,
        source_resident=True,
        requires_repair=True,
    )
    instruction = PartialReuseRepairInstruction(
        source_block_id=42,
        target_block_id=63,
        target_block_index=5,
        target_token_indices=(20, 21),
    )

    metrics = summarize_repair_selection(
        "edit_proximity", (candidate,), (instruction,), block_size=4
    )

    assert metrics.selector == "edit_proximity"
    assert metrics.candidate_tokens == 4
    assert metrics.repair_tokens == 2
    assert metrics.skipped_repair_tokens == 2
    assert metrics.copied_blocks == 0
    assert metrics.copied_tokens == 0
    assert not metrics.execution_eligible
    assert metrics.execution_reason == "not_evaluated"
    assert metrics.reused_batch_rows == 0
    assert metrics.compute_batch_rows == 0
    assert metrics.compute_span_count == 0
    assert not metrics.compacted_batch_built
    assert not metrics.compacted_batch_executed
    assert not metrics.span_metadata_built
    assert metrics.span_metadata_count == 0

    executed_metrics = record_copy_execution(metrics, copied_blocks=1, block_size=4)
    assert executed_metrics.copied_blocks == 1
    assert executed_metrics.copied_tokens == 4

    eligible_metrics = record_batch_execution_decision(
        executed_metrics,
        eligible=True,
        reason="eligible",
        reused_batch_rows=2,
        compute_batch_rows=2,
    )
    assert eligible_metrics.execution_eligible
    assert eligible_metrics.execution_reason == "eligible"
    assert eligible_metrics.reused_batch_rows == 2
    assert eligible_metrics.compute_batch_rows == 2
    assert eligible_metrics.compute_span_count == 0
    assert not eligible_metrics.compacted_batch_built
    assert not eligible_metrics.compacted_batch_executed

    compacted_metrics = record_compacted_batch_construction(
        eligible_metrics,
        compacted_rows=2,
        compute_span_count=1,
    )
    assert compacted_metrics.compacted_batch_built
    assert compacted_metrics.compute_span_count == 1
    assert not compacted_metrics.compacted_batch_executed
    assert not compacted_metrics.span_metadata_built
    assert compacted_metrics.span_metadata_count == 0

    metadata_metrics = record_span_attention_metadata_construction(
        compacted_metrics,
        metadata_count=1,
    )
    assert metadata_metrics.span_metadata_built
    assert metadata_metrics.span_metadata_count == 1
    assert not metadata_metrics.compacted_batch_executed

    executed_batch_metrics = record_compacted_batch_execution(
        metadata_metrics,
        executed_span_count=1,
    )
    assert executed_batch_metrics.compacted_batch_executed

    timed_metrics = record_preparation_time(executed_batch_metrics, 1.25)
    timed_metrics = record_preparation_time(timed_metrics, 0.75)
    timed_metrics = record_gpu_execution_times(
        timed_metrics,
        copy_time_ms=0.5,
        forward_time_ms=4.5,
    )
    assert timed_metrics.preparation_time_ms == 2.0
    assert timed_metrics.copy_time_ms == 0.5
    assert timed_metrics.forward_time_ms == 4.5
