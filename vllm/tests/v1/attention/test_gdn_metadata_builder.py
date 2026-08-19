# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
    checkpoint_block_index_for_token_end,
    plan_gdn_checkpoint_writes,
    prepare_gdn_decode_checkpoints,
    write_gdn_prefill_checkpoints,
    write_gdn_reused_block_checkpoint,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


# Map fine logical spans into the larger physical checkpoint page table.
def test_maps_logical_span_end_to_physical_checkpoint() -> None:
    assert checkpoint_block_index_for_token_end(64, 2048) == 0
    assert checkpoint_block_index_for_token_end(2048, 2048) == 0
    assert checkpoint_block_index_for_token_end(2112, 2048) == 1

    with pytest.raises(ValueError, match="end_token must be positive"):
        checkpoint_block_index_for_token_end(0, 2048)


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    mamba_cache_mode: str = "none",
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


# Verify that all mode retains every physical checkpoint address and its geometry.
def test_all_mode_exposes_block_checkpoint_metadata():
    """GDN all mode should describe the recurrent states a prefill reads and writes."""
    builder = _create_gdn_builder(mamba_cache_mode="all")
    batch = BatchSpec(seq_lens=[80, 50], query_lens=[48, 30])
    common = create_common_attn_metadata(
        batch,
        BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    )
    common.contextual_block_hashes = ((b"a", b"b", b"c"), (b"d", b"e"))

    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert meta.checkpoint_state_indices is not None
    assert torch.equal(meta.checkpoint_state_indices, common.block_table_tensor)
    assert meta.num_computed_tokens is not None
    assert meta.num_computed_tokens.tolist() == [32, 20]
    assert meta.num_computed_tokens_cpu is not None
    assert meta.num_computed_tokens_cpu.tolist() == [32, 20]
    assert meta.prefill_query_start_loc_cpu is not None
    assert meta.prefill_query_start_loc_cpu.tolist() == [0, 48, 78]
    assert meta.contextual_block_hashes == common.contextual_block_hashes
    assert meta.block_idx_last_computed_token is not None
    assert meta.block_idx_last_computed_token.tolist() == [1, 1]
    assert meta.block_idx_first_scheduled_token is not None
    assert meta.block_idx_first_scheduled_token.tolist() == [2, 1]
    assert meta.block_idx_last_scheduled_token is not None
    assert meta.block_idx_last_scheduled_token.tolist() == [4, 3]


# Verify that align mode remains on the existing single-state path.
def test_align_mode_does_not_build_block_checkpoint_metadata():
    """GDN align mode should not allocate or expose historical checkpoints."""
    builder = _create_gdn_builder(mamba_cache_mode="align")
    meta = _build(builder, BatchSpec(seq_lens=[80], query_lens=[48]))

    assert meta.checkpoint_state_indices is None
    assert meta.block_idx_last_computed_token is None
    assert meta.block_idx_first_scheduled_token is None
    assert meta.block_idx_last_scheduled_token is None
    assert meta.num_computed_tokens is None


# Reject speculative decoding until checkpoint routing covers every draft state.
def test_all_mode_rejects_speculative_decoding():
    """All mode must fail closed rather than reuse block zero for draft tokens."""
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        _create_gdn_builder(
            num_speculative_tokens=2,
            mamba_cache_mode="all",
        )


# Verify the chunk-state mapping for a fresh multi-block prefill.
def test_plans_gdn_checkpoint_writes_from_prompt_start():
    """Every complete non-final block should map to its following chunk state."""
    plan = plan_gdn_checkpoint_writes(
        first_chunk_index=0,
        num_computed_tokens=0,
        first_scheduled_block=0,
        last_scheduled_block=4,
        block_size=64,
        chunk_size=16,
    )

    assert plan.destination_block_indices == (0, 1, 2, 3)
    assert plan.source_chunk_indices == (4, 8, 12, 16)


# Verify that a restored prefix and preceding batch rows shift source chunks.
def test_plans_gdn_checkpoint_writes_after_cached_prefix():
    """Chunk indices should be relative to both the restored prefix and batch row."""
    plan = plan_gdn_checkpoint_writes(
        first_chunk_index=10,
        num_computed_tokens=128,
        first_scheduled_block=2,
        last_scheduled_block=4,
        block_size=64,
        chunk_size=16,
    )

    assert plan.destination_block_indices == (2, 3)
    assert plan.source_chunk_indices == (14, 18)


# Verify that the final-state-only case needs no intermediate copies.
def test_plans_no_intermediate_write_for_one_partial_block():
    """A single partial block is written from the kernel's final state later."""
    plan = plan_gdn_checkpoint_writes(
        first_chunk_index=0,
        num_computed_tokens=128,
        first_scheduled_block=2,
        last_scheduled_block=2,
        block_size=64,
        chunk_size=16,
    )

    assert plan.destination_block_indices == ()
    assert plan.source_chunk_indices == ()


# Reject a schedule whose first cache boundary falls inside a kernel chunk.
def test_rejects_unaligned_gdn_checkpoint_write():
    """A checkpoint must never claim a state that the kernel did not emit."""
    with pytest.raises(ValueError, match="do not align"):
        plan_gdn_checkpoint_writes(
            first_chunk_index=0,
            num_computed_tokens=130,
            first_scheduled_block=2,
            last_scheduled_block=4,
            block_size=64,
            chunk_size=16,
        )


# Verify that logical block checkpoints land in their physical cache slots.
def test_writes_gdn_prefill_checkpoints():
    """Intermediate chunks and the final state should populate mapped slots."""
    state_cache = torch.zeros((12, 1), dtype=torch.float32)
    chunk_states = torch.arange(20, dtype=torch.float32).view(1, 20, 1)
    final_states = torch.tensor([[99.0]])
    checkpoint_table = torch.tensor([[7, 3, 10, 1, 8]], dtype=torch.int32)

    write_gdn_prefill_checkpoints(
        state_cache=state_cache,
        chunk_states=chunk_states,
        final_states=final_states,
        checkpoint_state_indices=checkpoint_table,
        chunk_offsets=torch.tensor([0, 20], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0], dtype=torch.int32),
        first_scheduled_blocks=torch.tensor([0], dtype=torch.int32),
        last_scheduled_blocks=torch.tensor([4], dtype=torch.int32),
        block_size=64,
        chunk_size=16,
    )

    assert state_cache[7].item() == 4
    assert state_cache[3].item() == 8
    assert state_cache[10].item() == 12
    assert state_cache[1].item() == 16
    assert state_cache[8].item() == 99


# Reject a missing destination before a negative block ID can index the cache.
def test_rejects_missing_gdn_checkpoint_slot():
    """Null physical slots must fail closed instead of writing the wrong state."""
    with pytest.raises(ValueError, match="no physical slot"):
        write_gdn_prefill_checkpoints(
            state_cache=torch.zeros((4, 1)),
            chunk_states=torch.zeros((1, 4, 1)),
            final_states=torch.ones((1, 1)),
            checkpoint_state_indices=torch.tensor([[0]], dtype=torch.int32),
            chunk_offsets=torch.tensor([0, 4], dtype=torch.int32),
            num_computed_tokens=torch.tensor([0], dtype=torch.int32),
            first_scheduled_blocks=torch.tensor([0], dtype=torch.int32),
            last_scheduled_blocks=torch.tensor([0], dtype=torch.int32),
            block_size=64,
            chunk_size=16,
        )


# Verify that one reused logical block saves its outgoing state physically.
def test_writes_gdn_reused_block_checkpoint():
    """Active reuse should populate the target block's mapped cache slot."""
    state_cache = torch.zeros((6, 2), dtype=torch.float16)
    checkpoint_table = torch.tensor([[4, 1, 5]], dtype=torch.int32)

    write_gdn_reused_block_checkpoint(
        state_cache=state_cache,
        checkpoint_state_indices=checkpoint_table,
        sequence_index=0,
        target_block_index=1,
        final_state=torch.tensor([[7.0, 8.0]], dtype=torch.float32),
    )

    torch.testing.assert_close(
        state_cache[1],
        torch.tensor([7.0, 8.0], dtype=torch.float16),
    )


# Verify that decode copies state only according to logical-to-physical mapping.
def test_prepares_gdn_decode_checkpoint_destination():
    """A boundary-crossing token should start from the preceding block state."""
    state_cache = torch.arange(12, dtype=torch.float32).view(12, 1)
    checkpoint_table = torch.tensor(
        [[7, 3, 10], [2, 8, 5]],
        dtype=torch.int32,
    )

    output_slots = prepare_gdn_decode_checkpoints(
        state_cache=state_cache,
        checkpoint_state_indices=checkpoint_table,
        input_block_indices=torch.tensor([1, 0], dtype=torch.int32),
        output_block_indices=torch.tensor([2, 0], dtype=torch.int32),
    )

    assert output_slots.tolist() == [10, 2]
    assert state_cache[10].item() == 3
    assert state_cache[2].item() == 2


# Reject a null state before it can be treated as Python's final tensor row.
def test_rejects_missing_gdn_decode_checkpoint():
    """Decode must fail closed when either checkpoint block is not resident."""
    with pytest.raises(RuntimeError, match="no physical slot"):
        prepare_gdn_decode_checkpoints(
            state_cache=torch.zeros((4, 1)),
            checkpoint_state_indices=torch.tensor([[1, 0]], dtype=torch.int32),
            input_block_indices=torch.tensor([0], dtype=torch.int32),
            output_block_indices=torch.tensor([1], dtype=torch.int32),
        )
