# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


@dataclass(frozen=True)
class GDNCheckpointWritePlan:
    """Map cached block boundaries to states emitted by the GDN chunk kernel."""

    destination_block_indices: tuple[int, ...]
    source_chunk_indices: tuple[int, ...]


# Map completed GDN blocks to the chunk-start states representing their ends.
def plan_gdn_checkpoint_writes(
    *,
    first_chunk_index: int,
    num_computed_tokens: int,
    first_scheduled_block: int,
    last_scheduled_block: int,
    block_size: int,
    chunk_size: int,
) -> GDNCheckpointWritePlan:
    """Plan intermediate checkpoint copies; the final block is saved separately."""
    if block_size <= 0 or chunk_size <= 0:
        raise ValueError("block_size and chunk_size must be positive")
    if block_size % chunk_size != 0:
        raise ValueError("GDN checkpoint blocks must align to whole kernel chunks")
    if min(first_chunk_index, num_computed_tokens, first_scheduled_block) < 0:
        raise ValueError("checkpoint positions must be non-negative")
    if last_scheduled_block < first_scheduled_block:
        raise ValueError("last_scheduled_block must not precede the first block")

    destination_blocks = tuple(range(first_scheduled_block, last_scheduled_block))
    if not destination_blocks:
        return GDNCheckpointWritePlan((), ())

    # The chunk kernel's h[i] is the state immediately before chunk i. The
    # state after a complete cache block is therefore the first chunk state
    # belonging to the following block.
    first_boundary_token = (first_scheduled_block + 1) * block_size
    tokens_until_first_boundary = first_boundary_token - num_computed_tokens
    if tokens_until_first_boundary < 0:
        raise ValueError("the first checkpoint boundary precedes computed tokens")
    if tokens_until_first_boundary % chunk_size != 0:
        raise ValueError("computed tokens do not align with a checkpoint chunk")

    first_source_chunk = first_chunk_index + tokens_until_first_boundary // chunk_size
    chunks_per_block = block_size // chunk_size
    source_chunks = tuple(
        first_source_chunk + offset * chunks_per_block
        for offset in range(len(destination_blocks))
    )
    return GDNCheckpointWritePlan(destination_blocks, source_chunks)


# Copy intermediate and final recurrent states into their physical cache slots.
def write_gdn_prefill_checkpoints(
    *,
    state_cache: torch.Tensor,
    chunk_states: torch.Tensor,
    final_states: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    first_scheduled_blocks: torch.Tensor,
    last_scheduled_blocks: torch.Tensor,
    block_size: int,
    chunk_size: int,
) -> None:
    """Persist each completed block boundary and the final prefill state."""
    num_sequences = final_states.shape[0]
    if checkpoint_state_indices.shape[0] != num_sequences:
        raise ValueError("checkpoint table and final states disagree on batch size")
    if chunk_offsets.numel() != num_sequences + 1:
        raise ValueError("chunk_offsets must contain one start and one final offset")
    for values in (
        num_computed_tokens,
        first_scheduled_blocks,
        last_scheduled_blocks,
    ):
        if values.numel() != num_sequences:
            raise ValueError("checkpoint geometry must contain one value per sequence")

    for sequence_index in range(num_sequences):
        first_chunk_index = int(chunk_offsets[sequence_index].item())
        first_block = int(first_scheduled_blocks[sequence_index].item())
        last_block = int(last_scheduled_blocks[sequence_index].item())
        plan = plan_gdn_checkpoint_writes(
            first_chunk_index=first_chunk_index,
            num_computed_tokens=int(num_computed_tokens[sequence_index].item()),
            first_scheduled_block=first_block,
            last_scheduled_block=last_block,
            block_size=block_size,
            chunk_size=chunk_size,
        )

        if plan.destination_block_indices:
            destination_blocks = torch.tensor(
                plan.destination_block_indices,
                dtype=torch.long,
                device=checkpoint_state_indices.device,
            )
            source_chunks = torch.tensor(
                plan.source_chunk_indices,
                dtype=torch.long,
                device=chunk_states.device,
            )
            physical_slots = checkpoint_state_indices[
                sequence_index, destination_blocks
            ].long()
            if torch.any(physical_slots == NULL_BLOCK_ID).item():
                raise ValueError("an intermediate checkpoint has no physical slot")
            state_cache[physical_slots] = chunk_states[0, source_chunks].to(
                state_cache.dtype
            )

        final_slot = int(checkpoint_state_indices[sequence_index, last_block].item())
        if final_slot == NULL_BLOCK_ID:
            raise ValueError("the final checkpoint has no physical slot")
        state_cache[final_slot] = final_states[sequence_index].to(state_cache.dtype)


# Persist the recurrent state produced by one actively reused complete block.
def write_gdn_reused_block_checkpoint(
    *,
    state_cache: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    sequence_index: int,
    target_block_index: int,
    final_state: torch.Tensor,
) -> None:
    """Write one reused block's outgoing state to its physical cache slot."""
    if checkpoint_state_indices.ndim != 2:
        raise ValueError("checkpoint_state_indices must be a two-dimensional table")
    if sequence_index < 0 or sequence_index >= checkpoint_state_indices.shape[0]:
        raise ValueError("sequence_index is outside the checkpoint table")
    if (
        target_block_index < 0
        or target_block_index >= checkpoint_state_indices.shape[1]
    ):
        raise ValueError("target_block_index is outside the checkpoint table")
    expected_shape = (1, *state_cache.shape[1:])
    if final_state.shape != expected_shape:
        raise ValueError("final_state has an incompatible recurrent-state shape")

    physical_slot = int(
        checkpoint_state_indices[sequence_index, target_block_index].item()
    )
    if physical_slot == NULL_BLOCK_ID:
        raise ValueError("the reused block checkpoint has no physical slot")
    state_cache[physical_slot] = final_state[0].to(state_cache.dtype)


# Copy decode inputs to their destination block and return destination slots.
def prepare_gdn_decode_checkpoints(
    *,
    state_cache: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    input_block_indices: torch.Tensor,
    output_block_indices: torch.Tensor,
) -> torch.Tensor:
    """Prepare in-place decode kernels when a token crosses a cache boundary."""
    if checkpoint_state_indices.ndim != 2:
        raise ValueError("checkpoint_state_indices must be a two-dimensional table")
    batch_size = checkpoint_state_indices.shape[0]
    if input_block_indices.shape != (batch_size,):
        raise ValueError("input block indices must contain one value per request")
    if output_block_indices.shape != (batch_size,):
        raise ValueError("output block indices must contain one value per request")
    input_slots = checkpoint_state_indices.gather(
        1,
        input_block_indices.long().unsqueeze(1),
    ).squeeze(1)
    output_slots = checkpoint_state_indices.gather(
        1,
        output_block_indices.long().unsqueeze(1),
    ).squeeze(1)
    # Keep the hot per-token path asynchronous while still failing on an
    # unallocated slot when the device reaches this assertion.
    torch._assert_async(
        torch.all(input_slots != NULL_BLOCK_ID),
        "decode input checkpoint has no physical slot",
    )
    torch._assert_async(
        torch.all(output_slots != NULL_BLOCK_ID),
        "decode output checkpoint has no physical slot",
    )

    # Copying same-slot rows is harmless; crossing rows seed the new block with
    # the exact state that precedes the token being decoded.
    state_cache[output_slots.long()] = state_cache[input_slots.long()]
    return output_slots.long()


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

    # All-mode checkpoint metadata is kept separate from the existing
    # single-state fields until the GDN kernels consume the full block table.
    checkpoint_state_indices: torch.Tensor | None = None
    block_idx_last_computed_token: torch.Tensor | None = None
    block_idx_first_scheduled_token: torch.Tensor | None = None
    block_idx_last_scheduled_token: torch.Tensor | None = None
    num_computed_tokens: torch.Tensor | None = None
    num_computed_tokens_cpu: torch.Tensor | None = None
    prefill_query_start_loc_cpu: torch.Tensor | None = None
    contextual_block_hashes: tuple[tuple[bytes, ...], ...] = ()
    gdn_delta_reuse_candidates: tuple[tuple[tuple[int, bytes], ...], ...] = ()

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        if (
            self.vllm_config.cache_config.mamba_cache_mode == "all"
            and self.use_spec_decode
        ):
            # Checkpoint input/output routing for draft-token branches is a
            # separate correctness problem and must not silently use block 0.
            raise NotImplementedError(
                "GDN checkpointing does not yet support speculative decoding"
            )
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    # Locate the recurrent-state checkpoints read and written by one GDN step.
    def _compute_prefix_caching_block_indices(
        self,
        num_computed_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size = self.kv_cache_spec.block_size

        # The initial state is stored at the end of the last computed block.
        block_idx_last_computed_token = (
            torch.div(
                num_computed_tokens + block_size - 1,
                block_size,
                rounding_mode="floor",
            )
            - 1
        )
        # An unaligned continuation must overwrite its partially filled block.
        block_idx_first_scheduled_token = (
            torch.div(
                num_computed_tokens + block_size,
                block_size,
                rounding_mode="floor",
            )
            - 1
        )
        block_idx_last_scheduled_token = (
            torch.div(
                seq_lens + block_size - 1,
                block_size,
                rounding_mode="floor",
            )
            - 1
        )
        block_idx_last_computed_token.clamp_(min=0)
        block_idx_last_scheduled_token.clamp_(min=0)
        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        checkpoint_state_indices: torch.Tensor | None = None
        block_idx_last_computed_token: torch.Tensor | None = None
        block_idx_first_scheduled_token: torch.Tensor | None = None
        block_idx_last_scheduled_token: torch.Tensor | None = None
        num_computed_tokens: torch.Tensor | None = None
        num_computed_tokens_cpu: torch.Tensor | None = None
        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            # Preserve the complete table; the legacy fields below deliberately
            # keep selecting one state until checkpoint execution is implemented.
            checkpoint_state_indices = block_table_tensor
            num_computed_tokens = context_lens_tensor
            if m.seq_lens_cpu_upper_bound is not None:
                query_lens_cpu = m.query_start_loc_cpu[1:] - m.query_start_loc_cpu[:-1]
                num_computed_tokens_cpu = (
                    m.seq_lens_cpu_upper_bound[: m.num_reqs] - query_lens_cpu
                )
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                num_computed_tokens,
                m.seq_lens,
            )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_query_start_loc_cpu: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            from vllm.third_party.flash_linear_attention.ops.utils import (
                FLA_CHUNK_SIZE,
            )

            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            if self.gdn_prefill_backend == "cutedsl":
                from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                    prepare_metadata_cutedsl,
                )

                assert prefill_query_start_loc is not None
                assert prefill_query_start_loc_cpu is not None
                total_tokens = int(prefill_query_start_loc_cpu[-1].item())
                chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
                    prefill_query_start_loc,
                    total_tokens,
                    FLA_CHUNK_SIZE,
                )
            else:
                gpu_device = query_start_loc.device
                # Only prefill batches use FLA chunk ops.
                # Pre-compute on CPU and async-copy to GPU to avoid
                # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
                from vllm.third_party.flash_linear_attention.ops.index import (
                    prepare_chunk_indices,
                    prepare_chunk_offsets,
                )

                assert prefill_query_start_loc_cpu is not None
                chunk_indices = async_tensor_h2d(
                    prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                    device=gpu_device,
                )
                chunk_offsets = async_tensor_h2d(
                    prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                    device=gpu_device,
                )

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            checkpoint_state_indices=checkpoint_state_indices,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            num_computed_tokens=num_computed_tokens,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
            contextual_block_hashes=m.contextual_block_hashes,
            gdn_delta_reuse_candidates=m.gdn_delta_reuse_candidates,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
