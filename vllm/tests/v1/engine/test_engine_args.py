# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError

import pytest

from vllm.config import CacheConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import _xxhash


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching, (
        "V1 turns on prefix caching by default."
    )

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not vllm_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to sha256_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256_cbor"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])


# Check CacheSelect repair-policy defaults and explicit CLI overrides.
def test_cacheselect_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    args = parser.parse_args([])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert not engine_args.enable_cacheselect
    assert engine_args.cacheselect_repair_selector == "full_block"
    assert engine_args.cacheselect_edit_radius == 1
    assert engine_args.cacheselect_mlp_model is None
    assert not engine_args.cacheselect_execute_partial_reuse
    assert engine_args.cacheselect_repack_partial_reuse
    assert engine_args.cacheselect_min_reuse_span_blocks == 1
    assert engine_args.cacheselect_min_repacked_reuse_span_blocks is None
    assert not engine_args.cacheselect_correct_kv_positions
    assert engine_args.gdn_delta_cache_capacity == 0
    assert engine_args.gdn_delta_block_size == 64
    assert engine_args.gdn_delta_execution_mode == "shadow"

    args = parser.parse_args(["--no-cacheselect-repack-partial-reuse"])
    assert not EngineArgs.from_cli_args(args=args).cacheselect_repack_partial_reuse

    args = parser.parse_args(
        [
            "--cacheselect-repair-selector",
            "mlp",
            "--cacheselect-mlp-model",
            "model.json",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.cacheselect_repair_selector == "mlp"
    assert engine_args.cacheselect_mlp_model == "model.json"

    args = parser.parse_args(
        [
            "--enable-cacheselect",
            "--cacheselect-repair-selector",
            "edit_proximity",
            "--cacheselect-edit-radius",
            "2",
            "--cacheselect-execute-partial-reuse",
            "--cacheselect-repack-partial-reuse",
            "--cacheselect-min-reuse-span-blocks",
            "8",
            "--cacheselect-min-repacked-reuse-span-blocks",
            "16",
            "--cacheselect-correct-kv-positions",
            "--gdn-delta-cache-capacity",
            "8",
            "--gdn-delta-block-size",
            "128",
            "--gdn-delta-execution-mode",
            "active",
            "--mamba-cache-mode",
            "all",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_cacheselect
    assert engine_args.cacheselect_repair_selector == "edit_proximity"
    assert engine_args.cacheselect_edit_radius == 2
    assert engine_args.cacheselect_execute_partial_reuse
    assert engine_args.cacheselect_repack_partial_reuse
    assert engine_args.cacheselect_min_reuse_span_blocks == 8
    assert engine_args.cacheselect_min_repacked_reuse_span_blocks == 16
    assert engine_args.cacheselect_correct_kv_positions
    assert engine_args.gdn_delta_cache_capacity == 8
    assert engine_args.gdn_delta_block_size == 128
    assert engine_args.gdn_delta_execution_mode == "active"
    cache_config = engine_args.create_engine_config().cache_config
    assert cache_config.cacheselect_execute_partial_reuse
    assert cache_config.cacheselect_repack_partial_reuse
    assert cache_config.cacheselect_min_reuse_span_blocks == 8
    assert cache_config.cacheselect_min_repacked_reuse_span_blocks == 16
    assert cache_config.cacheselect_correct_kv_positions
    assert cache_config.gdn_delta_cache_capacity == 8
    assert cache_config.gdn_delta_block_size == 128
    assert cache_config.prefix_match_unit is None
    assert cache_config.gdn_delta_execution_mode == "active"

    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        parser.parse_args(["--cacheselect-repair-selector", "invalid_selector"])


def test_cacheselect_position_correction_requires_execution():
    with pytest.raises(ValueError, match="position correction requires execution"):
        CacheConfig(cacheselect_correct_kv_positions=True)


@pytest.mark.skipif(_xxhash is None, reason="xxhash not installed")
def test_prefix_caching_xxhash_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # set hash algorithm to xxhash (pickle)
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to xxhash_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash_cbor"


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config: VllmConfig = engine_args.create_engine_config(UsageContext.LLM_CLASS)

    from vllm.platforms import current_platform
    from vllm.utils.mem_constants import GiB_bytes

    device_memory = current_platform.get_device_total_memory()
    device_name = current_platform.get_device_name().lower()
    if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
        # For GPUs like H100, H200, and MI300x with >= 70GB memory
        default_llm_tokens = 16384
        default_server_tokens = 8192
        default_max_num_seqs = 1024
    else:
        default_llm_tokens = 8192
        default_server_tokens = 2048
        default_max_num_seqs = 256

    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_llm_tokens  # noqa: E501

    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_server_tokens  # noqa: E501


def test_mm_prefix_lm_raises_batched_tokens_floor():
    """Verify that prefix-LM multimodal models auto-raise
    max_num_batched_tokens to fit at least one multimodal item.

    Regression test for https://github.com/vllm-project/vllm/issues/42687
    """
    from unittest.mock import patch

    # Simulate a prefix-LM multimodal model whose largest modality
    # (video) requires 2496 tokens — more than the 2048 default.
    fake_mm_min = (2496, "video")

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_model_len=2048,
        enforce_eager=True,
    )

    with (
        patch.object(
            type(engine_args),
            "_get_min_mm_batched_tokens",
            staticmethod(lambda _mc: fake_mm_min),
        ),
        patch(
            "vllm.config.ModelConfig.is_multimodal_model",
            new_callable=lambda: property(lambda self: True),
        ),
        patch(
            "vllm.config.ModelConfig.is_mm_prefix_lm",
            new_callable=lambda: property(lambda self: True),
        ),
    ):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.scheduler_config.max_num_batched_tokens >= 2496
