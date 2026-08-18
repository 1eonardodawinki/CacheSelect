# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interface declarations for Qwen3.5 recurrent-state prefix caching."""

from vllm.model_executor.models.interfaces import supports_mamba_prefix_caching
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM,
)


# Verify every registered Qwen3.5 text wrapper advertises checkpoint support.
def test_qwen3_5_models_support_mamba_prefix_caching():
    """Dense, MoE, and multimodal wrappers should select Mamba all mode."""
    assert supports_mamba_prefix_caching(Qwen3_5ForCausalLM)
    assert supports_mamba_prefix_caching(Qwen3_5MoeForCausalLM)
    assert supports_mamba_prefix_caching(Qwen3_5ForConditionalGeneration)
