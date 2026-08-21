from unittest import TestCase

from benchmarks.run_full_attention_model_validation import (
    EXPECTED_OUTPUT,
    assess_full_attention_configuration,
    assess_uncached_stability,
)


# Build the minimal effective server configuration expected from the A40 run.
def _server_info():
    return {
        "vllm_config": {
            "model_config": {
                "dtype": "torch.bfloat16",
                "quantization": None,
            },
            "cache_config": {
                "block_size": 16,
                "enable_prefix_caching": True,
                "enable_cacheselect": False,
                "cacheselect_execute_partial_reuse": False,
            },
        }
    }


# Describe the conventional 40-layer Qwen3-14B architecture.
def _hf_config():
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 40,
    }


class FullAttentionConfigurationTests(TestCase):
    # Accept only the intended unquantized BF16 full-attention configuration.
    def test_accepts_qwen3_14b_bfloat16(self):
        assessment = assess_full_attention_configuration(
            _server_info(), _hf_config()
        )

        self.assertTrue(assessment["passed"])
        self.assertTrue(all(assessment["checks"].values()))

    # Catch accidentally selecting a hybrid checkpoint before GPU experiments.
    def test_rejects_hybrid_layer_types(self):
        hf_config = _hf_config()
        hf_config["layer_types"] = ["linear_attention", "full_attention"] * 20

        assessment = assess_full_attention_configuration(_server_info(), hf_config)

        self.assertFalse(assessment["passed"])
        self.assertFalse(assessment["checks"]["no_hybrid_layer_types"])

    # Catch an accidental quantized or non-BF16 server launch.
    def test_rejects_wrong_runtime_precision(self):
        server_info = _server_info()
        server_info["vllm_config"]["model_config"].update(
            {"dtype": "torch.float16", "quantization": "awq"}
        )

        assessment = assess_full_attention_configuration(server_info, _hf_config())

        self.assertFalse(assessment["checks"]["bfloat16_runtime"])
        self.assertFalse(assessment["checks"]["unquantized_runtime"])


class UncachedStabilityTests(TestCase):
    # Accept two uncached, complete, byte-identical expected answers.
    def test_accepts_stable_full_computation(self):
        rows = [
            {
                "cached_tokens": 0,
                "finish_reason": "stop",
                "output_text": EXPECTED_OUTPUT,
            }
            for _ in range(2)
        ]

        self.assertTrue(assess_uncached_stability(rows)["passed"])

    # Matching output is not evidence of full computation when APC was used.
    def test_rejects_cached_repetition(self):
        rows = [
            {
                "cached_tokens": cached,
                "finish_reason": "stop",
                "output_text": EXPECTED_OUTPUT,
            }
            for cached in (0, 16)
        ]

        assessment = assess_uncached_stability(rows)

        self.assertFalse(assessment["passed"])
        self.assertFalse(assessment["checks"]["both_uncached"])

    # Different valid-looking text still fails the determinism gate.
    def test_rejects_output_drift(self):
        rows = [
            {
                "cached_tokens": 0,
                "finish_reason": "stop",
                "output_text": output,
            }
            for output in (EXPECTED_OUTPUT, EXPECTED_OUTPUT + " Extra")
        ]

        assessment = assess_uncached_stability(rows)

        self.assertFalse(assessment["checks"]["exactly_stable"])
        self.assertFalse(assessment["checks"]["instruction_followed"])
