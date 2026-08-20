import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT / "vllm" / "vllm" / "model_executor" / "models" / "qwen3_next.py"
)
PROFILER_PATH = PROJECT_ROOT / "vllm" / "vllm" / "profiler" / "wrapper.py"


class HybridProfileScopeTests(unittest.TestCase):
    # Keep the profiler labels stable so GPU traces remain comparable across runs.
    def test_qwen_hybrid_layers_have_named_opt_in_scopes(self) -> None:
        source = MODEL_PATH.read_text(encoding="utf-8")

        ast.parse(source)
        self.assertIn("record_function_or_nullcontext", source)
        for label in (
            "cacheselect_hybrid:gdn_attention",
            "cacheselect_hybrid:full_attention",
            "cacheselect_hybrid:mlp",
        ):
            with self.subTest(label=label):
                self.assertEqual(source.count(label), 1)

    # Keep a compact structured artifact alongside the much larger Torch trace.
    def test_profiler_writes_hybrid_scope_summary(self) -> None:
        source = PROFILER_PATH.read_text(encoding="utf-8")

        ast.parse(source)
        self.assertIn("cacheselect_scope_summary_", source)
        self.assertIn('event.key.startswith("cacheselect_hybrid:")', source)
        self.assertIn('"device_time_total_us"', source)


if __name__ == "__main__":
    unittest.main()
