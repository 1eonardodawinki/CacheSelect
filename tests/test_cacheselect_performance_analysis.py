from unittest import TestCase

from benchmarks.analyze_cacheselect_performance import _pair_trials, _summarize


# Build the smallest complete analyzer row for one synthetic serving mode.
def _trial(mode: str, output_text: str, quality_passed: bool) -> dict:
    executed = mode == "active"
    return {
        "target_tokens": 1024,
        "position": "middle",
        "mode": mode,
        "repetition": 1,
        "prompt_tokens": 1020,
        "native_cached_tokens": 0,
        "candidate_tokens": 640 if mode != "native" else 0,
        "reused_rows": 639 if mode != "native" else 0,
        "compute_rows": 381 if mode != "native" else 0,
        "span_count": 2 if mode != "native" else 0,
        "execution_eligible": executed,
        "execution_reason": "eligible" if executed else None,
        "executed": executed,
        "copy_time_ms": 1.0 if executed else 0.0,
        "preparation_time_ms": 2.0 if executed else 0.0,
        "forward_time_ms": 80.0 if executed else 120.0,
        "ttft_ms": 100.0 if executed else 140.0,
        "output_text": output_text,
        "quality_passed": quality_passed,
        "result_file": f"synthetic-{mode}.json",
    }


class CacheSelectPerformanceAnalysisTests(TestCase):
    # Verify an active-reuse error becomes a measured result, not discarded data.
    def test_active_quality_loss_is_reported(self):
        rows = [
            _trial("native", "The access code is NORTH-731.", True),
            _trial("shadow", "The access code is NORTH-731.", True),
            _trial("active", "The access code is SOUTH-913.", False),
        ]

        pair = _pair_trials(rows)[0]
        summary = _summarize([pair])[0]

        self.assertFalse(pair["active_quality_passed"])
        self.assertFalse(pair["active_matches_shadow"])
        self.assertLess(pair["active_shadow_word_similarity"], 1.0)
        self.assertEqual(summary["active_quality_pass_rate"], 0.0)
        self.assertEqual(summary["active_shadow_exact_match_rate"], 0.0)
