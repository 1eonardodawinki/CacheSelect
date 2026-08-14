from unittest import TestCase

from benchmarks.analyze_cacheselect_performance import (
    _exclude_invalid_repetitions,
    _pair_trials,
    _summarize,
)


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

    # Verify an invalid full-compute workload is labelled without aborting the run.
    def test_reference_quality_failure_is_reported(self):
        rows = [
            _trial("native", "The access code is UNKNOWN.", False),
            _trial("shadow", "The access code is UNKNOWN.", False),
            _trial("active", "The access code is UNKNOWN.", False),
        ]

        pair = _pair_trials(rows)[0]
        summary = _summarize([pair])[0]

        self.assertFalse(pair["reference_quality_passed"])
        self.assertEqual(summary["reference_quality_pass_rate"], 0.0)
        self.assertEqual(summary["active_quality_pass_rate"], 0.0)

    # Verify one invalid mode removes only its corresponding paired repetition.
    def test_invalid_timing_repetition_is_excluded(self):
        rows = [
            _trial("native", "answer", True),
            _trial("shadow", "answer", True),
            _trial("active", "answer", True),
        ]
        invalid = [
            {
                "target_tokens": 1024,
                "position": "middle",
                "mode": "active",
                "repetition": 1,
            }
        ]

        self.assertEqual(_exclude_invalid_repetitions(rows, invalid), [])

    # Verify summaries expose discarded measurements instead of hiding them.
    def test_summary_reports_invalid_repetition_count(self):
        rows = [
            _trial("native", "answer", True),
            _trial("shadow", "answer", True),
            _trial("active", "answer", True),
        ]

        summary = _summarize(_pair_trials(rows), attempted_repetitions=3)[0]

        self.assertEqual(summary["valid_repetitions"], 1)
        self.assertEqual(summary["invalid_repetitions"], 2)
        self.assertAlmostEqual(summary["valid_measurement_rate"], 1 / 3)

    # Verify a wholly invalid position remains visible with unavailable metrics.
    def test_summary_preserves_position_without_valid_repetitions(self):
        rows = [
            _trial("native", "answer", True),
            _trial("shadow", "answer", True),
            _trial("active", "answer", True),
        ]

        summaries = _summarize(
            _pair_trials(rows), attempted_repetitions=3, target_tokens=1024
        )
        early = next(row for row in summaries if row["position"] == "early")

        self.assertEqual(len(summaries), 3)
        self.assertEqual(early["valid_repetitions"], 0)
        self.assertEqual(early["invalid_repetitions"], 3)
        self.assertIsNone(early["mean_active_vs_native_ttft_ms"])
