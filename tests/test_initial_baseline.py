import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from benchmarks.run_vllm_baseline import (
    _observe_request,
    _shadow_plan_requests,
    _source_metadata_by_request,
    _validate_observability,
)
from benchmarks.evaluation import score_response
from benchmarks.generate_length_calibration import _rendered_token_count
from benchmarks.analyze_length_calibration import _selected_results
from benchmarks.length_calibration import (
    EDIT_POSITIONS,
    build_length_calibration_trace,
)
from benchmarks.schema import load_trace, save_trace
from benchmarks.workloads import (
    build_chat_trace,
    build_periodic_agent_trace,
    build_rag_trace,
)
from cacheselect.features import token_transition_features
from cacheselect.planner import NativeAPCFallbackPlanner
from observability.request_recorder import RequestRecorder, validate_ledger


class WorkloadTests(TestCase):
    def test_all_traces_have_valid_adjacent_transitions(self):
        traces = [
            build_rag_trace(),
            build_periodic_agent_trace(),
            build_chat_trace(),
        ]
        for trace in traces:
            request_ids = [request.request_id for request in trace.requests]
            self.assertEqual(len(trace.transitions), len(trace.requests) - 1)
            for index, transition in enumerate(trace.transitions, start=1):
                self.assertEqual(
                    transition.previous_request_id,
                    request_ids[index - 1],
                )
                self.assertEqual(
                    transition.current_request_id,
                    request_ids[index],
                )

    def test_trace_round_trip_preserves_segments(self):
        trace = build_rag_trace()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            save_trace(trace, path)
            loaded = load_trace(path)
        self.assertEqual(loaded, trace)

    def test_chat_contains_append_and_history_edit_controls(self):
        change_types = [
            transition.ground_truth.change_type
            for transition in build_chat_trace().transitions
        ]
        self.assertIn("append_turn", change_types)
        self.assertIn("edit_history", change_types)

    def test_periodic_trace_is_self_contained_and_has_exact_answers(self):
        trace = build_periodic_agent_trace(count=3, window_size=6)
        self.assertEqual(len(trace.requests), 3)
        self.assertEqual(
            trace.requests[0].ground_truth.expected_answer,
            (
                '{"mean_cpu":44.5,"max_cpu":47,'
                '"max_timestamp":"2026-07-30T09:20:00Z","alert":false}'
            ),
        )
        self.assertEqual(
            trace.requests[2].ground_truth.expected_answer,
            (
                '{"mean_cpu":52.7,"max_cpu":87,'
                '"max_timestamp":"2026-07-30T09:35:00Z","alert":true}'
            ),
        )

    def test_ground_truth_is_not_part_of_model_payload(self):
        request = build_rag_trace().requests[0]
        payload = request.api_payload("test-model", 8)
        serialised = json.dumps(payload)
        self.assertNotIn("expected_answer", serialised)
        self.assertNotIn("ground_truth", serialised)

    def test_source_metadata_names_only_the_previous_request(self):
        trace = build_rag_trace()

        metadata = _source_metadata_by_request(trace)

        self.assertEqual(
            metadata[trace.requests[0].request_id],
            {"cacheselect_request_id": trace.requests[0].request_id},
        )
        self.assertEqual(
            metadata[trace.requests[1].request_id],
            {
                "cacheselect_request_id": trace.requests[1].request_id,
                "cacheselect_source_request_id": trace.requests[0].request_id,
                "cacheselect_transition_id": trace.transitions[0].transition_id,
            },
        )

    def test_length_calibration_controls_target_and_edit_position(self):
        def count_words(messages):
            return 5 + sum(len(message["content"].split()) for message in messages)

        for edit_position in EDIT_POSITIONS:
            trace = build_length_calibration_trace(
                target_prompt_tokens=256,
                edit_position=edit_position,
                token_counter=count_words,
                tokenizer_name="word-counter-test",
            )
            counts = [count_words(request.messages) for request in trace.requests]
            self.assertTrue(all(abs(count - 256) <= 16 for count in counts))
            self.assertEqual(len(trace.requests), 2)
            self.assertEqual(
                trace.transitions[0].ground_truth.changed_segment_ids,
                [f"{edit_position}_marker"],
            )
            base_segments = {
                segment.segment_id: segment for segment in trace.requests[0].segments
            }
            edited_segments = {
                segment.segment_id: segment for segment in trace.requests[1].segments
            }
            self.assertEqual(base_segments[f"{edit_position}_marker"].version, 1)
            self.assertEqual(edited_segments[f"{edit_position}_marker"].version, 2)

    def test_calibration_token_counter_accepts_transformers_return_shapes(self):
        class FakeTokenizer:
            def __init__(self, encoded):
                self.encoded = encoded

            def apply_chat_template(self, *args, **kwargs):
                return self.encoded

        messages = [{"role": "user", "content": "test"}]
        self.assertEqual(
            _rendered_token_count(FakeTokenizer([1, 2, 3]), messages),
            3,
        )
        self.assertEqual(
            _rendered_token_count(
                FakeTokenizer({"input_ids": [1, 2, 3, 4]}),
                messages,
            ),
            4,
        )

    def test_calibration_quality_checks_fact_not_citation_format(self):
        def count_words(messages):
            return 5 + sum(len(message["content"].split()) for message in messages)

        trace = build_length_calibration_trace(
            target_prompt_tokens=256,
            edit_position="early",
            token_counter=count_words,
            tokenizer_name="word-counter-test",
        )
        score = score_response(
            "The verified project code is NORTH-731.",
            trace.requests[0].ground_truth,
        )
        self.assertTrue(score["passed"])
        self.assertEqual(score["requirements_total"], 1)

    def test_later_calibration_root_supersedes_duplicate_condition(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            (first / "results").mkdir(parents=True)
            (second / "results").mkdir(parents=True)
            for target in (256, 1024, 4096):
                for position in ("early", "middle", "late"):
                    for apc in ("off", "on"):
                        name = f"tokens-{target}-{position}-apc-{apc}-rep-1.json"
                        (first / "results" / name).write_text("{}")
            replacement = second / "results" / "tokens-4096-early-apc-off-rep-1.json"
            replacement.write_text("{}")

            selected, superseded, raw_count = _selected_results([first, second])

        self.assertEqual(raw_count, 19)
        self.assertEqual(len(selected), 18)
        self.assertEqual(selected[(4096, "early", "off")][1], replacement)
        self.assertEqual(len(superseded), 1)


class EvaluationTests(TestCase):
    def test_all_required_facts_must_be_present(self):
        ground_truth = build_rag_trace().requests[0].ground_truth
        passed = score_response(
            "The answer is 88% [doc_battery_2024].",
            ground_truth,
        )
        failed = score_response("The answer is 88%.", ground_truth)
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])


class FeatureTests(TestCase):
    def test_exact_prefix_is_detected(self):
        features = token_transition_features([1, 2, 3], [1, 2, 3, 4])
        self.assertTrue(features["previous_is_exact_prefix"])
        self.assertEqual(features["common_prefix_tokens"], 3)

    def test_middle_edit_retains_suffix_signal(self):
        features = token_transition_features(
            [1, 2, 3, 4, 5],
            [1, 2, 9, 4, 5],
        )
        self.assertFalse(features["previous_is_exact_prefix"])
        self.assertEqual(features["first_changed_token"], 2)
        self.assertEqual(features["common_suffix_tokens"], 2)


class BaselineRunnerTests(TestCase):
    def test_shadow_planner_compares_adjacent_rendered_prompts(self):
        class SequentialTokenizer:
            def __init__(self):
                self.outputs = iter(([1, 2], [1, 2, 3], [1, 9, 3]))

            def apply_chat_template(self, *args, **kwargs):
                return next(self.outputs)

        requests = build_chat_trace().requests[:3]
        planned = _shadow_plan_requests(
            requests,
            tokenizer=SequentialTokenizer(),
            planner=NativeAPCFallbackPlanner(),
        )

        self.assertEqual(
            planned[requests[0].request_id][1].policy.value,
            "FULL_RECOMPUTE",
        )
        self.assertEqual(
            planned[requests[1].request_id][1].policy.value,
            "VLLM_NATIVE_APC",
        )
        self.assertEqual(
            planned[requests[2].request_id][1].policy.value,
            "VLLM_NATIVE_APC",
        )

    def test_observation_keeps_request_rendering_tokens_and_metrics(self):
        response = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "test output"}}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
            "prompt_text": "rendered prompt",
            "prompt_token_ids": [1, 2, 3],
            "metrics": {"time_to_first_token_ms": 4.5},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(response).encode()

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            request = build_chat_trace().requests[0]
            observation = _observe_request(
                request,
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                vllm_xargs={
                    "cacheselect_request_id": "chat-target",
                    "cacheselect_source_request_id": "chat-source",
                    "cacheselect_transition_id": "chat-transition",
                },
                cache_salt="isolated-repetition-1",
            )

        sent_request = urlopen.call_args.args[0]
        sent_payload = json.loads(sent_request.data)
        self.assertTrue(sent_payload["return_token_ids"])
        self.assertTrue(sent_payload["return_prompt_text"])
        self.assertEqual(sent_payload["cache_salt"], "isolated-repetition-1")
        self.assertEqual(
            sent_payload["vllm_xargs"],
            {
                "cacheselect_request_id": "chat-target",
                "cacheselect_source_request_id": "chat-source",
                "cacheselect_transition_id": "chat-transition",
            },
        )
        self.assertEqual(observation["rendered_prompt"], "rendered prompt")
        self.assertEqual(observation["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(observation["cached_tokens"], 2)
        self.assertEqual(
            observation["server_metrics"]["time_to_first_token_ms"],
            4.5,
        )

    def test_missing_server_observability_fails_clearly(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "enable-prompt-tokens-details",
        ):
            _validate_observability({"usage": {}})

    def test_vllm_planner_metrics_are_recorded_as_runtime_policy(self):
        response = {
            "choices": [{"message": {"role": "assistant", "content": "output"}}],
            "usage": {
                "prompt_tokens": 80,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "prompt_text": "rendered prompt",
            "prompt_token_ids": [1, 2, 3],
            "metrics": {
                "time_to_first_token_ms": 4.5,
                "cacheselect_policy": "FULL_RECOMPUTE",
                "cacheselect_reason": "no_native_prefix",
                "cacheselect_native_cached_tokens": 0,
                "cacheselect_partial_reuse_plan": {
                    "source_request_id": "rag-source",
                    "candidate_token_count": 48,
                },
            },
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(response).encode()

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            observation = _observe_request(
                build_rag_trace().requests[0],
                url="http://vllm.test/v1/chat/completions",
                model="test-model",
                max_completion_tokens=8,
                api_key=None,
                timeout_seconds=2.0,
                require_cacheselect_metrics=True,
            )

        self.assertEqual(
            observation["runtime_policy"],
            {
                "policy": "FULL_RECOMPUTE",
                "reason": "no_native_prefix",
                "native_cached_tokens": 0,
                "partial_reuse_plan": {
                    "source_request_id": "rag-source",
                    "candidate_token_count": 48,
                },
            },
        )

    def test_missing_vllm_planner_metrics_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "CacheSelect metrics"):
            _validate_observability(
                {
                    "usage": {"prompt_tokens_details": {}},
                    "prompt_text": "prompt",
                    "prompt_token_ids": [1],
                    "metrics": {"time_to_first_token_ms": 1.0},
                },
                require_cacheselect_metrics=True,
            )

    def test_planner_token_mismatch_fails_the_recorded_request(self):
        response = {
            "choices": [{"message": {"role": "assistant", "content": "output"}}],
            "usage": {
                "prompt_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "prompt_text": "rendered prompt",
            "prompt_token_ids": [1, 2, 3],
            "metrics": {"time_to_first_token_ms": 4.5},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(response).encode()

        with TemporaryDirectory() as directory:
            recorder = RequestRecorder(
                run_id="planner-token-mismatch",
                model="test-model",
                backend="vllm-test",
                log_dir=directory,
            )
            with patch(
                "urllib.request.urlopen",
                return_value=FakeResponse(),
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    _observe_request(
                        build_rag_trace().requests[0],
                        url="http://vllm.test/v1/chat/completions",
                        model="test-model",
                        max_completion_tokens=8,
                        api_key=None,
                        timeout_seconds=2.0,
                        recorder=recorder,
                        expected_prompt_token_ids=[1, 2, 9],
                    )
            summary = validate_ledger(recorder.path)

        self.assertTrue(summary.is_complete)
        self.assertEqual(summary.failed, 1)

    def test_observation_is_written_to_full_request_ledger(self):
        response = {
            "choices": [{"message": {"role": "assistant", "content": "full output"}}],
            "usage": {
                "prompt_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "prompt_text": "rendered prompt",
            "prompt_token_ids": [1, 2, 3],
            "metrics": {"time_to_first_token_ms": 4.5},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(response).encode()

        with TemporaryDirectory() as directory:
            recorder = RequestRecorder(
                run_id="baseline-ledger-test",
                model="test-model",
                backend="vllm-test",
                log_dir=directory,
            )
            request = build_rag_trace().requests[0]
            with patch(
                "urllib.request.urlopen",
                return_value=FakeResponse(),
            ):
                _observe_request(
                    request,
                    url="http://vllm.test/v1/chat/completions",
                    model="test-model",
                    max_completion_tokens=8,
                    api_key="must-not-be-recorded",
                    timeout_seconds=2.0,
                    recorder=recorder,
                    policy_metadata={"policy": "VLLM_NATIVE_APC"},
                )

            events = [
                json.loads(line)
                for line in Path(recorder.path).read_text().splitlines()
            ]
            summary = validate_ledger(recorder.path)

        self.assertTrue(summary.is_complete)
        self.assertEqual(
            events[0]["model_input"]["payload"]["messages"], request.messages
        )
        self.assertEqual(events[1]["output"]["text"], "full output")
        self.assertEqual(
            events[0]["evaluation"]["prompt_segments"][0]["segment_id"],
            request.segments[0].segment_id,
        )
        self.assertNotIn("must-not-be-recorded", json.dumps(events))

    def test_every_request_in_the_three_workloads_is_recorded(self):
        response = {
            "choices": [{"message": {"role": "assistant", "content": "test output"}}],
            "usage": {
                "prompt_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "prompt_text": "rendered prompt",
            "prompt_token_ids": [1, 2, 3],
            "metrics": {"time_to_first_token_ms": 4.5},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(response).encode()

        traces = [
            build_rag_trace(),
            build_chat_trace(),
            build_periodic_agent_trace(),
        ]
        expected_requests = sum(len(trace.requests) for trace in traces)

        with TemporaryDirectory() as directory:
            recorder = RequestRecorder(
                run_id="complete-suite-test",
                model="test-model",
                backend="vllm-test",
                log_dir=directory,
            )
            with patch(
                "urllib.request.urlopen",
                return_value=FakeResponse(),
            ):
                for trace in traces:
                    for request in trace.requests:
                        _observe_request(
                            request,
                            url="http://vllm.test/v1/chat/completions",
                            model="test-model",
                            max_completion_tokens=8,
                            api_key=None,
                            timeout_seconds=2.0,
                            recorder=recorder,
                        )

            events = [
                json.loads(line)
                for line in Path(recorder.path).read_text().splitlines()
            ]
            summary = validate_ledger(recorder.path)

        self.assertEqual(expected_requests, 14)
        self.assertEqual(summary.started, expected_requests)
        self.assertEqual(summary.completed, expected_requests)
        self.assertEqual(summary.failed, 0)
        self.assertTrue(summary.is_complete)
        self.assertEqual(len(events), expected_requests * 2)
        self.assertTrue(
            all(
                event["model_input"]["payload"]["messages"]
                for event in events
                if event["event"] == "request_started"
            )
        )
        self.assertTrue(
            all(
                "raw_response" in event["output"]
                for event in events
                if event["event"] == "request_completed"
            )
        )
