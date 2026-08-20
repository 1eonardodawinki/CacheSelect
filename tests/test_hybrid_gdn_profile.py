import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.hybrid_gdn_profile import capture_profiled_request


class HybridGDNProfileTests(unittest.TestCase):
    # Stop profiling after the request and identify only its newly written trace.
    def test_captures_one_isolated_request_trace(self) -> None:
        actions = []
        with tempfile.TemporaryDirectory() as directory:
            profile_dir = Path(directory)
            old_trace = profile_dir / "old.pt.trace.json.gz"
            old_trace.write_bytes(b"old")

            # Mimic vLLM writing its trace while handling the stop endpoint.
            def fake_control(**kwargs) -> None:
                actions.append(kwargs["action"])
                if kwargs["action"] == "stop":
                    (profile_dir / "new.pt.trace.json.gz").write_bytes(b"new")

            with patch(
                "benchmarks.hybrid_gdn_profile._post_profile_control",
                side_effect=fake_control,
            ):
                result, trace = capture_profiled_request(
                    base_url="http://127.0.0.1:8000",
                    profile_dir=profile_dir,
                    run_request=lambda: {"output": "complete"},
                    timeout_seconds=10.0,
                )

        self.assertEqual(actions, ["start", "stop"])
        self.assertEqual(result, {"output": "complete"})
        self.assertEqual(trace.name, "new.pt.trace.json.gz")

    # A request error must not leave the expensive profiler running.
    def test_stops_profiler_when_request_fails(self) -> None:
        actions = []

        # Raise from the model request after profiling has started.
        def fail_request() -> None:
            raise RuntimeError("request failed")

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "benchmarks.hybrid_gdn_profile._post_profile_control",
                side_effect=lambda **kwargs: actions.append(kwargs["action"]),
            ):
                with self.assertRaisesRegex(RuntimeError, "request failed"):
                    capture_profiled_request(
                        base_url="http://127.0.0.1:8000",
                        profile_dir=Path(directory),
                        run_request=fail_request,
                        timeout_seconds=10.0,
                    )

        self.assertEqual(actions, ["start", "stop"])


if __name__ == "__main__":
    unittest.main()
