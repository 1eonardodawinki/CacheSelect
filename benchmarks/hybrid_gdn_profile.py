"""Capture one request inside an isolated vLLM Torch-profiler window."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar


ResultT = TypeVar("ResultT")


# Record trace metadata so both new files and overwritten files are detected.
def _trace_snapshot(profile_dir: Path) -> dict[Path, tuple[int, int]]:
    """Return the size and modification time of every Torch trace."""
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in profile_dir.rglob("*.pt.trace.json*")
    }


# Call the local development-only endpoint that controls worker profiling.
def _post_profile_control(
    *,
    base_url: str,
    action: str,
    timeout_seconds: float,
) -> None:
    """Start or stop the profiler and require a successful HTTP response."""
    if action not in {"start", "stop"}:
        raise ValueError("profile action must be start or stop")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{action}_profile",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM {action}_profile returned {response.status}")


# Isolate one model request and return the trace created when profiling stops.
def capture_profiled_request(
    *,
    base_url: str,
    profile_dir: Path,
    run_request: Callable[[], ResultT],
    timeout_seconds: float,
) -> tuple[ResultT, Path]:
    """Run one request in a profile window and return its result and trace."""
    before = _trace_snapshot(profile_dir)
    _post_profile_control(
        base_url=base_url,
        action="start",
        timeout_seconds=timeout_seconds,
    )
    try:
        result = run_request()
    finally:
        _post_profile_control(
            base_url=base_url,
            action="stop",
            timeout_seconds=timeout_seconds,
        )

    after = _trace_snapshot(profile_dir)
    changed = tuple(
        path for path, metadata in after.items() if before.get(path) != metadata
    )
    if len(changed) != 1:
        raise RuntimeError(
            "profile window must produce exactly one worker trace; "
            f"observed {len(changed)}"
        )
    return result, changed[0]
