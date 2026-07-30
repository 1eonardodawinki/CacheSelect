"""Append-only recording of every conceptual LLM generation request.

A request is one externally meaningful generation attempt. Internal model
forwards used to build a KV cache, process a suffix, or decode one token are
part of that request and are not logged as separate requests.

The recorder writes a ``request_started`` event before inference and a matching
``request_completed`` or ``request_failed`` event afterward. This two-event
format preserves the input even if inference crashes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - CacheSelect targets Linux/macOS.
    fcntl = None


SCHEMA_VERSION = 1
_THREAD_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned or "run"


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return repr(value)


class RequestRecorder:
    """Writes correlated request lifecycle events to an append-only JSONL file."""

    def __init__(
        self,
        *,
        run_id: str,
        model: str,
        backend: str,
        log_dir: str | os.PathLike[str] | None = None,
        invocation_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.model = model
        self.backend = backend
        self.invocation_id = str(uuid.uuid4())
        self.invocation_metadata = invocation_metadata or {}

        if log_dir is None:
            configured = os.environ.get("CACHESELECT_REQUEST_LOG_DIR")
            if configured:
                log_root = Path(configured)
            else:
                log_root = Path(__file__).resolve().parents[1] / "request_logs"
        else:
            log_root = Path(log_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        self.path = log_root / f"{_safe_name(run_id)}.jsonl"

    def start(
        self,
        *,
        model_input: dict[str, Any],
        sampling: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
    ) -> PendingRequest:
        """Persist a complete request input before model execution starts."""

        request_id = str(uuid.uuid4())
        started_monotonic = time.perf_counter()
        self._append(
            {
                "schema_version": SCHEMA_VERSION,
                "event": "request_started",
                "timestamp_utc": _utc_now(),
                "run_id": self.run_id,
                "invocation_id": self.invocation_id,
                "request_id": request_id,
                "model": self.model,
                "backend": self.backend,
                "model_input": model_input,
                "sampling": sampling or {},
                "metadata": metadata or {},
                # Kept outside model_input so benchmark answers cannot be
                # mistaken for information visible to the model or planner.
                "evaluation": evaluation or {},
                "invocation_metadata": self.invocation_metadata,
            }
        )
        return PendingRequest(
            recorder=self,
            request_id=request_id,
            started_monotonic=started_monotonic,
        )

    def _append(self, event: dict[str, Any]) -> None:
        encoded = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        with _THREAD_LOCK:
            with self.path.open("a", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclasses.dataclass
class PendingRequest:
    """Handle used to complete or fail a previously persisted request."""

    recorder: RequestRecorder
    request_id: str
    started_monotonic: float
    _finished: bool = False

    def complete(
        self,
        *,
        output: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._finished:
            raise RuntimeError(f"Request {self.request_id} was already finished")
        self._finished = True
        self.recorder._append(
            {
                "schema_version": SCHEMA_VERSION,
                "event": "request_completed",
                "timestamp_utc": _utc_now(),
                "run_id": self.recorder.run_id,
                "invocation_id": self.recorder.invocation_id,
                "request_id": self.request_id,
                "duration_seconds": time.perf_counter() - self.started_monotonic,
                "output": output,
                "metrics": metrics or {},
                "metadata": metadata or {},
            }
        )

    def fail(
        self,
        error: BaseException,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._finished:
            raise RuntimeError(f"Request {self.request_id} was already finished")
        self._finished = True
        self.recorder._append(
            {
                "schema_version": SCHEMA_VERSION,
                "event": "request_failed",
                "timestamp_utc": _utc_now(),
                "run_id": self.recorder.run_id,
                "invocation_id": self.recorder.invocation_id,
                "request_id": self.request_id,
                "duration_seconds": time.perf_counter() - self.started_monotonic,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    ),
                },
                "metadata": metadata or {},
            }
        )


@dataclasses.dataclass(frozen=True)
class LedgerSummary:
    """Completeness summary for one append-only request ledger."""

    path: Path
    started: int
    completed: int
    failed: int
    incomplete_request_ids: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.incomplete_request_ids


def validate_ledger(path: str | os.PathLike[str]) -> LedgerSummary:
    """Validate pairing and uniqueness of request lifecycle events.

    Failed requests count as terminal and therefore complete records. A
    ``request_started`` event without a matching completion/failure is returned
    as incomplete, which is expected after a killed process but must not pass a
    final experiment audit.
    """

    ledger_path = Path(path)
    started: dict[str, int] = {}
    terminal: dict[str, tuple[str, int]] = {}

    with ledger_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{ledger_path}:{line_number} is not valid JSON"
                ) from error
            request_id = event.get("request_id")
            event_name = event.get("event")
            if not request_id:
                raise ValueError(
                    f"{ledger_path}:{line_number} has no request_id"
                )
            if event_name == "request_started":
                if request_id in started:
                    raise ValueError(
                        f"Duplicate request_started for {request_id} at "
                        f"{ledger_path}:{line_number}"
                    )
                started[request_id] = line_number
            elif event_name in {"request_completed", "request_failed"}:
                if request_id in terminal:
                    raise ValueError(
                        f"Duplicate terminal event for {request_id} at "
                        f"{ledger_path}:{line_number}"
                    )
                terminal[request_id] = (event_name, line_number)
            else:
                raise ValueError(
                    f"{ledger_path}:{line_number} has unknown event "
                    f"{event_name!r}"
                )

    orphaned = sorted(set(terminal) - set(started))
    if orphaned:
        raise ValueError(
            "Terminal events without request_started: "
            + ", ".join(orphaned)
        )

    incomplete = tuple(sorted(set(started) - set(terminal)))
    return LedgerSummary(
        path=ledger_path,
        started=len(started),
        completed=sum(
            event_name == "request_completed"
            for event_name, _ in terminal.values()
        ),
        failed=sum(
            event_name == "request_failed"
            for event_name, _ in terminal.values()
        ),
        incomplete_request_ids=incomplete,
    )
