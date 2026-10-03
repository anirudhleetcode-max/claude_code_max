"""Typed events and an in-process event bus.

Every significant action in the agent emits an :class:`Event`. Subscribers
(trace writer, SQLite event log, metrics, CLI renderer, web SSE stream) receive
events after secret redaction has been applied.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .ids import new_id

log = logging.getLogger(__name__)


class EventType(StrEnum):
    SESSION_STARTED = "SESSION_STARTED"
    TASK_CREATED = "TASK_CREATED"
    TASK_STARTED = "TASK_STARTED"
    TASK_RESUMED = "TASK_RESUMED"
    STAGE_STARTED = "STAGE_STARTED"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    UNDERSTANDING_CREATED = "UNDERSTANDING_CREATED"
    QUESTION_ASKED = "QUESTION_ASKED"
    PLAN_CREATED = "PLAN_CREATED"
    SUBTASK_STARTED = "SUBTASK_STARTED"
    SUBTASK_COMPLETED = "SUBTASK_COMPLETED"
    MODEL_CALLED = "MODEL_CALLED"
    MODEL_RESULT = "MODEL_RESULT"
    MODEL_RETRY = "MODEL_RETRY"
    MODEL_FALLBACK = "MODEL_FALLBACK"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_DENIED = "TOOL_DENIED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVAL_RESOLVED = "APPROVAL_RESOLVED"
    FILE_CHANGED = "FILE_CHANGED"
    TEST_STARTED = "TEST_STARTED"
    TEST_PASSED = "TEST_PASSED"
    TEST_FAILED = "TEST_FAILED"
    VALIDATION_RESULT = "VALIDATION_RESULT"
    FAILURE_RECORDED = "FAILURE_RECORDED"
    FIX_ATTEMPTED = "FIX_ATTEMPTED"
    LOOP_DETECTED = "LOOP_DETECTED"
    REVIEW_STARTED = "REVIEW_STARTED"
    REVIEW_COMPLETED = "REVIEW_COMPLETED"
    GATES_EVALUATED = "GATES_EVALUATED"
    CHECKPOINT_CREATED = "CHECKPOINT_CREATED"
    CHECKPOINT_RESTORED = "CHECKPOINT_RESTORED"
    COMMIT_CREATED = "COMMIT_CREATED"
    CONTEXT_COMPACTED = "CONTEXT_COMPACTED"
    MEMORY_UPDATED = "MEMORY_UPDATED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_BLOCKED = "TASK_BLOCKED"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_INTERRUPTED = "TASK_INTERRUPTED"
    REPORT_CREATED = "REPORT_CREATED"
    WARNING = "WARNING"
    ERROR = "ERROR"
    INFO = "INFO"


class Event(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
    type: EventType
    message: str = ""
    session_id: str = ""
    task_id: str = ""
    subtask_id: str = ""
    stage: str = ""
    level: str = "info"
    data: dict[str, Any] = Field(default_factory=dict)


Handler = Callable[[Event], None]
Redactor = Callable[[Any], Any]


class EventBus:
    """Synchronous fan-out with optional asyncio queue subscribers.

    Handlers run inline and must be cheap. A failing handler is logged and
    never interrupts the publisher.
    """

    def __init__(self, redactor: Redactor | None = None) -> None:
        self._handlers: list[tuple[Handler, Callable[[Event], bool] | None]] = []
        self._lock = threading.Lock()
        self._redactor = redactor
        self.context: dict[str, str] = {}

    def set_redactor(self, redactor: Redactor | None) -> None:
        self._redactor = redactor

    def subscribe(self, handler: Handler, where: Callable[[Event], bool] | None = None) -> Callable[[], None]:
        entry = (handler, where)
        with self._lock:
            self._handlers.append(entry)

        def unsubscribe() -> None:
            with self._lock, contextlib.suppress(ValueError):
                self._handlers.remove(entry)

        return unsubscribe

    def subscribe_queue(self, maxsize: int = 10000) -> tuple[asyncio.Queue[Event], Callable[[], None]]:
        """Return an asyncio queue receiving events (for async consumers such as SSE)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)

        def push(event: Event) -> None:
            def _put() -> None:
                if queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                queue.put_nowait(event)

            try:
                if asyncio.get_running_loop() is loop:
                    _put()
                    return
            except RuntimeError:
                pass
            loop.call_soon_threadsafe(_put)

        return queue, self.subscribe(push)

    def emit(self, type: EventType, message: str = "", **fields: Any) -> Event:
        data = fields.pop("data", None) or {}
        base = {k: v for k, v in self.context.items() if k in ("session_id", "task_id", "subtask_id", "stage")}
        base.update({k: v for k, v in fields.items() if k in ("session_id", "task_id", "subtask_id", "stage", "level")})
        extra = {k: v for k, v in fields.items() if k not in base and k != "level"}
        data = {**extra, **data}
        if self._redactor is not None:
            message = self._redactor(message)
            data = self._redactor(data)
        event = Event(type=type, message=message, data=data, **base)
        self.publish(event)
        return event

    def publish(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._handlers)
        for handler, where in handlers:
            try:
                if where is None or where(event):
                    handler(event)
            except Exception:
                log.exception("event handler failed for %s", event.type)
