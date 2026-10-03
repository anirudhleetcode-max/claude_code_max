"""JSONL trace writer: every event, redacted (the bus applies redaction), one line each."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import IO

from ..core.events import Event, EventBus, EventType

_VERBOSE_ONLY = {EventType.MODEL_CALLED}


class TraceWriter:
    def __init__(self, path: Path, debug: bool = False) -> None:
        self.path = path
        self.debug = debug
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        if not self.debug and event.type in _VERBOSE_ONLY:
            return
        line = event.model_dump_json()
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def read_trace(path: Path, task_id: str | None = None) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if task_id is None or data.get("task_id") == task_id:
            events.append(data)
    return events
