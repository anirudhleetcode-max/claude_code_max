"""Append-only, redacted audit log (JSON lines)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..core.util import utcnow_iso
from .secrets import Redactor


class AuditLog:
    def __init__(self, path: Path | None, redactor: Redactor | None = None, max_bytes: int = 10_000_000) -> None:
        self.path = path
        self.redactor = redactor or Redactor()
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, **fields: Any) -> None:
        if self.path is None:
            return
        entry = {"ts": utcnow_iso(), "action": action, **self.redactor.redact(fields)}
        line = json.dumps(entry, default=str, ensure_ascii=False)
        with self._lock:
            try:
                if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                    rotated = self.path.with_suffix(self.path.suffix + ".1")
                    self.path.replace(rotated)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                # Auditing must never crash the agent; failures surface via logging.
                import logging

                logging.getLogger(__name__).warning("could not write audit log %s", self.path)

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
