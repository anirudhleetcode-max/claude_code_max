"""Python logging configuration with secret redaction."""

from __future__ import annotations

import logging
from pathlib import Path

from ..security.secrets import Redactor


class RedactingFilter(logging.Filter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        record.msg = self.redactor.redact_text(message)
        record.args = ()
        return True


def configure_logging(level: str = "WARNING", log_file: Path | None = None, redactor: Redactor | None = None, debug: bool = False) -> None:
    root = logging.getLogger("ai_engineer")
    root.setLevel(logging.DEBUG if debug else getattr(logging, level.upper(), logging.WARNING))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    redact = RedactingFilter(redactor or Redactor())
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else getattr(logging, level.upper(), logging.WARNING))
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    console.addFilter(redact)
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        fh.addFilter(redact)
        root.addHandler(fh)
    root.propagate = False
