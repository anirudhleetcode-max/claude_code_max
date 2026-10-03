"""Data models for validation commands and their parsed results."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, Field


class CheckKind(StrEnum):
    TEST = "test"
    LINT = "lint"
    FORMAT = "format"
    TYPECHECK = "typecheck"
    BUILD = "build"
    AUDIT = "audit"


CheckStatus = Literal["passed", "failed", "error", "timeout", "skipped", "unavailable", "cancelled"]


class ValidationCommand(BaseModel):
    """A concrete command that validates one aspect of the project."""

    kind: CheckKind
    command: str
    source: str
    cwd: str = "."
    timeout_s: float = 600.0
    confidence: float = 1.0
    scope: Literal["full", "targeted"] = "full"
    available: bool = True
    unavailable_reason: str = ""


class TestCaseFailure(BaseModel):
    """One failing (or erroring) test case."""

    __test__: ClassVar[bool] = False  # not a pytest test class

    test_id: str
    file: str | None = None
    line: int | None = None
    message: str = ""
    failure_type: str = "unknown"  # assertion|error|import|syntax|timeout|type|unknown


class Diagnostic(BaseModel):
    """A compiler / linter / type checker finding."""

    file: str | None = None
    line: int | None = None
    column: int | None = None
    code: str | None = None
    message: str
    severity: Literal["error", "warning"] = "error"


# --- signature normalisation --------------------------------------------------------------

_WIN_ABS = re.compile(r"(?<![\w])[A-Za-z]:[\\/](?:[^\\/\s:'\"()<>\[\]]+[\\/])*")
_POSIX_ABS = re.compile(r"(?<![\w.~-])/(?:[^/\s:'\"()<>\[\]]+/)+")
_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_DURATION = re.compile(r"\(?\b\d+(?:\.\d+)?\s?(?:ms|s|sec|secs|seconds|m|min)\b\)?")
_NUMBER = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
_ERRORISH = re.compile(r"(?i)error|fail|exception|not found|no module|cannot|undefined|traceback|panic")


def normalize_text(text: str) -> str:
    """Strip volatile parts (paths, addresses, durations, numbers) from a message."""
    text = _WIN_ABS.sub("", text)
    text = _POSIX_ABS.sub("", text)
    text = _HEX.sub("<addr>", text)
    text = _DURATION.sub("", text)
    text = _NUMBER.sub("#", text)
    return _SPACES.sub(" ", text).strip()


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


class CheckResult(BaseModel):
    """The parsed outcome of running one validation command."""

    kind: CheckKind
    command: str
    status: CheckStatus
    exit_code: int | None = None
    duration_s: float = 0.0
    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    skipped: int | None = None
    failures: list[TestCaseFailure] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    # passed | test_failure | collection_error | import_error | missing_dependency | command_not_found |
    # syntax_error | type_errors | lint_errors | build_error | timeout | no_tests | environment |
    # vulnerabilities | unknown
    classification: str = ""
    summary: str = ""
    output_tail: str = ""

    def ok(self) -> bool:
        return self.status == "passed"

    def signature(self) -> str:
        """Stable fingerprint of *what* failed; identical failures give identical signatures."""
        if self.ok():
            return ""
        parts: set[str] = set()
        for f in self.failures:
            parts.add(f"T|{f.test_id}|{normalize_text(_first_line(f.message))[:200]}")
        diags = [d for d in self.diagnostics if d.severity == "error"] or self.diagnostics
        for d in diags:
            parts.add(f"D|{normalize_text(d.file or '')}|{d.code or ''}|{normalize_text(_first_line(d.message))[:200]}")
        if not parts:
            lines = [ln for ln in self.output_tail.splitlines() if ln.strip()]
            interesting = [ln for ln in lines if _ERRORISH.search(ln)][-5:] or lines[-3:]
            parts.update(f"O|{normalize_text(ln)[:200]}" for ln in interesting)
        material = "\n".join([str(self.kind), self.status, self.classification, *sorted(parts)])
        digest = hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"{self.kind}:{self.classification or self.status}:{digest}"
