"""Structured failure analysis and fix-loop detection.

For every failed validation the agent records:
ERROR · CONTEXT · LIKELY CAUSES · EVIDENCE · HYPOTHESIS · CHANGE · RESULT.
Attempted fixes are fingerprinted so the same fix is never applied twice, and
repeated or oscillating failure signatures stop the repair loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..core.ids import new_id
from ..core.util import sha1_text, utcnow_iso

LIKELY_CAUSES: dict[str, list[str]] = {
    "test_failure": [
        "the implementation does not satisfy what the failing test asserts",
        "an edge case (empty input, None, boundary values) is not handled",
        "a recent change broke behaviour that other code relies on",
    ],
    "collection_error": ["a test module fails to import (syntax or import error in code it loads)"],
    "import_error": ["wrong import path or a renamed/moved symbol", "a module was not created or is not on the import path"],
    "missing_dependency": ["a third-party package is not installed in the environment", "the dependency is missing from the manifest"],
    "syntax_error": ["a recent edit introduced invalid syntax"],
    "type_errors": ["types of changed code do not match their usage", "a signature changed without updating callers"],
    "lint_errors": ["style or static-analysis rule violations in changed code"],
    "build_error": ["code does not compile: undefined names, wrong types or missing files"],
    "timeout": ["an infinite loop or blocking call", "a test waits for network, input or a resource that never arrives"],
    "no_tests": ["no tests match the discovery pattern", "tests were not created where the runner looks for them"],
    "command_not_found": ["a required tool is not installed — this is an environment problem, not a code bug"],
    "environment": ["the environment is misconfigured (missing services, variables or permissions)"],
}

NON_CODE_CLASSIFICATIONS = {"command_not_found", "environment"}


class FailureRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("fail"))
    ts: str = Field(default_factory=utcnow_iso)
    task_id: str = ""
    subtask_id: str = ""
    attempt: int = 0
    error: str
    context: str = ""
    likely_causes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    hypothesis: str = ""
    change: str = ""
    fix_fingerprint: str = ""
    result: str = "pending"  # pending | fixed | still_failing | different_failure | abandoned
    signature: str = ""
    classification: str = ""

    def render(self) -> str:
        lines = [
            f"ERROR: {self.error}",
            f"CONTEXT: {self.context}" if self.context else "",
            "LIKELY CAUSES: " + "; ".join(self.likely_causes) if self.likely_causes else "",
            "EVIDENCE:\n" + "\n".join(f"  - {e}" for e in self.evidence[:15]) if self.evidence else "",
            f"HYPOTHESIS: {self.hypothesis}" if self.hypothesis else "",
            f"CHANGE: {self.change}" if self.change else "",
            f"RESULT: {self.result}" if self.result != "pending" else "",
        ]
        return "\n".join(line for line in lines if line)


def _evidence(results: Sequence[Any], limit: int = 20) -> list[str]:
    evidence: list[str] = []
    for r in results:
        for f in getattr(r, "failures", [])[:limit]:
            loc = f"{f.file}:{f.line}" if f.file and f.line else (f.file or "")
            evidence.append(f"[{r.kind}] {f.test_id}{' (' + loc + ')' if loc else ''}: {f.message[:300]}")
        for d in getattr(r, "diagnostics", [])[:limit]:
            loc = f"{d.file}:{d.line}" if d.file else ""
            evidence.append(f"[{r.kind}] {loc} {d.code or ''} {d.message[:300]}".strip())
        if not getattr(r, "failures", None) and not getattr(r, "diagnostics", None):
            tail = (r.output_tail or "").strip().splitlines()[-12:]
            if tail:
                evidence.append(f"[{r.kind}] output tail:\n    " + "\n    ".join(line[:300] for line in tail))
    return evidence[: limit * 2]


_NOT_RUN = ("unavailable", "skipped", "cancelled")


def is_failure(result: Any) -> bool:
    """A check that ran and failed (checks that could not run are not failures)."""
    return not result.ok() and getattr(result, "status", "failed") not in _NOT_RUN


def combined_signature(results: Sequence[Any]) -> str:
    sigs = sorted(f"{r.kind}:{r.signature()}" for r in results if is_failure(r))
    return sha1_text("|".join(sigs))[:16] if sigs else ""


class FailureTracker:
    """Per-subtask record of failures and fix attempts with loop detection."""

    def __init__(self, task_id: str = "", subtask_id: str = "", max_same_signature: int = 3) -> None:
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.max_same_signature = max_same_signature
        self.records: list[FailureRecord] = []
        self._fix_fingerprints: set[str] = set()
        self.duplicate_fixes = 0

    def record_failure(self, results: Sequence[Any], context: str) -> FailureRecord:
        failing = [r for r in results if is_failure(r)]
        classifications = [r.classification for r in failing if r.classification]
        causes: list[str] = []
        for cls in classifications:
            for cause in LIKELY_CAUSES.get(cls, []):
                if cause not in causes:
                    causes.append(cause)
        record = FailureRecord(
            task_id=self.task_id,
            subtask_id=self.subtask_id,
            attempt=len(self.records) + 1,
            error="; ".join(r.summary or f"{r.kind} failed" for r in failing) or "validation failed",
            context=context,
            likely_causes=causes,
            evidence=_evidence(failing),
            signature=combined_signature(failing),
            classification=",".join(dict.fromkeys(classifications)),
        )
        self.records.append(record)
        return record

    def record_fix(self, record: FailureRecord, hypothesis: str, change: str, diff_text: str) -> bool:
        """Attach the attempted fix. Returns False when this exact change was already tried."""
        fingerprint = sha1_text(diff_text)[:16] if diff_text.strip() else ""
        record.hypothesis = hypothesis.strip()[:2000]
        record.change = change.strip()[:2000]
        record.fix_fingerprint = fingerprint
        if not fingerprint:
            return True
        if fingerprint in self._fix_fingerprints:
            self.duplicate_fixes += 1
            return False
        self._fix_fingerprints.add(fingerprint)
        return True

    def record_outcome(self, record: FailureRecord, results: Sequence[Any]) -> str:
        failing = [r for r in results if is_failure(r)]
        if not failing:
            record.result = "fixed"
        elif combined_signature(failing) == record.signature:
            record.result = "still_failing"
        else:
            record.result = "different_failure"
        return record.result

    def only_environmental(self, results: Sequence[Any]) -> bool:
        failing = [r for r in results if is_failure(r)]
        return bool(failing) and all(r.classification in NON_CODE_CLASSIFICATIONS for r in failing)

    def loop_detected(self) -> str | None:
        sigs = [r.signature for r in self.records if r.signature]
        if len(sigs) >= self.max_same_signature and len(set(sigs[-self.max_same_signature :])) == 1:
            return f"the same failure persisted through {self.max_same_signature} consecutive attempts"
        if len(sigs) >= 4 and sigs[-1] == sigs[-3] and sigs[-2] == sigs[-4] and sigs[-1] != sigs[-2]:
            return "fixes are oscillating between two failure states"
        if self.duplicate_fixes >= 2:
            return "the same fix was produced repeatedly"
        return None

    def history_prompt(self) -> str:
        done = [r for r in self.records if r.result != "pending" or r.change]
        if not done:
            return ""
        lines = ["Previous fix attempts for this failure (do NOT repeat a failed approach):"]
        for r in done:
            lines.append(
                f"- Attempt {r.attempt}: hypothesis: {r.hypothesis or 'n/a'} | change: {r.change or 'n/a'} | result: {r.result}"
            )
        return "\n".join(lines)
