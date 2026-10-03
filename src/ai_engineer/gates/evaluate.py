"""Quality gates.

A task is COMPLETE only when every configured gate actually ran and passed.
Gates that could not run are UNVERIFIED (never silently passed). Failures that
already existed before the agent started (baseline) are reported as
PRE-EXISTING: they do not block, but the task is then COMPLETED_UNVERIFIED for
that check because the agent's change cannot be proven clean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config.settings import GateMode, GatesSettings

CHECK_GATES = ("tests", "lint", "typecheck", "build")


class GateStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNVERIFIED = "UNVERIFIED"
    PREEXISTING = "PRE-EXISTING"


class GateResult(BaseModel):
    name: str
    mode: GateMode
    status: GateStatus
    detail: str = ""
    evidence: list[str] = Field(default_factory=list)


class GateReport(BaseModel):
    results: list[GateResult]
    verdict: Literal["COMPLETED", "COMPLETED_UNVERIFIED", "FAILED"]

    def get(self, name: str) -> GateResult | None:
        return next((r for r in self.results if r.name == name), None)

    def failed(self) -> list[GateResult]:
        return [r for r in self.results if r.status == GateStatus.FAILED]

    def unverified(self) -> list[GateResult]:
        return [r for r in self.results if r.status in (GateStatus.UNVERIFIED, GateStatus.PREEXISTING)]

    def table(self) -> str:
        lines = ["| Gate | Mode | Status | Detail |", "|---|---|---|---|"]
        for r in self.results:
            detail = r.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r.name} | {r.mode} | {r.status} | {detail} |")
        return "\n".join(lines)

    def checklist(self) -> str:
        marks = {GateStatus.PASSED: "[x]", GateStatus.FAILED: "[ ] FAILED:", GateStatus.SKIPPED: "[-] skipped:", GateStatus.UNVERIFIED: "[?] UNVERIFIED:", GateStatus.PREEXISTING: "[~] pre-existing failures:"}
        return "\n".join(f"{marks[r.status]} {r.name} — {r.detail}" for r in self.results)


@dataclass
class GateInputs:
    requirements_understood: bool = True
    implementation_done: bool = True
    implementation_detail: str = ""
    checks: dict[str, Any] = field(default_factory=dict)  # gate name -> CheckResult | None
    baseline: dict[str, Any] = field(default_factory=dict)
    security_findings: list[Any] = field(default_factory=list)
    security_ran: bool = False
    audit: Any = None
    audit_baseline: Any = None
    review: Any = None
    docs: tuple[str, str] | None = None  # (passed|failed|skipped|unverified, detail)
    git_state: tuple[bool, str] | None = None
    is_git: bool = True
    in_scope_tests: set[str] = field(default_factory=set)


_NUM = re.compile(r"\d+")


def _test_file(f: Any) -> str:
    return str(getattr(f, "file", None) or str(getattr(f, "test_id", "")).split("::")[0]).replace("\\", "/")


def failure_entries(result: Any) -> dict[str, str]:
    """Failure key -> file the failure belongs to."""
    entries: dict[str, str] = {}
    for f in getattr(result, "failures", []) or []:
        entries[f"test:{f.test_id}"] = _test_file(f)
    for d in getattr(result, "diagnostics", []) or []:
        entries[f"diag:{d.file}:{d.code}:{_NUM.sub('#', d.message)[:120]}"] = str(d.file or "")
    return entries


def failure_keys(result: Any) -> set[str]:
    return set(failure_entries(result))


def compare_with_baseline(current: Any, baseline: Any, in_scope_tests: set[str] | None = None) -> tuple[bool, list[str]]:
    """Returns (only_preexisting_failures, new_failure_keys).

    Baseline test failures in ``in_scope_tests`` (tests related to the files the agent
    changed) are *not* excused: fixing them is usually the point of the task.
    """
    if baseline is None or baseline.ok() or baseline.status in ("unavailable", "skipped"):
        return False, sorted(failure_keys(current))
    if current.status != baseline.status:
        # e.g. the suite used to fail and now hangs (timeout) or crashes: never "pre-existing"
        return False, [f"(status changed from {baseline.status} to {current.status})", *sorted(failure_keys(current))]
    cur = failure_keys(current)
    base_entries = failure_entries(baseline)
    if in_scope_tests:
        scope = {t.replace("\\", "/") for t in in_scope_tests}
        base_entries = {k: f for k, f in base_entries.items() if not (k.startswith("test:") and any(f == t or f.endswith("/" + t) or t.endswith("/" + f) for t in scope))}
    base = set(base_entries)
    if not cur:
        # nothing parsed from the current output: only identical output counts as the old failure
        same = current.signature() == baseline.signature()
        return same, [] if same else ["(unstructured output differs from baseline)"]
    new = cur - base
    return not new, sorted(new)


def _check_gate(name: str, mode: GateMode, result: Any, baseline: Any, in_scope: set[str] | None = None) -> GateResult:
    if result is None or result.status in ("unavailable", "skipped"):
        reason = (getattr(result, "summary", "") or f"no {name} command detected") if result is not None else f"no {name} command detected"
        status = GateStatus.UNVERIFIED if mode == GateMode.REQUIRED else GateStatus.SKIPPED
        return GateResult(name=name, mode=mode, status=status, detail=reason)
    if result.ok():
        return GateResult(name=name, mode=mode, status=GateStatus.PASSED, detail=result.summary or "passed")
    if name == "tests" and result.classification == "no_tests":
        status = GateStatus.UNVERIFIED if mode == GateMode.REQUIRED else GateStatus.SKIPPED
        return GateResult(name=name, mode=mode, status=status, detail="no tests exist or none were collected")
    if result.classification in ("command_not_found",):
        status = GateStatus.UNVERIFIED if mode == GateMode.REQUIRED else GateStatus.SKIPPED
        return GateResult(name=name, mode=mode, status=status, detail=f"could not run: {result.summary}")
    only_pre, new = compare_with_baseline(result, baseline, in_scope if name == "tests" else None)
    if only_pre:
        return GateResult(
            name=name, mode=mode, status=GateStatus.PREEXISTING,
            detail=f"failures already present before the change; no new failures ({result.summary})",
        )
    evidence = new[:20]
    return GateResult(name=name, mode=mode, status=GateStatus.FAILED, detail=result.summary or "failed", evidence=evidence)


def evaluate_gates(settings: GatesSettings, inputs: GateInputs) -> GateReport:
    results: list[GateResult] = []

    def add(name: str, build: Any) -> None:
        mode: GateMode = getattr(settings, name)
        if mode == GateMode.DISABLED:
            results.append(GateResult(name=name, mode=mode, status=GateStatus.SKIPPED, detail="disabled by configuration"))
            return
        results.append(build(mode))

    add("requirements", lambda m: GateResult(
        name="requirements", mode=m,
        status=GateStatus.PASSED if inputs.requirements_understood else GateStatus.FAILED,
        detail="requirements and acceptance criteria recorded" if inputs.requirements_understood else "requirements could not be established",
    ))
    add("implementation", lambda m: GateResult(
        name="implementation", mode=m,
        status=GateStatus.PASSED if inputs.implementation_done else GateStatus.FAILED,
        detail=inputs.implementation_detail or ("implementation completed" if inputs.implementation_done else "implementation incomplete"),
    ))
    for gate in ("typecheck", "lint", "tests", "build"):
        add(gate, lambda m, g=gate: _check_gate(g, m, inputs.checks.get(g), inputs.baseline.get(g), inputs.in_scope_tests))

    def security(mode: GateMode) -> GateResult:
        if not inputs.security_ran:
            return GateResult(name="security", mode=mode, status=GateStatus.UNVERIFIED if mode == GateMode.REQUIRED else GateStatus.SKIPPED, detail="security scan did not run")
        high = [f for f in inputs.security_findings if f.severity == "high"]
        other = [f for f in inputs.security_findings if f.severity != "high"]
        evidence = [f"{f.path}:{f.line} {f.rule}: {f.message}" for f in inputs.security_findings][:20]
        audit_note = ""
        if inputs.audit is not None and inputs.audit.status not in ("unavailable", "skipped"):
            if not inputs.audit.ok():
                only_pre, _ = compare_with_baseline(inputs.audit, inputs.audit_baseline)
                if not only_pre:
                    return GateResult(name="security", mode=mode, status=GateStatus.FAILED, detail=f"dependency audit found new issues: {inputs.audit.summary}", evidence=evidence)
                audit_note = "; dependency audit: pre-existing advisories"
            else:
                audit_note = "; dependency audit clean"
        if high:
            return GateResult(name="security", mode=mode, status=GateStatus.FAILED, detail=f"{len(high)} high-severity finding(s) in changed code", evidence=evidence)
        detail = "no high-severity findings in changed code" + (f" ({len(other)} lower-severity warning(s))" if other else "") + audit_note
        return GateResult(name="security", mode=mode, status=GateStatus.PASSED, detail=detail, evidence=evidence)

    add("security", security)

    def review(mode: GateMode) -> GateResult:
        r = inputs.review
        if r is None:
            return GateResult(name="review", mode=mode, status=GateStatus.UNVERIFIED if mode == GateMode.REQUIRED else GateStatus.SKIPPED, detail="independent review was not performed")
        blocking = r.blocking()
        evidence = [f"{i.severity}: {i.description}" for i in r.issues][:20]
        if blocking:
            return GateResult(name="review", mode=mode, status=GateStatus.FAILED, detail=f"{len(blocking)} blocking issue(s) remain", evidence=evidence)
        unmet = [c for c in r.requirements if c.status == "unmet"]
        if unmet:
            return GateResult(name="review", mode=mode, status=GateStatus.FAILED, detail=f"{len(unmet)} acceptance criterion/criteria unmet", evidence=[c.criterion for c in unmet])
        if getattr(r, "verdict", "approve") == "request_changes":
            return GateResult(name="review", mode=mode, status=GateStatus.FAILED, detail=f"the reviewer requested changes: {r.summary[:300]}", evidence=evidence)
        minor = len(r.issues)
        if r.source == "deterministic-only":
            status = GateStatus.UNVERIFIED if mode == GateMode.REQUIRED else GateStatus.SKIPPED
            return GateResult(name="review", mode=mode, status=status, detail=f"independent model review unavailable; automated checks found no blocking issues ({r.summary})", evidence=evidence)
        return GateResult(name="review", mode=mode, status=GateStatus.PASSED, detail=f"approved ({r.source})" + (f"; {minor} non-blocking note(s)" if minor else ""), evidence=evidence)

    add("review", review)

    def docs(mode: GateMode) -> GateResult:
        if inputs.docs is None:
            return GateResult(name="docs", mode=mode, status=GateStatus.SKIPPED, detail="no documentation impact identified")
        status, detail = inputs.docs
        mapping = {"passed": GateStatus.PASSED, "failed": GateStatus.FAILED, "skipped": GateStatus.SKIPPED, "unverified": GateStatus.UNVERIFIED}
        st = mapping.get(status, GateStatus.UNVERIFIED)
        if st == GateStatus.UNVERIFIED and mode == GateMode.IF_AVAILABLE:
            st = GateStatus.SKIPPED
        return GateResult(name="docs", mode=mode, status=st, detail=detail)

    add("docs", docs)

    def git_state(mode: GateMode) -> GateResult:
        if not inputs.is_git:
            return GateResult(name="git_state", mode=mode, status=GateStatus.SKIPPED, detail="workspace is not a git repository (file checkpoints used)")
        if inputs.git_state is None:
            return GateResult(name="git_state", mode=mode, status=GateStatus.UNVERIFIED if mode == GateMode.REQUIRED else GateStatus.SKIPPED, detail="git state not verified")
        ok, detail = inputs.git_state
        return GateResult(name="git_state", mode=mode, status=GateStatus.PASSED if ok else GateStatus.FAILED, detail=detail)

    add("git_state", git_state)

    counted = [r for r in results if r.mode != GateMode.DISABLED]
    if any(r.status == GateStatus.FAILED for r in counted):
        verdict: Literal["COMPLETED", "COMPLETED_UNVERIFIED", "FAILED"] = "FAILED"
    elif any(r.status in (GateStatus.UNVERIFIED, GateStatus.PREEXISTING) for r in counted if r.mode == GateMode.REQUIRED) or any(
        r.status == GateStatus.PREEXISTING for r in counted
    ):
        verdict = "COMPLETED_UNVERIFIED"
    else:
        verdict = "COMPLETED"
    return GateReport(results=results, verdict=verdict)
