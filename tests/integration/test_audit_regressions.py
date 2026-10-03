"""Regression tests for the bugs confirmed by the correctness audit.

Each test pins down the fixed behaviour of one reproduced bug: real repositories and real
test commands, scripted (deterministic) models. The docstring of each test names the bug.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from ai_engineer.config.settings import GatesSettings
from ai_engineer.core.errors import ContextLengthError, StateError
from ai_engineer.core.events import EventType
from ai_engineer.core.types import Message
from ai_engineer.executor.loop import AgentLoop, LoopLimits
from ai_engineer.gates.evaluate import GateInputs, GateStatus, compare_with_baseline, evaluate_gates
from ai_engineer.models.base import ModelRequest
from ai_engineer.providers.scripted import ScriptedProvider
from ai_engineer.reviewer.review import CriterionStatus, ReviewIssue, ReviewResult, combine
from ai_engineer.runtime import Runtime
from ai_engineer.tasks.store import StateStore, Task, TaskStatus
from ai_engineer.tester.models import CheckKind, CheckResult, Diagnostic, TestCaseFailure
from ai_engineer.tools.process import run_shell
from tests.conftest import git, make_router
from tests.integration.helpers import APPROVE, call, j, make_calc_repo, open_runtime, understanding

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX shell scripts / signals")

FIX_ADD = call("edit_file", path="calc/__init__.py", old_string="return a - b\n\n\ndef sub", new_string="return a + b\n\n\ndef sub")


def _gates(state: dict[str, Any]) -> dict[str, str]:
    return {g["name"]: g["status"] for g in (state.get("gates") or {}).get("results", [])}


def _requests(provider: ScriptedProvider, role: str, stage: str | None = None) -> list[ModelRequest]:
    return [r for r in provider.requests if r.metadata.get("role") == role and (stage is None or r.metadata.get("stage") == stage)]


async def _run(rt: Runtime, task_id: str, **kwargs: Any) -> Task:
    try:
        return await rt.orchestrator.run(task_id, **kwargs)
    finally:
        await rt.aclose()


# --------------------------------------------------------------------------- 1. hang vs baseline


def test_hang_without_parsed_failures_is_not_excused_by_unrelated_baseline_failure() -> None:
    """repro_01 (unit): a timeout is a status change, never a 'pre-existing' failure."""
    baseline = CheckResult(kind=CheckKind.TEST, command="pytest", status="failed",
                           failures=[TestCaseFailure(test_id="tests/test_other.py::test_x", file="tests/test_other.py")])
    hang = CheckResult(kind=CheckKind.TEST, command="pytest", status="timeout", classification="timeout", output_tail="....")
    only_preexisting, new = compare_with_baseline(hang, baseline, {"tests/test_calc.py"})
    assert only_preexisting is False
    assert new and "timeout" in new[0]
    report = evaluate_gates(GatesSettings(), GateInputs(checks={"tests": hang}, baseline={"tests": baseline}, security_ran=True))
    assert report.get("tests").status == GateStatus.FAILED


async def test_agent_introduced_hang_is_repaired_and_never_committed(tmp_path: Path) -> None:
    """repro_01 (pipeline): a hang next to an unrelated failing test fails the tests gate and goes to repair."""
    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    (repo / "tests" / "test_other.py").write_text("def test_x():\n    assert 1 == 2  # pre-existing, unrelated\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "pre-existing failing test")
    script = {
        "classifier": [understanding(summary="Make add() faster", requirements=["add stays correct"])],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            call("edit_file", path="calc/__init__.py", old_string="return a + b\n", new_string="while True:\n        pass\n"),
            call("submit_work", summary="optimised add", files_changed=["calc/__init__.py"]),
        ],
        "debugger": [call("submit_work", summary="could not find the problem")] * 4,
        "reviewer": [APPROVE] * 2,
    }
    provider = ScriptedProvider("s", by_role=script)
    rt = open_runtime(repo, {"s": provider}, validation={"test_timeout_s": 6}, agent={"max_repair_iterations": 1})
    task = rt.create_task("Make add() in calc faster")
    result = await _run(rt, task.id)

    assert result.state["baseline"]["test"]["status"] == "failed"
    sub = result.state["subtasks"]["s1"]
    assert [v["status"] for v in sub["validation"] if v["kind"] == "test"] == ["timeout"]
    assert sub["repair_iterations"] == 1 and sub["status"] == "failed"
    assert _requests(provider, "debugger"), "the debugger must be consulted for the hang"
    assert _gates(result.state)["tests"] == "FAILED"
    assert {g["name"]: g["status"] for g in sub["gates"]["results"]}["tests"] == "FAILED"
    assert result.status == TaskStatus.FAILED
    assert "while True" not in git(repo, "show", "HEAD:calc/__init__.py")
    assert "aie:" not in git(repo, "log", "--format=%s")


# --------------------------------------------------------------------------- 2. docs stage


async def test_docs_stage_edits_are_validated(tmp_path: Path) -> None:
    """repro_02: code broken by the docs stage is caught by final validation (not COMPLETED/PASSED)."""
    repo = make_calc_repo(tmp_path / "repo")
    script = {
        "classifier": [understanding(needs={"tests": True, "docs_update": True})],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            FIX_ADD,
            call("submit_work", summary="fixed add", files_changed=["calc/__init__.py"]),
            # docs stage: "updates the docstring" and breaks the code while at it
            call("read_file", path="calc/__init__.py"),
            call("edit_file", path="calc/__init__.py", old_string="return a + b\n\n\ndef sub", new_string="return a * b  # documented\n\n\ndef sub"),
            call("write_file", path="README.md", content="# calc\n\nadd(a, b) returns the sum.\n"),
            call("submit_work", summary="docs updated", files_changed=["README.md", "calc/__init__.py"]),
        ],
        "debugger": [call("submit_work", summary="no idea")] * 2,
        "reviewer": [APPROVE] * 3,
    }
    provider = ScriptedProvider("s", by_role=script)
    rt = open_runtime(repo, {"s": provider}, agent={"max_repair_iterations": 1})
    task = rt.create_task("Fix the add function in calc so the tests pass; document the CLI api")
    result = await _run(rt, task.id)

    assert _requests(provider, "coder", "docs"), "the docs stage must have run"
    assert "return a * b" in (repo / "calc" / "__init__.py").read_text()
    final_tests = [v["status"] for v in result.state["final_validation"] if v["kind"] == "test"]
    assert final_tests == ["failed"]
    assert _requests(provider, "debugger"), "the failure introduced by the docs stage must go to repair"
    assert _gates(result.state)["tests"] == "FAILED"
    assert result.status != TaskStatus.COMPLETED
    assert result.status == TaskStatus.FAILED


# --------------------------------------------------------------------------- 3. deleted test file


async def test_deleting_a_test_file_is_flagged_by_the_subtask_review(tmp_path: Path) -> None:
    """repro_03: 'fixing' a failing test by deleting its file is a major review issue, not COMPLETED."""
    repo = make_calc_repo(tmp_path / "repo")
    (repo / "tests" / "test_sub_only.py").write_text("from calc import sub\n\n\ndef test_sub2():\n    assert sub(3, 1) == 2\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "another test file")
    script = {
        "classifier": [understanding()],
        "coder": [
            call("delete_file", path="tests/test_calc.py"),
            call("submit_work", summary="removed the failing test", files_changed=["tests/test_calc.py"]),
            call("submit_work", summary="nothing else to change"),  # review-fix round
        ],
        "reviewer": [APPROVE] * 3,
    }
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)}, agent={"max_review_iterations": 1})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    result = await _run(rt, task.id)

    assert not (repo / "tests" / "test_calc.py").exists()
    sub = result.state["subtasks"]["s1"]
    issues = [(i["severity"], i["description"]) for i in sub["review"]["issues"]]
    assert ("major", "a test file was deleted") in issues
    assert sub["review"]["verdict"] == "request_changes"
    assert result.status != TaskStatus.COMPLETED
    assert _gates(result.state)["review"] == "FAILED"


# --------------------------------------------------------------------------- 4. resume mid-implementation


async def test_resume_mid_implementation_reenters_the_implementer(tmp_path: Path) -> None:
    """repro_13: an interrupted subtask is continued by the implementer, so the missing part gets done."""
    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    holder: dict[str, Any] = {}

    def coder_run1(req: ModelRequest) -> dict[str, Any]:
        n = holder.setdefault("n", 0)
        holder["n"] = n + 1
        if n == 0:
            return call("read_file", path="calc/__init__.py")
        if n == 1:  # first half of the work: mul()
            return call("edit_file", path="calc/__init__.py", old_string="def sub(a, b):", new_string="def mul(a, b):\n    return a * b\n\n\ndef sub(a, b):")
        holder["rt"].tool_ctx.cancel.cancel("stop requested (Ctrl+C)")  # the user stops the run
        return call("list_directory", path=".")

    u = understanding(summary="Add mul() and div() to calc", requirements=["mul(a, b) returns a * b", "div(a, b) returns a / b"],
                      acceptance_criteria=["calc.mul and calc.div exist"])
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role={"classifier": [u], "coder": [coder_run1] * 4})})
    holder["rt"] = rt
    task = rt.create_task("Add mul() and div() to calc")
    first = await _run(rt, task.id)
    assert first.status == TaskStatus.INTERRUPTED
    assert first.state["subtasks"]["s1"]["status"] == "running"

    coder2 = [
        call("read_file", path="calc/__init__.py"),
        call("edit_file", path="calc/__init__.py", old_string="def sub(a, b):", new_string="def div(a, b):\n    return a / b\n\n\ndef sub(a, b):"),
        call("submit_work", summary="added mul and div"),
    ]
    provider2 = ScriptedProvider("s", by_role={"coder": coder2, "reviewer": [APPROVE] * 2})
    resumed = await _run(open_runtime(repo, {"s": provider2}), task.id)

    assert provider2.by_role["coder"] == [], "the scripted implementer steps must be used after the resume"
    coder_requests = _requests(provider2, "coder")
    assert "Resuming an interrupted attempt" in coder_requests[0].messages[0].text()
    assert "calc/__init__.py" in coder_requests[0].messages[0].text()
    text = (repo / "calc" / "__init__.py").read_text()
    assert "def mul" in text and "def div" in text
    sub = resumed.state["subtasks"]["s1"]
    assert sub["loop_status"] == "finished"
    assert resumed.status == TaskStatus.COMPLETED, resumed.error


# --------------------------------------------------------------------------- 5. aie plan and branches


async def test_plan_only_does_not_switch_branches_and_resume_never_commits_on_main(tmp_path: Path) -> None:
    """repro_04: `aie plan` leaves the branch alone; a resume on a dirty main does not commit there."""
    repo = make_calc_repo(tmp_path / "repo")
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role={"classifier": [understanding()]})})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    planned = await _run(rt, task.id, stop_after="plan")
    assert planned.status == TaskStatus.PENDING
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert git(repo, "branch", "--list", "aie/*").strip() == ""

    # the user keeps working on main before resuming
    p = repo / "calc" / "__init__.py"
    p.write_text(p.read_text().replace('"""Tiny calculator."""', '"""Tiny calculator. (user WIP, not ready)"""'))
    script = {
        "coder": [call("read_file", path="calc/__init__.py"), FIX_ADD, call("submit_work", summary="fixed add", files_changed=["calc/__init__.py"])],
        "reviewer": [APPROVE] * 2,
    }
    resumed = await _run(open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)}), task.id)

    assert resumed.status == TaskStatus.COMPLETED, resumed.error
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert git(repo, "log", "--format=%s", "main").strip().splitlines() == ["initial"]
    assert "user WIP" not in git(repo, "show", "main:calc/__init__.py")
    text = p.read_text()
    assert "user WIP" in text and "return a + b" in text  # both left in the working tree for the user


# --------------------------------------------------------------------------- 6. non-git resume


def _make_plain_calc(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "calc").mkdir()
    (root / "calc" / "__init__.py").write_text("def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_calc.py").write_text(
        "from calc import add, sub\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\ndef test_sub():\n    assert sub(5, 3) == 2\n"
    )
    (root / "pyproject.toml").write_text('[project]\nname = "calc"\nversion = "0.0.1"\n')
    return root


async def test_non_git_workspace_tracks_and_restores_edits_after_resume(tmp_path: Path) -> None:
    """repro_08: file-backup checkpoints keep working after a resume in a new process."""
    ws = _make_plain_calc(tmp_path / "ws")
    down = ScriptedProvider("s", responder=lambda req: {"raise": {"type": "unavailable"}})
    rt = open_runtime(ws, {"s": down})
    assert rt.git is None, "the workspace must not be a git repository"
    task = rt.create_task("Fix the add function in calc so the tests pass")
    first = await _run(rt, task.id)
    assert first.status == TaskStatus.BLOCKED
    assert first.state["subtasks"]["s1"]["status"] == "running"

    script = {
        "coder": [
            call("read_file", path="calc/__init__.py"),
            call("edit_file", path="calc/__init__.py", old_string="    return a - b\n\n\ndef sub", new_string="    return a + b\n\n\ndef sub"),
            call("submit_work", summary="fixed add", files_changed=["calc/__init__.py"]),
        ],
        "reviewer": [APPROVE] * 2,
    }
    rt2 = open_runtime(ws, {"s": ScriptedProvider("s", by_role=script)})
    try:
        resumed = await rt2.orchestrator.run(task.id)
        since_sub = await rt2.checkpoints.changed_files_since(resumed.state["subtasks"]["s1"]["checkpoint_before"])
        since_start = await rt2.checkpoints.changed_files_since(resumed.state["start_checkpoint"])
    finally:
        await rt2.aclose()
    assert since_sub == ["calc/__init__.py"]
    assert since_start == ["calc/__init__.py"]
    assert _gates(resumed.state)["implementation"] == "PASSED"
    assert resumed.status == TaskStatus.COMPLETED, resumed.error

    rt3 = Runtime.open(ws, use_global_config=False)
    try:
        _, restored = await rt3.checkpoints.restore(resumed.state["start_checkpoint"])
    finally:
        await rt3.aclose()
    assert restored == ["calc/__init__.py"]
    assert "return a - b\n\n\ndef sub" in (ws / "calc" / "__init__.py").read_text()


# --------------------------------------------------------------------------- 7. concurrent Runtime.open


async def test_opening_another_runtime_does_not_interrupt_a_live_subtask(tmp_path: Path) -> None:
    """repro_05: recover_interrupted() in another Runtime leaves the running subtask alone."""
    repo = make_calc_repo(tmp_path / "repo")
    seen: dict[str, Any] = {}

    async def coder_first(req: ModelRequest) -> dict[str, Any]:
        # e.g. `aie status` run from another terminal while the subtask is executing
        other = Runtime.open(repo, use_global_config=False)
        try:
            child = other.store.get_task(seen["task_id"] + ":s1")
            seen["child"] = (child.status, child.error)
        finally:
            await other.aclose()
        return call("read_file", path="calc/__init__.py")

    script = {
        "classifier": [understanding()],
        "coder": [coder_first, FIX_ADD, call("submit_work", summary="fixed add", files_changed=["calc/__init__.py"])],
        "reviewer": [APPROVE] * 2,
    }
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    seen["task_id"] = task.id
    try:
        result = await rt.orchestrator.run(task.id)
        child = rt.store.get_task(task.id + ":s1")
        interrupted = rt.store.events(task.id + ":s1", types=[str(EventType.TASK_INTERRUPTED)])
    finally:
        await rt.aclose()
    assert seen["child"] == (TaskStatus.RUNNING, None)
    assert result.status == TaskStatus.COMPLETED, result.error
    assert child.status == TaskStatus.COMPLETED and child.error is None
    assert interrupted == []


# --------------------------------------------------------------------------- 8. circuit breaker


async def test_context_length_errors_do_not_open_the_circuit_breaker() -> None:
    """repro_06: request-specific errors say nothing about model health."""
    primary = ScriptedProvider("a", steps=[{"raise": {"type": "context"}}] * 3 + ["primary answer"])
    backup = ScriptedProvider("b", steps=["backup answer"] * 3)
    router = make_router(primary, backup, roles={"default": ["a:big", "b:small"]})
    coder = router.for_role("default")
    for _ in range(3):
        with pytest.raises(ContextLengthError):
            await coder.generate(ModelRequest(messages=[Message.user("x" * 10)]))
    assert not router.breaker.is_open("a:big")
    resp = await coder.generate(ModelRequest(messages=[Message.user("small request")]))
    assert resp.provider == "a" and resp.text() == "primary answer"
    assert backup.requests == []


# --------------------------------------------------------------------------- 9. agent loop budgets


async def test_oversized_task_context_does_not_reset_every_step(tmp_path: Path) -> None:
    """repro_09 (A): when the fresh context alone exceeds the budget, tool results are kept."""
    repo = make_calc_repo(tmp_path / "repo")
    provider = ScriptedProvider("s", steps=[call("read_file", path="calc/__init__.py"), FIX_ADD, call("submit_work", summary="fixed add")])
    rt = open_runtime(repo, {"s": provider})
    events: list[Any] = []
    rt.bus.subscribe(events.append)
    task_message = "Fix add() in calc.\n\n# Context\n" + ("relevant context line\n" * 1800)  # ~10k tokens
    loop = AgentLoop(rt.router.for_role("coder"), rt.executor, rt.tool_ctx, role="coder", stage="implement",
                     system="You are a coder.", tools=["read_file", "edit_file", "list_directory"], finish_tool="submit_work",
                     limits=LoopLimits(max_steps=10, context_budget_tokens=8000))
    try:
        result = await loop.run(task_message)
    finally:
        await rt.aclose()
    assert sum(1 for e in events if e.type == EventType.CONTEXT_COMPACTED) == 0
    assert [len(r.messages) for r in provider.requests] == [1, 3, 5]
    edit_results = [b for msg in provider.requests[-1].messages for b in msg.content
                    if getattr(b, "type", "") == "tool_result" and b.name == "edit_file"]
    assert edit_results and "edited calc/__init__.py" in edit_results[0].content
    assert result.status == "finished"
    assert "return a + b" in (repo / "calc" / "__init__.py").read_text()


async def test_text_turns_between_tool_calls_do_not_end_the_loop(tmp_path: Path) -> None:
    """repro_09 (B): the nudge limit counts consecutive text-only turns only."""
    repo = make_calc_repo(tmp_path / "repo")
    steps = [
        "Let me look at the package first.", call("list_directory", path="."),
        "Now the module.", call("read_file", path="calc/__init__.py"),
        "I see the bug, fixing it now.", FIX_ADD,
        call("submit_work", summary="fixed add operator"),
    ]
    provider = ScriptedProvider("s", steps=steps)
    rt = open_runtime(repo, {"s": provider})
    loop = AgentLoop(rt.router.for_role("coder"), rt.executor, rt.tool_ctx, role="coder", stage="implement",
                     system="You are a coder.", tools=["read_file", "edit_file", "list_directory"], finish_tool="submit_work",
                     limits=LoopLimits(max_steps=20))
    try:
        result = await loop.run("Fix add() in calc.")
    finally:
        await rt.aclose()
    assert result.status == "finished"
    assert provider.remaining() == 0
    assert "return a + b" in (repo / "calc" / "__init__.py").read_text()


# --------------------------------------------------------------------------- 10. UTF-8 chunks


async def test_multibyte_characters_split_across_reads_are_decoded(tmp_path: Path) -> None:
    """repro_10: a character split across two 64 KiB pipe reads must not become U+FFFD."""
    (tmp_path / "emit.py").write_text('import sys\nsys.stdout.buffer.write(b"a" + "\\u00e9".encode("utf-8") * 40000)\n')
    out = await run_shell(f'"{sys.executable}" emit.py', tmp_path, dict(os.environ), 60, 10**6)
    assert out.exit_code == 0
    assert "\ufffd" not in out.output
    assert out.output.count("\u00e9") == 40000


# --------------------------------------------------------------------------- 11. reviewer verdict


def _minor_rejection() -> ReviewResult:
    return ReviewResult(
        verdict="request_changes",
        summary="Do not merge: add() now silently truncates floats (add(0.5, 0.5) == 0).",
        issues=[ReviewIssue(severity="minor", category="correctness", file="calc/__init__.py", description="float handling regressed")],
        requirements=[CriterionStatus(criterion="tests/test_calc.py passes", status="met", evidence="pytest")],
        source="model",
    )


def test_request_changes_with_only_minor_issues_is_honoured_by_combine_and_gates() -> None:
    """repro_11 (unit): the reviewer's explicit request_changes verdict stands."""
    combined = combine([], _minor_rejection())
    assert combined.verdict == "request_changes"
    report = evaluate_gates(GatesSettings(), GateInputs(review=combined, security_ran=True))
    assert report.get("review").status == GateStatus.FAILED
    direct = evaluate_gates(GatesSettings(), GateInputs(review=_minor_rejection(), security_ran=True))
    assert direct.get("review").status == GateStatus.FAILED
    assert direct.verdict == "FAILED"


async def test_reviewer_request_changes_triggers_a_fix_round(tmp_path: Path) -> None:
    """repro_11 (pipeline): request_changes is stored, a fix round is requested and the gate fails."""
    repo = make_calc_repo(tmp_path / "repo")
    reject = j({
        "verdict": "request_changes",
        "summary": "Do not merge: add() now silently truncates floats (add(0.5, 0.5) == 0); this breaks callers.",
        "issues": [{"severity": "minor", "category": "correctness", "file": "calc/__init__.py", "description": "float handling regressed"}],
        "requirements": [{"criterion": "tests/test_calc.py passes", "status": "met", "evidence": "pytest"}],
    })
    script = {
        "classifier": [understanding()],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            call("edit_file", path="calc/__init__.py", old_string="return a - b\n\n\ndef sub", new_string="return int(a) + int(b)\n\n\ndef sub"),
            call("submit_work", summary="fixed add", files_changed=["calc/__init__.py"]),
            call("submit_work", summary="kept as is"),  # review-fix round
        ],
        "reviewer": [reject] * 3,
    }
    provider = ScriptedProvider("s", by_role=script)
    rt = open_runtime(repo, {"s": provider}, agent={"max_review_iterations": 1})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    result = await _run(rt, task.id)

    sub = result.state["subtasks"]["s1"]
    assert sub["review"]["verdict"] == "request_changes"
    assert sub["review_iterations"] == 2
    fix_requests = _requests(provider, "coder", "review_fix")
    assert len(fix_requests) == 1 and "float handling regressed" in fix_requests[0].messages[0].text()
    assert {g["name"]: g["status"] for g in sub["gates"]["results"]}["review"] == "FAILED"
    assert _gates(result.state)["review"] == "FAILED"
    assert result.status == TaskStatus.FAILED


# --------------------------------------------------------------------------- 12. small units


def _advisory_audit() -> CheckResult:
    advisory = Diagnostic(file="requirements.txt", code="PYSEC-2020-1", message="requests 2.0 has a known vulnerability")
    return CheckResult(kind=CheckKind.AUDIT, command="pip-audit", status="failed", diagnostics=[advisory], classification="vulnerabilities")


def test_security_gate_passes_for_preexisting_advisories_with_a_baseline() -> None:
    """repro_12 (a, unit): advisories present in the baseline audit do not fail the security gate."""
    audit = _advisory_audit()
    with_baseline = evaluate_gates(GatesSettings(), GateInputs(security_ran=True, audit=audit, audit_baseline=_advisory_audit()))
    assert with_baseline.get("security").status == GateStatus.PASSED
    assert "pre-existing advisories" in with_baseline.get("security").detail
    without = evaluate_gates(GatesSettings(), GateInputs(security_ran=True, audit=audit))
    assert without.get("security").status == GateStatus.FAILED


@POSIX_ONLY
async def test_pipeline_records_an_audit_baseline_and_excuses_preexisting_advisories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """repro_12 (a, pipeline): inspect records a baseline audit and final QA compares against it."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "pip-audit"  # an advisory that pre-dates the task, reported on every run
    fake.write_text(
        "#!/bin/sh\n"
        "echo 'Name     Version ID           Fix Versions'\n"
        "echo '-------- ------- ------------ ------------'\n"
        "echo 'requests 2.0.0   PYSEC-2020-1 2.20.0'\n"
        "exit 1\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    repo = make_calc_repo(tmp_path / "repo")
    script = {
        "classifier": [understanding()],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            FIX_ADD,
            call("read_file", path="pyproject.toml"),  # a manifest change makes final QA run the audit
            call("edit_file", path="pyproject.toml", old_string='version = "0.0.1"', new_string='version = "0.0.2"'),
            call("submit_work", summary="fixed add, bumped version", files_changed=["calc/__init__.py", "pyproject.toml"]),
        ],
        "reviewer": [APPROVE] * 3,
    }
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)}, validation={"dependency_audit": True})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    result = await _run(rt, task.id)

    baseline_audit = result.state["baseline"].get("audit")
    assert baseline_audit is not None, "inspect must record a baseline audit when an audit command exists"
    assert baseline_audit["status"] == "failed" and [d["code"] for d in baseline_audit["diagnostics"]] == ["PYSEC-2020-1"]
    security = next(g for g in result.state["gates"]["results"] if g["name"] == "security")
    assert security["status"] == "PASSED" and "pre-existing advisories" in security["detail"]
    assert result.status == TaskStatus.COMPLETED, result.error


def test_require_task_unique_prefix_of_a_parent_with_subtasks(tmp_path: Path) -> None:
    """repro_12 (b): subtask rows ("<id>:<sub>") do not make a parent's unique prefix ambiguous."""
    store = StateStore(tmp_path / "state.db")
    try:
        parent = store.create_task(Task(title="t", description="d"))
        for sid in ("s1", "s2"):
            child = Task(parent_id=parent.id, title=sid, description=sid)
            child.id = f"{parent.id}:{sid}"
            store.create_task(child)
        assert store.require_task(parent.id[:-3]).id == parent.id
        assert store.require_task(f"{parent.id}:s1").id == f"{parent.id}:s1"  # exact ids still work
    finally:
        store.close()


def test_require_task_prefix_treats_like_wildcards_literally(tmp_path: Path) -> None:
    """repro_12 (b): '%' and '_' in a prefix are not SQL LIKE wildcards."""
    store = StateStore(tmp_path / "state.db")
    try:
        only = store.create_task(Task(title="t", description="d"))
        assert store.require_task(only.id[:7]).id == only.id  # a real underscore ("task_...") still matches
        for wildcard in ("%", "_" * 8, "task%", only.id[:-2] + "__"):
            with pytest.raises(StateError):
                store.require_task(wildcard)
    finally:
        store.close()


def _question_script(coder_step: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "classifier": [understanding(task_type="question", summary="where is sub", acceptance_criteria=[])],
        "coder": [coder_step],
    }


async def test_stale_stop_request_does_not_stop_the_next_run(tmp_path: Path) -> None:
    """repro_12 (c): a stop request left over from an earlier run is dropped when a run starts."""
    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    answer = {"name": "submit_answer", "input": {"answer": "calc/__init__.py", "evidence": ["calc/__init__.py:8"]}}
    # long enough for the control watcher (2 s poll) to see a pending stop request
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=_question_script({"delay_s": 2.6, "tool_calls": [answer]}))})
    task = rt.create_task("Where is sub implemented?")
    # `aie tasks stop` was issued but the run it targeted ended before consuming it
    rt.store.update_task(task.id, status=TaskStatus.INTERRUPTED)
    rt.request_stop(task.id)
    try:
        result = await rt.orchestrator.run(task.id)
        leftover = rt.store.pop_control(task.id)
    finally:
        await rt.aclose()
    assert result.status == TaskStatus.COMPLETED, result.error
    assert leftover is None


async def test_cancelled_run_is_recorded_as_interrupted_and_releases_its_lease(tmp_path: Path) -> None:
    """repro_12 (d): asyncio cancellation leaves the task INTERRUPTED, unleased, and propagates."""
    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    provider = ScriptedProvider("s", by_role=_question_script({"delay_s": 60, "text": "..."}))
    rt = open_runtime(repo, {"s": provider})
    task = rt.create_task("Where is sub implemented?")
    try:
        run = asyncio.create_task(rt.orchestrator.run(task.id))
        deadline = time.monotonic() + 30
        while not _requests(provider, "coder") and not run.done() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert _requests(provider, "coder") and not run.done(), "the run should be waiting on the model"
        run.cancel()  # what a server does to a run that outlives its shutdown timeout
        with pytest.raises(asyncio.CancelledError):
            await run
        row = rt.store.require_task(task.id)
    finally:
        await rt.aclose()
    assert row.status == TaskStatus.INTERRUPTED
    assert row.lease_owner is None and row.lease_expires is None


# --------------------------------------------------------------------------- 13. blocking questions


async def test_blocked_question_keeps_triage_and_resumes_with_supplied_answers(tmp_path: Path) -> None:
    """repro_14: on_questions = "block" keeps the triage; answers supplied on resume unblock the task."""
    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    question = "Round half up or half to even?"
    triage = j({"summary": "Change rounding", "task_type": "change", "complexity": "small",
                "requirements": ["round results"], "blocking_questions": [question]})
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role={"classifier": [triage]})}, agent={"on_questions": "block"})
    task = rt.create_task("Make add() round its result")
    first = await _run(rt, task.id)
    assert first.status == TaskStatus.BLOCKED
    assert first.state["stage"] == "understand"
    assert question in (first.error or "")

    # what `aie resume <id> --answer "Round half up"` stores before resuming; the classifier would
    # ask the same question again if it were consulted
    provider = ScriptedProvider("s", by_role={"classifier": [triage]})
    rt2 = open_runtime(repo, {"s": provider}, agent={"on_questions": "block"})
    state = dict(rt2.store.require_task(task.id).state)
    state["supplied_answers"] = ["Round half up"]
    rt2.store.update_task(task.id, state=state)
    resumed = await _run(rt2, task.id, stop_after="plan")

    assert _requests(provider, "classifier") == [], "the triage must be kept, not redone"
    assert resumed.status != TaskStatus.BLOCKED, resumed.error
    assert resumed.state["stage"] not in ("understand", "inspect")
    assert resumed.state["clarifications"] == [{"question": question, "answer": "Round half up"}]
    assert any("Round half up" in r for r in resumed.state["understanding"]["requirements"])


# --------------------------------------------------------------------------- 14. daemon Ctrl+C


def _top_level_statuses(db: Path) -> dict[str, str]:
    con = sqlite3.connect(str(db), timeout=10)
    try:
        return dict(con.execute("SELECT title, status FROM tasks WHERE parent_id IS NULL").fetchall())
    finally:
        con.close()


@POSIX_ONLY
async def test_daemon_exits_on_ctrl_c_and_leaves_the_queue_alone(tmp_path: Path) -> None:
    """repro_07: one Ctrl+C stops the daemon after the current task; the next task stays QUEUED."""
    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    answer = {"name": "submit_answer", "input": {"answer": "calc/__init__.py", "evidence": ["calc/__init__.py:8"]}}
    script_file = tmp_path / "script.json"
    script_file.write_text(json.dumps({"by_role": _question_script({"delay_s": 8, "tool_calls": [answer]})}))
    (repo / ".agent").mkdir(exist_ok=True)
    (repo / ".agent" / "config.toml").write_text(
        f"[models.providers.s]\ntype = \"scripted\"\noptions = {{ script_file = {json.dumps(str(script_file))} }}\n"
        "[models.roles]\ndefault = [\"s:m\"]\n[validation]\ndependency_audit = false\n"
    )
    rt = Runtime.open(repo, use_global_config=False)
    try:
        rt.create_task("Where is sub implemented? (first)", priority=10, queue=True)
        rt.create_task("Where is sub implemented? (second)", priority=20, queue=True)
    finally:
        await rt.aclose()
    db = repo / ".agent" / "state.db"
    log_path = tmp_path / "daemon.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "ai_engineer.ui.cli", "-C", str(repo), "daemon", "--interval", "1", "-q"],
            stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ),
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and proc.poll() is None:
                if _top_level_statuses(db).get("Where is sub implemented? (first)") == "RUNNING":
                    break
                await asyncio.sleep(0.1)
            assert proc.poll() is None, log_path.read_text()
            assert _top_level_statuses(db)["Where is sub implemented? (first)"] == "RUNNING", log_path.read_text()
            await asyncio.sleep(1.0)
            proc.send_signal(signal.SIGINT)  # a single Ctrl+C
            try:
                await asyncio.to_thread(proc.wait, 30)
            except subprocess.TimeoutExpired:
                pytest.fail("the daemon kept running after Ctrl+C:\n" + log_path.read_text())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(10)
    statuses = _top_level_statuses(db)
    assert statuses["Where is sub implemented? (first)"] == "INTERRUPTED", (statuses, log_path.read_text())
    assert statuses["Where is sub implemented? (second)"] == "QUEUED", (statuses, log_path.read_text())


# --------------------------------------------------------------------------- 15. dead dependencies


def test_next_queued_blocks_tasks_whose_dependency_cannot_complete(tmp_path: Path) -> None:
    """A queued task depending on a FAILED or missing task is BLOCKED with the reason, not returned."""
    store = StateStore(tmp_path / "state.db")
    try:
        failed = store.create_task(Task(title="dep", description="dep", status=TaskStatus.FAILED))
        on_failed = store.create_task(Task(title="a", description="a", status=TaskStatus.QUEUED, priority=10, depends_on=[failed.id]))
        on_missing = store.create_task(Task(title="b", description="b", status=TaskStatus.QUEUED, priority=10, depends_on=["task_gone"]))
        assert store.next_queued() is None
        for task, dep in ((on_failed, failed.id), (on_missing, "task_gone")):
            row = store.require_task(task.id)
            assert row.status == TaskStatus.BLOCKED
            assert "dependency cannot complete" in (row.error or "") and dep in (row.error or "")
        ready = store.create_task(Task(title="c", description="c", status=TaskStatus.QUEUED, priority=20))
        assert store.next_queued().id == ready.id
    finally:
        store.close()


# --------------------------------------------------------------------------- 16. secrets in secret files


async def test_git_state_flags_secrets_written_into_a_changed_secret_file(tmp_path: Path) -> None:
    """Checkpoint diffs redact secret-file values, but the git_state gate still sees the secret."""
    repo = make_calc_repo(tmp_path / "repo")
    token = "ghp_" + "A1b2C3d4E5" * 3 + "F6g7H8"  # GitHub token shape: ghp_ + 36 alphanumerics
    assert len(token) == 40
    script = {
        "classifier": [understanding()],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            FIX_ADD,
            call("write_file", path="credentials.json", content=json.dumps({"github_token": token}) + "\n"),
            call("submit_work", summary="fixed add", files_changed=["calc/__init__.py", "credentials.json"]),
        ],
        "reviewer": [APPROVE] * 3,
    }
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.orchestrator.run(task.id)
        diff = await rt.checkpoints.diff_since(result.state["start_checkpoint"])
    finally:
        await rt.aclose()
    assert "credentials.json" in diff and token not in diff  # the premise: the diff is redacted
    git_state = next(g for g in result.state["gates"]["results"] if g["name"] == "git_state")
    assert git_state["status"] == "FAILED"
    assert "possible secrets" in git_state["detail"] and "github_token" in git_state["detail"]
    assert result.status == TaskStatus.FAILED
    assert token not in git(repo, "log", "-p", "--branches")  # never committed to a branch
