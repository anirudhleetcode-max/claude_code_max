"""End-to-end pipeline runs against real repositories with real test commands.

The model is scripted (deterministic), so these tests verify the *harness*: stage
routing, tool execution, validation, repair, review, gates, git, checkpoints,
recovery and reporting. They say nothing about the quality of any real model.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_engineer.core.events import EventType
from ai_engineer.providers.scripted import ScriptedProvider
from ai_engineer.tasks.store import TaskStatus
from tests.conftest import git
from tests.integration.helpers import APPROVE, call, make_calc_repo, open_runtime, understanding


def bugfix_script(first_fix: str = "return a * b") -> dict[str, list]:
    return {
        "classifier": [understanding()],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            call("edit_file", path="calc/__init__.py", old_string="return a - b\n\n\ndef sub", new_string=f"{first_fix}\n\n\ndef sub"),
            call("submit_work", summary="changed add", files_changed=["calc/__init__.py"]),
        ],
        "debugger": [
            call("read_file", path="calc/__init__.py"),
            call("run_tests", files=["tests/test_calc.py"]),
            call("edit_file", path="calc/__init__.py", old_string=first_fix, new_string="return a + b"),
            call("run_tests", files=["tests/test_calc.py"]),
            call("submit_work", summary="root cause: add used the wrong operator; now returns a + b", verification="pytest tests/test_calc.py: 2 passed"),
        ],
        "reviewer": [APPROVE],
    }


async def test_bugfix_with_repair_loop_review_commit_and_report(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    provider = ScriptedProvider("s", by_role=bugfix_script())
    rt = open_runtime(repo, {"s": provider})
    events = []
    rt.bus.subscribe(events.append)
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
    finally:
        await rt.aclose()

    assert result.status == TaskStatus.COMPLETED, (result.error, json.dumps(result.state.get("gates"), indent=1))
    assert "return a + b" in (repo / "calc" / "__init__.py").read_text()
    state = result.state
    # baseline captured the pre-existing failure, which was in scope and therefore had to be fixed
    assert state["baseline"]["test"]["status"] == "failed"
    sub = state["subtasks"]["s1"]
    assert sub["status"] == "completed" and sub["repair_iterations"] == 1
    assert sub["failures"][0]["result"] == "fixed"
    gates = {g["name"]: g["status"] for g in state["gates"]["results"]}
    assert gates["tests"] == "PASSED" and gates["review"] == "PASSED" and gates["security"] == "PASSED" and gates["git_state"] == "PASSED"
    assert gates["lint"] == "SKIPPED"  # no linter configured in this repo: reported, not faked
    # git: clean start -> task branch with a commit, user's main branch untouched
    branch = state["branch"]
    assert branch.startswith("aie/")
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == branch
    assert "aie: " in git(repo, "log", "-1", "--format=%s")
    assert "return a - b" in git(repo, "show", "main:calc/__init__.py")
    # observability
    types = {e.type for e in events}
    for expected in (EventType.TASK_STARTED, EventType.UNDERSTANDING_CREATED, EventType.PLAN_CREATED, EventType.TOOL_CALLED,
                     EventType.FILE_CHANGED, EventType.TEST_FAILED, EventType.FAILURE_RECORDED, EventType.FIX_ATTEMPTED,
                     EventType.TEST_PASSED, EventType.REVIEW_STARTED, EventType.CHECKPOINT_CREATED, EventType.COMMIT_CREATED,
                     EventType.GATES_EVALUATED, EventType.REPORT_CREATED, EventType.TASK_COMPLETED):
        assert expected in types, expected
    report = Path(state["report_path"]).read_text()
    for section in ("## Quality gates", "## Failures and fixes", "## Checkpoints", "## Metrics", "## Final validation"):
        assert section in report
    assert (repo / ".agent" / "reports" / f"{task.id}.json").exists()
    assert (repo / ".agent" / "tasks.json").exists()
    # the scripted model was consulted per role, and nothing was left unused
    assert provider.remaining() == 0
    # the agent's state never leaks into the repository
    assert ".agent" not in git(repo, "status", "--porcelain")


async def test_unfixed_bug_fails_gates_honestly(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    script = bugfix_script()
    # the debugger keeps making the same wrong change
    script["debugger"] = [call("submit_work", summary="I think it is fine now")] * 6
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)}, agent={"max_repair_iterations": 3})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    assert result.status == TaskStatus.FAILED
    gates = {g["name"]: g["status"] for g in result.state["gates"]["results"]}
    assert gates["tests"] == "FAILED"
    sub = result.state["subtasks"]["s1"]
    # the repeated identical failure was detected instead of looping forever
    assert sub["repair_iterations"] <= 3
    assert any(f["result"] in ("still_failing", "abandoned") for f in sub["failures"])
    # nothing was committed for a failed subtask
    assert "aie:" not in git(repo, "log", "--format=%s")


async def test_question_task_answers_with_evidence_without_changes(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    script = {
        "classifier": [understanding(task_type="question", summary="Where is subtraction implemented?", acceptance_criteria=[])],
        "coder": [
            call("search_text", pattern="def sub"),
            call("submit_answer", answer="`sub` is implemented in calc/__init__.py.", evidence=["calc/__init__.py:8"], confidence="high"),
        ],
    }
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)}, permissions={"mode": "safe"})
    task = rt.create_task("Where is subtraction implemented?")
    try:
        result = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    assert result.status == TaskStatus.COMPLETED
    assert result.state["answer"]["evidence"] == ["calc/__init__.py:8"]
    assert git(repo, "status", "--porcelain").strip() == ""


async def test_provider_failure_falls_back_to_next_model(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    broken = ScriptedProvider("primary", responder=lambda req: {"raise": {"type": "unavailable", "message": "connection refused"}})
    script = bugfix_script("return a + b")
    script.pop("debugger")  # a correct first fix needs no repair
    backup = ScriptedProvider("backup", by_role=script)
    rt = open_runtime(repo, {"primary": broken, "backup": backup}, roles={"default": ["primary:big", "backup:small"]})
    events = []
    rt.bus.subscribe(events.append)
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    assert result.status == TaskStatus.COMPLETED, result.error
    assert any(e.type == EventType.MODEL_FALLBACK for e in events)
    assert len(broken.requests) >= 2 and backup.remaining() == 0


async def test_all_models_down_blocks_then_resume_completes(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    down = ScriptedProvider("s", responder=lambda req: {"raise": {"type": "unavailable"}})
    rt = open_runtime(repo, {"s": down})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    # understanding falls back to heuristics; implementation cannot proceed without a model
    assert result.status == TaskStatus.BLOCKED
    assert "no model available" in (result.error or "")
    assert result.state["stage"] == "execute"
    # provider comes back: resume continues from the persisted stage
    script = bugfix_script("return a + b")
    script.pop("classifier")
    rt2 = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)})
    try:
        resumed = await rt2.run_task(task.id)
    finally:
        await rt2.aclose()
    assert resumed.status == TaskStatus.COMPLETED, resumed.error
    assert resumed.state["resumed"] == 1


async def test_interrupted_subtask_is_reverified_not_assumed(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    holder: dict = {}

    def coder(req):  # implements the fix, then the user stops the run before submit
        n = holder.setdefault("n", 0)
        holder["n"] = n + 1
        if n == 0:
            return call("read_file", path="calc/__init__.py")
        if n == 1:
            return call("edit_file", path="calc/__init__.py", old_string="return a - b\n\n\ndef sub", new_string="return a + b\n\n\ndef sub")
        holder["rt"].tool_ctx.cancel.cancel("stop requested (test)")
        return call("list_directory", path=".")

    script = {"classifier": [understanding()], "coder": [coder, coder, coder, coder]}
    provider = ScriptedProvider("s", by_role=script)
    rt = open_runtime(repo, {"s": provider})
    holder["rt"] = rt
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        first = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    assert first.status == TaskStatus.INTERRUPTED
    assert first.state["subtasks"]["s1"]["status"] == "running"
    # the edit happened on disk, but resuming must validate it rather than assume success
    events = []
    rt2 = open_runtime(repo, {"s": ScriptedProvider("s", by_role={"reviewer": [APPROVE]})})
    rt2.bus.subscribe(events.append)
    try:
        resumed = await rt2.run_task(task.id)
    finally:
        await rt2.aclose()
    assert resumed.status == TaskStatus.COMPLETED, resumed.error
    assert any("re-verifying" in e.message for e in events)
    assert any(e.type == EventType.TEST_PASSED for e in events)


async def test_read_only_checks_run_concurrently_and_writers_serialize(tmp_path: Path) -> None:
    import asyncio
    import time

    from ai_engineer.tester.models import CheckKind, CheckResult

    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    rt = open_runtime(repo, {"s": ScriptedProvider("s")})
    active = {"now": 0, "peak": 0}

    async def fake_run_check(kind, *, targeted_files=None, cancel=None):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.3)
        active["now"] -= 1
        return CheckResult(kind=kind, command=str(kind), status="passed", summary="ok")

    rt.validation.run_check = fake_run_check
    task = rt.create_task("x")
    try:
        started = time.monotonic()
        results = await rt.orchestrator._run_checks_parallel(task, None, [CheckKind.LINT, CheckKind.TYPECHECK])
        assert time.monotonic() - started < 0.55 and active["peak"] == 2
        assert [r.kind for r in results] == [CheckKind.LINT, CheckKind.TYPECHECK]
        active["peak"] = 0
        started = time.monotonic()
        await rt.orchestrator._run_checks_parallel(task, None, [CheckKind.TEST, CheckKind.BUILD])
        assert time.monotonic() - started >= 0.55 and active["peak"] == 1  # writers never overlap
    finally:
        await rt.aclose()
