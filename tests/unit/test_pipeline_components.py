"""Gates, reviewer, planner, debugger tracker and agent loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_engineer.config.settings import GateMode, GatesSettings, Settings
from ai_engineer.core.cancel import CancellationToken
from ai_engineer.core.errors import CancelledByUser
from ai_engineer.debugger.failures import FailureTracker
from ai_engineer.executor.control import CONTROL_TOOLS
from ai_engineer.executor.loop import AgentLoop, LoopLimits
from ai_engineer.gates.evaluate import GateInputs, GateStatus, compare_with_baseline, evaluate_gates
from ai_engineer.planner.models import Plan, Subtask, validate_plan
from ai_engineer.planner.planner import heuristic_understanding, make_plan, understand
from ai_engineer.providers.scripted import ScriptedProvider
from ai_engineer.reviewer.review import deterministic_review, review_change
from ai_engineer.tools.approval import AllowAllBroker
from ai_engineer.tools.executor import ToolExecutor, ToolRegistry
from ai_engineer.tools.factory import default_tools, make_context
from ai_engineer.tools.permissions import PermissionPolicy
from tests.conftest import make_router


def check(kind: str, status: str = "passed", failures=(), diagnostics=(), classification: str = "", summary: str = ""):
    sig = "|".join(sorted(f.test_id for f in failures)) or "|".join(sorted(d.message for d in diagnostics))
    return SimpleNamespace(
        kind=kind, status=status, failures=list(failures), diagnostics=list(diagnostics), classification=classification,
        summary=summary or f"{kind} {status}", output_tail="", command=f"run {kind}",
        ok=lambda: status == "passed", signature=lambda: "" if status == "passed" else sig,
    )


def tf(test_id: str, message: str = "boom"):
    return SimpleNamespace(test_id=test_id, file=None, line=None, message=message, failure_type="assertion")


# ---- gates ------------------------------------------------------------------------------------


def test_gates_all_pass() -> None:
    review = SimpleNamespace(blocking=lambda: [], requirements=[], issues=[], source="model+deterministic", summary="ok")
    report = evaluate_gates(GatesSettings(), GateInputs(
        checks={"tests": check("test"), "lint": check("lint"), "typecheck": check("typecheck"), "build": check("build")},
        security_ran=True, review=review, git_state=(True, "clean"),
    ))
    assert report.verdict == "COMPLETED", report.table()


def test_missing_required_check_is_unverified_not_passed() -> None:
    review = SimpleNamespace(blocking=lambda: [], requirements=[], issues=[], source="model", summary="ok")
    report = evaluate_gates(GatesSettings(), GateInputs(checks={"tests": None}, security_ran=True, review=review, git_state=(True, "ok")))
    assert report.get("tests").status == GateStatus.UNVERIFIED
    assert report.get("lint").status == GateStatus.SKIPPED  # if_available
    assert report.verdict == "COMPLETED_UNVERIFIED"


def test_no_tests_and_deterministic_only_review_are_unverified() -> None:
    review = SimpleNamespace(blocking=lambda: [], requirements=[], issues=[], source="deterministic-only", summary="model down")
    report = evaluate_gates(GatesSettings(), GateInputs(checks={"tests": check("test", "failed", classification="no_tests")}, security_ran=True, review=review, git_state=(True, "ok")))
    assert report.get("tests").status == GateStatus.UNVERIFIED
    assert report.get("review").status == GateStatus.UNVERIFIED
    assert report.verdict == "COMPLETED_UNVERIFIED"


def test_preexisting_failures_vs_new_failures() -> None:
    baseline = check("test", "failed", failures=[tf("t::old")])
    same = check("test", "failed", failures=[tf("t::old")])
    worse = check("test", "failed", failures=[tf("t::old"), tf("t::new")])
    assert compare_with_baseline(same, baseline) == (True, [])
    assert compare_with_baseline(worse, baseline) == (False, ["test:t::new"])
    review = SimpleNamespace(blocking=lambda: [], requirements=[], issues=[], source="model", summary="")
    r1 = evaluate_gates(GatesSettings(), GateInputs(checks={"tests": same}, baseline={"tests": baseline}, security_ran=True, review=review, git_state=(True, "")))
    assert r1.get("tests").status == GateStatus.PREEXISTING and r1.verdict == "COMPLETED_UNVERIFIED"
    r2 = evaluate_gates(GatesSettings(), GateInputs(checks={"tests": worse}, baseline={"tests": baseline}, security_ran=True, review=review, git_state=(True, "")))
    assert r2.get("tests").status == GateStatus.FAILED and r2.verdict == "FAILED"
    assert "test:t::new" in r2.get("tests").evidence


def test_security_and_review_failures_and_disabled_gates() -> None:
    finding = SimpleNamespace(severity="high", path="a.py", line=1, rule="py-eval", message="eval")
    review = SimpleNamespace(blocking=lambda: [SimpleNamespace(severity="blocker", description="bug")], requirements=[], issues=[SimpleNamespace(severity="blocker", description="bug")], source="model", summary="")
    gates = GatesSettings(tests=GateMode.DISABLED)
    report = evaluate_gates(gates, GateInputs(security_findings=[finding], security_ran=True, review=review, git_state=(False, "conflict markers")))
    assert report.get("tests").status == GateStatus.SKIPPED
    assert report.get("security").status == GateStatus.FAILED
    assert report.get("review").status == GateStatus.FAILED
    assert report.get("git_state").status == GateStatus.FAILED
    assert report.verdict == "FAILED"
    assert "FAILED" in report.checklist()


# ---- reviewer -----------------------------------------------------------------------------------


DIFF = """diff --git a/app/core.py b/app/core.py
--- a/app/core.py
+++ b/app/core.py
@@ -1,2 +1,5 @@
 def f(x):
+    print("debug", x)
+    result = eval(x)
+    # TODO handle errors
     return x
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,1 +1,3 @@
+@pytest.mark.skip(reason="flaky")
 def test_f():
"""


def test_deterministic_review_flags_real_problems() -> None:
    issues, findings = deterministic_review(DIFF, deleted_files=["tests/test_old.py"])
    kinds = {(i.severity, i.category) for i in issues}
    assert ("blocker", "security") in kinds  # eval
    assert ("minor", "maintainability") in kinds  # print
    assert ("nit", "maintainability") in kinds  # TODO
    assert any(i.category == "tests" and "skipped" in i.description for i in issues)
    assert any(i.category == "tests" and "deleted" in i.description for i in issues)
    assert findings and findings[0].rule == "py-eval"


async def test_model_review_combines_and_degrades() -> None:
    good = '{"verdict": "approve", "summary": "fine", "issues": [{"severity": "major", "category": "correctness", "description": "off by one"}], "requirements": [{"criterion": "c1", "status": "met"}]}'
    router = make_router(ScriptedProvider("r", steps=[good]))
    result, _ = await review_change(router.for_role("reviewer"), task="t", criteria=["c1"], diff="+++ b/x.py\n@@ -0,0 +1 @@\n+x = 1\n", validation_summary="ok")
    assert result.source == "model+deterministic"
    assert result.verdict == "request_changes"  # a major issue overrides the model's approve
    broken = make_router(ScriptedProvider("r", steps=["nope"] * 3))
    result2, _ = await review_change(broken.for_role("reviewer"), task="t", criteria=[], diff="+++ b/x.py\n@@ -0,0 +1 @@\n+x = 1\n", validation_summary="")
    assert result2.source == "deterministic-only"


# ---- planner ----------------------------------------------------------------------------------------


def test_validate_plan_orders_and_detects_problems() -> None:
    plan = Plan(goal="g", approach="a", subtasks=[
        Subtask(id="s2", title="b", description="b", depends_on=["s1"]),
        Subtask(id="s1", title="a", description="a"),
    ])
    ordered, problems = validate_plan(plan, 5)
    assert problems == [] and [s.id for s in ordered] == ["s1", "s2"]
    cyclic = Plan(goal="g", approach="a", subtasks=[
        Subtask(id="a", title="a", description="a", depends_on=["b"]),
        Subtask(id="b", title="b", description="b", depends_on=["a"]),
    ])
    assert any("cycle" in p for p in validate_plan(cyclic, 5)[1])
    assert validate_plan(Plan(goal="g", approach="a", subtasks=[Subtask(id="x", title="x", description="x", depends_on=["zz"])]), 5)[1]
    assert validate_plan(plan, 1)[1]


def test_heuristic_understanding_routes_reviews() -> None:
    u = heuristic_understanding("Add password reset tokens to the login API and document the endpoint")
    assert u.task_type == "change" and u.needs.security_review and u.needs.docs_update
    q = heuristic_understanding("Where is authentication handled?")
    assert q.task_type == "question" and not q.needs.tests


async def test_understand_falls_back_and_plan_repairs() -> None:
    router = make_router(ScriptedProvider("p", steps=["garbage"] * 3))
    u, source = await understand(router.for_role("classifier"), "Fix the crash in parser.py", "repo")
    assert source == "heuristic" and u.task_type == "change"
    bad_plan = '{"goal": "g", "approach": "a", "subtasks": [{"id": "s1", "title": "t", "description": "d", "depends_on": ["s9"]}]}'
    good_plan = '{"goal": "g", "approach": "a", "subtasks": [{"id": "s1", "title": "t", "description": "d"}, {"id": "s2", "title": "u", "description": "e", "depends_on": ["s1"]}]}'
    planner = ScriptedProvider("p", steps=[bad_plan, good_plan])
    u2 = u.model_copy(update={"complexity": "medium"})
    plan, ordered, source = await make_plan(make_router(planner).for_role("planner"), "task", u2, "brief", "ctx", 5)
    assert source == "model" and [s.id for s in ordered] == ["s1", "s2"]
    assert "invalid" in planner.requests[1].messages[-1].text()


# ---- debugger tracker ----------------------------------------------------------------------------


def test_failure_tracker_detects_loops_and_duplicate_fixes() -> None:
    tracker = FailureTracker("t", "s1", max_same_signature=3)
    failing = [check("test", "failed", failures=[tf("t::a", "assert 1 == 2")], classification="test_failure")]
    rec = tracker.record_failure(failing, "ctx")
    assert rec.likely_causes and rec.evidence and rec.signature
    assert tracker.record_fix(rec, "h1", "c1", "diff-1")
    assert tracker.record_outcome(rec, failing) == "still_failing"
    rec2 = tracker.record_failure(failing, "ctx")
    assert not tracker.record_fix(rec2, "h2", "c2", "diff-1")  # identical fix
    tracker.record_failure(failing, "ctx")
    assert "same failure" in (tracker.loop_detected() or "")
    assert "Attempt 1" in tracker.history_prompt()
    env = [check("test", "error", classification="command_not_found")]
    assert tracker.only_environmental(env)
    assert tracker.record_outcome(rec, [check("test")]) == "fixed"


# ---- agent loop -------------------------------------------------------------------------------------


def loop_harness(tmp_path: Path, steps, limits: LoopLimits | None = None):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    settings = Settings()
    ctx = make_context(ws, settings)
    registry = ToolRegistry([*default_tools(), *(c() for c in CONTROL_TOOLS)])
    executor = ToolExecutor(registry, PermissionPolicy(settings.permissions), AllowAllBroker())
    provider = ScriptedProvider("s", steps=steps)
    model = make_router(provider).for_role("coder")
    loop = AgentLoop(model, executor, ctx, role="coder", stage="implement", system="sys",
                     tools=["read_file", "write_file", "edit_file", "list_directory"], finish_tool="submit_work",
                     limits=limits or LoopLimits(max_steps=10))
    return ws, ctx, loop, provider


async def test_loop_executes_tools_and_finishes(tmp_path: Path) -> None:
    ws, ctx, loop, provider = loop_harness(tmp_path, [
        {"tool_calls": [{"name": "write_file", "input": {"path": "hello.py", "content": "print('hi')\n"}}]},
        {"text": "done", "tool_calls": [{"name": "submit_work", "input": {"summary": "created hello.py", "files_changed": ["hello.py"]}}]},
    ])
    result = await loop.run("create hello.py")
    assert result.status == "finished" and result.submission["summary"] == "created hello.py"
    assert (ws / "hello.py").exists() and result.tool_calls == 1
    # the second request contained the tool result
    assert provider.requests[1].messages[-1].tool_results()[0].content.startswith("created hello.py")


async def test_loop_invalid_submission_is_reported_back(tmp_path: Path) -> None:
    ws, ctx, loop, provider = loop_harness(tmp_path, [
        {"tool_calls": [{"name": "submit_work", "input": {"summary": "x"}}]},
        {"tool_calls": [{"name": "submit_work", "input": {"summary": "a proper summary"}}]},
    ])
    result = await loop.run("t")
    assert result.status == "finished"
    assert "invalid submit_work" in provider.requests[1].messages[-1].tool_results()[0].content


async def test_loop_nudges_then_gives_up_without_submit(tmp_path: Path) -> None:
    _, _, loop, provider = loop_harness(tmp_path, ["thinking", "still thinking", "final words"])
    result = await loop.run("t")
    assert result.status == "finished_no_submit" and result.final_text == "final words"
    assert "submit_work" in provider.requests[1].messages[-1].text()


async def test_loop_detects_repetition(tmp_path: Path) -> None:
    same = {"tool_calls": [{"name": "list_directory", "input": {"path": "."}}]}
    _, _, loop, provider = loop_harness(tmp_path, [same] * 12, LoopLimits(max_steps=12, repeated_call_limit=2))
    result = await loop.run("t")
    assert result.status in ("no_progress", "max_steps")
    joined = " ".join(m.text() for m in provider.requests[-1].messages)
    assert "Repeating it will not help" in joined


async def test_loop_step_budget_refusal_and_unavailable(tmp_path: Path) -> None:
    calls = [{"tool_calls": [{"name": "list_directory", "input": {"path": ".", "depth": d}}]} for d in (1, 2, 3, 4)]
    _, _, loop, _ = loop_harness(tmp_path, calls, LoopLimits(max_steps=3))
    assert (await loop.run("t")).status == "max_steps"
    _, _, loop, _ = loop_harness(tmp_path, [{"text": "", "stop_reason": "refusal"}])
    assert (await loop.run("t")).status == "refused"
    _, _, loop, _ = loop_harness(tmp_path, [{"raise": {"type": "auth"}}])
    assert (await loop.run("t")).status == "model_unavailable"


async def test_loop_context_reset_preserves_progress(tmp_path: Path) -> None:
    big = "x" * 4000
    steps = [{"text": f"note {i} {big}", "tool_calls": [{"name": "list_directory", "input": {"path": ".", "depth": 1 + i % 4}}]} for i in range(6)]
    steps.append({"tool_calls": [{"name": "submit_work", "input": {"summary": "finished after reset"}}]})
    _, ctx, loop, provider = loop_harness(tmp_path, steps, LoopLimits(max_steps=10, context_budget_tokens=3000))
    result = await loop.run("original task text")
    assert result.status == "finished" and result.context_resets >= 1
    reset_request = next(r for r in provider.requests if "Progress so far" in r.messages[0].text())
    assert reset_request.messages[0].text().startswith("original task text")
    assert len(reset_request.messages) == 1  # a fresh conversation, not an edited one


async def test_loop_context_length_error_triggers_reset(tmp_path: Path) -> None:
    steps = [
        {"tool_calls": [{"name": "list_directory", "input": {"path": "."}}]},
        {"raise": {"type": "context"}},
        {"tool_calls": [{"name": "submit_work", "input": {"summary": "ok after reset"}}]},
    ]
    _, _, loop, _ = loop_harness(tmp_path, steps)
    result = await loop.run("task")
    assert result.status == "finished" and result.context_resets == 1


async def test_loop_cancellation(tmp_path: Path) -> None:
    _, ctx, loop, _ = loop_harness(tmp_path, ["x"])
    ctx.cancel = CancellationToken()
    ctx.cancel.cancel("stop")
    with pytest.raises(CancelledByUser):
        await loop.run("t")


def test_baseline_failures_in_scope_are_not_excused() -> None:
    baseline = check("test", "failed", failures=[SimpleNamespace(test_id="tests/test_calc.py::test_add", file="tests/test_calc.py", line=3, message="x", failure_type="assertion")])
    current = check("test", "failed", failures=[SimpleNamespace(test_id="tests/test_calc.py::test_add", file="tests/test_calc.py", line=3, message="x", failure_type="assertion")])
    assert compare_with_baseline(current, baseline)[0] is True  # unrelated work: pre-existing
    only_pre, new = compare_with_baseline(current, baseline, {"tests/test_calc.py"})
    assert only_pre is False and new == ["test:tests/test_calc.py::test_add"]
