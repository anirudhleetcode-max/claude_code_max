"""Review gate robustness (never PASS without a real model review) and the test-outcome taxonomy."""

from __future__ import annotations

import pytest

from ai_engineer.config.settings import GatesSettings
from ai_engineer.gates.evaluate import GateInputs, GateStatus, compare_with_baseline, evaluate_gates
from ai_engineer.providers.scripted import ScriptedProvider
from ai_engineer.reviewer.review import review_change
from ai_engineer.tester.models import CheckKind, CheckResult, TestCaseFailure
from tests.conftest import make_router

DIFF = """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""
GOOD = '{"verdict": "approve", "summary": "add() now adds", "issues": [], "requirements": [{"criterion": "add returns the sum", "status": "met", "evidence": "calc.py:2"}]}'


def review_gate(review) -> tuple[GateStatus, str]:
    report = evaluate_gates(GatesSettings(), GateInputs(requirements_understood=True, implementation_done=True, review=review))
    gate = next(g for g in report.results if g.name == "review")
    return gate.status, gate.detail


async def run_review(steps: list, criteria: list[str] | None = None, timeout_s: float | None = None):
    router = make_router(ScriptedProvider("s", steps=steps))
    model = router.for_role("reviewer")
    if timeout_s is not None:
        model.settings.retry.max_attempts = 1
    review, _ = await review_change(model, task="fix add", criteria=criteria if criteria is not None else ["add returns the sum"], diff=DIFF, validation_summary="tests passed")
    return review


async def test_a_real_model_review_passes() -> None:
    review = await run_review([GOOD])
    assert review.source == "model+deterministic"
    assert review_gate(review)[0] == GateStatus.PASSED


@pytest.mark.parametrize(
    ("label", "steps"),
    [
        ("model unavailable / network down", []),
        ("provider errors on every attempt", [{"raise": {"type": "unavailable"}}] * 10),
        ("context too long for the reviewer", [{"raise": {"type": "context"}}]),
        ("invalid request", [{"raise": {"type": "invalid"}}] * 3),
        ("malformed response", ["this is not json at all"] * 10),
        ("empty review", ['{"verdict": "approve", "summary": "", "issues": [], "requirements": []}']),
        ("criteria ignored", ['{"verdict": "approve", "summary": "looks fine", "issues": [], "requirements": []}']),
    ],
)
async def test_review_gate_never_passes_without_a_real_review(label: str, steps: list) -> None:
    review = await run_review(steps)
    status, detail = review_gate(review)
    assert status == GateStatus.UNVERIFIED, (label, review.source, detail)
    assert review.source in ("deterministic-only", "model-incomplete"), label


async def test_reviewer_timeout_is_unverified() -> None:
    from ai_engineer.core.errors import RetryableProviderError

    slow = ScriptedProvider("s", steps=[RetryableProviderError("request timed out")] * 10)
    model = make_router(slow).for_role("reviewer")
    review, _ = await review_change(model, task="fix add", criteria=[], diff=DIFF, validation_summary="")
    assert review.source == "deterministic-only" and review_gate(review)[0] == GateStatus.UNVERIFIED


async def test_missing_reviewer_configuration_is_unverified() -> None:
    model = make_router(ScriptedProvider("s"), roles={"default": ["nonexistent:model"]}).for_role("reviewer")
    review, _ = await review_change(model, task="fix add", criteria=[], diff=DIFF, validation_summary="")
    assert review.source == "deterministic-only" and review_gate(review)[0] == GateStatus.UNVERIFIED


async def test_no_reviewer_and_no_review_are_unverified() -> None:
    review, _ = await review_change(None, task="fix add", criteria=[], diff=DIFF, validation_summary="")
    assert review_gate(review)[0] == GateStatus.UNVERIFIED
    assert review_gate(None)[0] == GateStatus.UNVERIFIED


async def test_deterministic_blockers_fail_even_without_a_model() -> None:
    diff = DIFF.replace("+    return a + b", "+<<<<<<< HEAD\n+    return a + b")
    review, _ = await review_change(None, task="fix add", criteria=[], diff=diff, validation_summary="")
    assert review_gate(review)[0] == GateStatus.FAILED  # deterministic and model review stay separate


# ---- test-outcome taxonomy ----------------------------------------------------------------


def check(status: str, classification: str = "", failures: list[str] | None = None, **kw) -> CheckResult:
    return CheckResult(
        kind=CheckKind.TEST, command="pytest", status=status, classification=classification,
        failures=[TestCaseFailure(test_id=t, file=t.split("::")[0]) for t in failures or []], **kw,
    )


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        (check("passed"), "PASS"),
        (check("failed", "test_failure", ["tests/test_a.py::test_x"]), "FAIL"),
        (check("timeout", "timeout"), "FAIL"),
        (check("failed", "syntax_error"), "FAIL"),
        (check("error", "environment"), "ENVIRONMENT_ERROR"),
        (check("failed", "missing_dependency"), "ENVIRONMENT_ERROR"),
        (check("unavailable"), "UNAVAILABLE"),
        (check("error", "command_not_found"), "UNAVAILABLE"),
        (check("skipped"), "SKIPPED"),
        (check("cancelled"), "CANCELLED"),
        (check("failed", "no_tests"), "UNVERIFIED"),
    ],
)
def test_check_outcomes(result: CheckResult, outcome: str) -> None:
    assert result.outcome() == outcome


def gate_for_tests(current: CheckResult | None, baseline: CheckResult | None = None) -> GateStatus:
    report = evaluate_gates(GatesSettings(), GateInputs(checks={"tests": current}, baseline={"tests": baseline}))
    return next(g for g in report.results if g.name == "tests").status


def test_gate_statuses_for_the_taxonomy() -> None:
    failing = check("failed", "test_failure", ["tests/test_a.py::test_x"])
    assert gate_for_tests(check("passed")) == GateStatus.PASSED
    assert gate_for_tests(failing) == GateStatus.FAILED  # NEW failure (no baseline)
    assert gate_for_tests(failing, failing) == GateStatus.PREEXISTING  # same failure before the change
    other = check("failed", "test_failure", ["tests/test_b.py::test_y"])
    assert gate_for_tests(other, failing) == GateStatus.FAILED  # a different, new failure
    # unavailable tooling is never a false failure ...
    assert gate_for_tests(None) == GateStatus.UNVERIFIED
    assert gate_for_tests(check("unavailable")) == GateStatus.UNVERIFIED
    assert gate_for_tests(check("error", "command_not_found")) == GateStatus.UNVERIFIED
    assert gate_for_tests(check("failed", "no_tests")) == GateStatus.UNVERIFIED
    # ... and a real failure is never success
    assert gate_for_tests(check("timeout", "timeout"), failing) == GateStatus.FAILED
    env = check("error", "environment", output_tail="ModuleNotFoundError: No module named 'numpy'")
    assert gate_for_tests(env, env) == GateStatus.PREEXISTING  # broken before the change, identically
    assert gate_for_tests(env, check("passed")) == GateStatus.FAILED  # the change broke the environment


def test_compare_with_baseline_never_excuses_a_status_change() -> None:
    before = check("failed", "test_failure", ["tests/test_a.py::test_x"])
    for after in (check("timeout", "timeout"), check("error", "environment")):
        only_pre, new = compare_with_baseline(after, before)
        assert not only_pre and new and "status changed" in new[0]
