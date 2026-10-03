"""The benchmark must distinguish real work from claims (negative controls)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ai_engineer.benchmark.cases import ALL_CASES, BUGFIX, EXPLORE, SECURITY, approve, submit, understanding
from ai_engineer.benchmark.runner import run_case, summarize


async def test_oracle_bugfix_case_verifies(tmp_path: Path) -> None:
    result = await run_case(BUGFIX, "harness", tmp_path)
    assert result.completed and result.verified, result
    assert result.fix_attempts == 1 and result.recovered


async def test_negative_control_claims_without_changes_are_not_verified(tmp_path: Path) -> None:
    lazy = dataclasses.replace(BUGFIX, oracle=lambda: {
        "classifier": [understanding("Fix paginate()", ["page 2 is [4, 5, 6]"])],
        "coder": [submit("Fixed it, all tests pass.")],  # a false claim: no edit made
        "debugger": [submit("Definitely fixed now.")] * 6,
        "reviewer": [approve(["page 2 is [4, 5, 6]"])] * 3,
    })
    result = await run_case(lazy, "harness", tmp_path)
    assert result.status == "FAILED"
    assert not result.completed and not result.verified
    assert "hidden tests" in result.verification


async def test_negative_control_wrong_answer_is_not_verified(tmp_path: Path) -> None:
    wrong = dataclasses.replace(EXPLORE, oracle=lambda: {
        "classifier": [understanding("Find the tax computation", [], task_type="question")],
        "coder": [{"tool_calls": [{"name": "submit_answer", "input": {"answer": "It is in utils.py", "evidence": ["utils.py:1"]}}]}],
    })
    result = await run_case(wrong, "harness", tmp_path)
    assert not result.verified


async def test_insecure_fix_fails_security_gate(tmp_path: Path) -> None:
    from ai_engineer.benchmark.cases import read, write

    insecure = dataclasses.replace(SECURITY, oracle=lambda: {
        "classifier": [understanding("Fix SQL injection", ["injection blocked"], security=True)],
        "coder": [read("db.py"), write("db.py", 'import sqlite3\n\n\ndef find_user(con, name):\n    return con.execute("SELECT id, name FROM users WHERE name = \'%s\'" % name).fetchall()\n'), submit("escaped it"),
                  # review-fix rounds: the model insists without changing anything
                  submit("it is safe"), submit("it is safe"), submit("it is safe")],
        "reviewer": [approve(["injection blocked"])] * 4,
    })
    result = await run_case(insecure, "harness", tmp_path)
    assert not result.verified
    assert result.status == "FAILED"  # the model's approval cannot override the deterministic finding


def test_cases_cover_all_requested_categories() -> None:
    categories = {c.category for c in ALL_CASES}
    assert categories == {
        "repository exploration", "bug fixing", "feature implementation", "refactoring", "test generation",
        "debugging", "documentation", "dependency upgrade", "security remediation", "multi-file architecture change",
    }
    assert summarize([], "harness", None)["cases"] == 0
