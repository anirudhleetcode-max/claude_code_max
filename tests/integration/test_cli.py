"""The `aie` command line, exercised through `main()` against real temporary projects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ai_engineer.ui.cli import main
from tests.conftest import git
from tests.integration.helpers import j, make_calc_repo

FAKE_KEY = "sk-" + "Ab3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3fG5hJ7kL9"


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out + out.err


def scripted_config(repo: Path, by_role: dict) -> None:
    script = repo / ".agent" / "script.json"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(json.dumps({"by_role": by_role}), encoding="utf-8")
    (repo / ".agent" / "config.toml").write_text(
        f'[models.providers.s]\ntype = "scripted"\noptions = {{ script_file = "{script.as_posix()}" }}\n\n'
        '[models.roles]\ndefault = ["s:m"]\n',
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AIE_MODEL", "AIE_REVIEW_MODEL", "AIE_FAST_MODEL", "AIE_MODE", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_doctor_reports_without_side_effects_or_secrets(tmp_path: Path, capsys, monkeypatch) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("AIE_MODEL", "openai:test-model")
    code, out = run(capsys, "-C", str(repo), "doctor", "--offline", "--json")
    checks = json.loads(out)
    by_name = {c["name"]: c for c in checks}
    assert FAKE_KEY not in out
    assert by_name["openai credentials"]["status"] == "ok" and "OPENAI_API_KEY is set" in by_name["openai credentials"]["detail"]
    assert by_name["python"]["status"] == "ok" and by_name["git repository"]["status"] == "ok"
    assert {c["section"] for c in checks} >= {"runtime", "dependencies", "tools", "database", "workspace", "security", "providers"}
    assert not (repo / ".agent").exists()  # diagnosing never initialises the project
    assert code == 0
    monkeypatch.delenv("OPENAI_API_KEY")
    code, out = run(capsys, "-C", str(repo), "doctor", "--offline")
    assert code == 1 and "set OPENAI_API_KEY" in out and "problem(s) found" in out


def test_doctor_flags_missing_model_configuration(tmp_path: Path, capsys) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    code, out = run(capsys, "-C", str(repo), "doctor", "--offline")
    assert code == 1 and "no model configured" in out


def pin_test_command(repo: Path) -> None:
    """Run the project's tests with this interpreter (the one that has pytest installed)."""
    (repo / ".agent").mkdir(exist_ok=True)
    command = f'"{Path(sys.executable).as_posix()}" -m pytest -q -p no:cacheprovider'
    (repo / ".agent" / "config.toml").write_text(f"[validation]\ntest_command = '{command}'\n")


def test_test_command_reports_outcomes_and_exit_codes(tmp_path: Path, capsys) -> None:
    repo = make_calc_repo(tmp_path / "repo")  # add() is buggy: the test fails
    pin_test_command(repo)
    code, out = run(capsys, "-C", str(repo), "test", "--kind", "test")
    assert code == 1 and out.startswith("FAIL") and "1 failed" in out
    code, out = run(capsys, "-C", str(repo), "test", "--kind", "lint")
    assert code == 2 and "UNAVAILABLE" in out  # missing tooling is never reported as a failure
    fixed = make_calc_repo(tmp_path / "fixed", buggy=False)
    pin_test_command(fixed)
    code, out = run(capsys, "-C", str(fixed), "test", "--kind", "test", "--json")
    assert code == 0 and out.startswith("PASS")


def test_review_without_model_is_unverified(tmp_path: Path, capsys) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    code, out = run(capsys, "-C", str(repo), "review", "--no-model")
    assert code == 0 and "Nothing to review" in out
    (repo / "calc" / "__init__.py").write_text((repo / "calc" / "__init__.py").read_text().replace("a - b", "a + b"))
    code, out = run(capsys, "-C", str(repo), "review", "--no-model")
    assert code == 2 and "UNVERIFIED" in out


def test_review_with_model_and_with_unavailable_model(tmp_path: Path, capsys) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    (repo / "calc" / "__init__.py").write_text((repo / "calc" / "__init__.py").read_text().replace("a - b", "a + b"))
    approve = j({"verdict": "approve", "summary": "add() now adds", "issues": [], "requirements": []})
    scripted_config(repo, {"reviewer": [approve]})
    code, out = run(capsys, "-C", str(repo), "review", "--task", "fix add")
    assert code == 0 and "approve" in out and "UNVERIFIED" not in out
    scripted_config(repo, {"reviewer": []})  # the reviewer model fails: never a pass
    code, out = run(capsys, "-C", str(repo), "review", "--task", "fix add")
    assert code == 2 and "UNVERIFIED" in out


def test_project_show_and_invalidate(tmp_path: Path, capsys) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    assert run(capsys, "-C", str(repo), "init")[0] == 0
    code, out = run(capsys, "-C", str(repo), "memory", "add", "The", "calc", "package", "is", "pure", "python")
    assert code == 0
    item_id = out.split()[-1]
    code, out = run(capsys, "-C", str(repo), "project")
    assert code == 0 and "Primary language: python" in out and "test: " in out and "Memory: 1 project item(s)" in out
    assert (repo / ".agent" / "indexes" / "index.db").exists()
    code, out = run(capsys, "-C", str(repo), "project", "invalidate")
    assert code == 0 and "code index" in out and "1 project memory item(s)" in out
    assert not (repo / ".agent" / "indexes" / "index.db").exists() and not (repo / ".agent" / "project.json").exists()
    code, out = run(capsys, "-C", str(repo), "memory", "search", "calc")
    assert item_id in out and "STALE" in out and "(invalidated)" in out
    code, out = run(capsys, "-C", str(repo), "memory", "update", item_id, "calc", "is", "pure", "python", "3.11+")
    assert code == 0 and "v2" in out
    code, out = run(capsys, "-C", str(repo), "memory", "search", "calc")
    assert "3.11+" in out and "STALE" not in out  # re-asserted content is fresh again


def test_logs_restore_and_aliases(tmp_path: Path, capsys) -> None:
    from ai_engineer.core.events import EventType
    from ai_engineer.runtime import Runtime

    repo = make_calc_repo(tmp_path / "repo")
    code, out = run(capsys, "-C", str(repo), "logs")
    assert code == 1 and "No tasks yet" in out
    rt = Runtime.open(repo, use_global_config=False)
    task = rt.create_task("demo task")
    for i in range(5):
        rt.bus.emit(EventType.TEST_FAILED if i == 3 else EventType.INFO, f"step {i}", task_id=task.id)
    import asyncio

    cp = asyncio.run(rt.checkpoints.create("before edit", task_id=task.id))
    asyncio.run(rt.aclose())
    code, out = run(capsys, "-C", str(repo), "logs")
    lines = [line for line in out.splitlines() if "step" in line]
    assert code == 0 and [line.split()[-1] for line in lines] == ["0", "1", "2", "3", "4"]
    code, out = run(capsys, "-C", str(repo), "logs", task.id, "--type", "test_failed")
    assert code == 0 and "step 3" in out and "step 2" not in out
    (repo / "calc" / "__init__.py").write_text("broken = True\n")
    code, out = run(capsys, "-C", str(repo), "restore", cp.id)
    assert code == 0 and "Undo with" in out and "return a - b" in (repo / "calc" / "__init__.py").read_text()
    code, out = run(capsys, "-C", str(repo), "checkpoint", "list")
    assert code == 0 and cp.id in out
    code, out = run(capsys, "-C", str(repo), "task", "list")
    assert code == 0 and task.id in out
    assert git(repo, "status", "--porcelain").strip() == ""  # restore left no stray changes
