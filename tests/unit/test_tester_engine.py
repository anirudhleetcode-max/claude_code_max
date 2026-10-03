from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ai_engineer.config.settings import Settings
from ai_engineer.core.cancel import CancellationToken
from ai_engineer.tester.engine import MIN_OUTPUT_CHARS, ValidationEngine
from ai_engineer.tester.models import CheckKind, ValidationCommand
from ai_engineer.tools.process import CommandOutcome

PYTEST_FAIL = """\
FAILED tests/test_calc.py::test_add - assert 3 == 4
========================= 1 failed, 4 passed in 0.10s ==========================
"""
RUFF_FAIL = "src/a.py:1:8: F401 [*] `os` imported but unused\nFound 1 error.\n"


class FakeRunner:
    """Records calls and returns canned outcomes keyed by a substring of the command."""

    def __init__(self, outputs: dict[str, tuple[int | None, str]] | None = None, **flags: Any) -> None:
        self.outputs = outputs or {}
        self.flags = flags
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, command, cwd, env, timeout_s, max_output_chars, cancel=None, shell=None) -> CommandOutcome:
        self.calls.append({
            "command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s,
            "max_output_chars": max_output_chars, "cancel": cancel, "shell": shell,
        })
        if self.flags.get("raise_os"):
            raise FileNotFoundError("no such shell")
        exit_code, output = next((v for k, v in self.outputs.items() if k in command), (0, ""))
        return CommandOutcome(
            command=command, exit_code=exit_code, output=output, duration_s=1.5,
            timed_out=bool(self.flags.get("timed_out")), cancelled=bool(self.flags.get("cancelled")),
        )


def make_profile(**commands: str) -> SimpleNamespace:
    return SimpleNamespace(
        commands={k: [SimpleNamespace(command=v, source="discovery", confidence=0.9)] for k, v in commands.items()},
        frameworks=["pytest"], package_managers=[], primary_language="python", test_files=[], root=None,
    )


def which_all(name: str) -> str | None:
    return f"/usr/bin/{name}"


def engine(tmp_path: Path, runner: FakeRunner, profile: Any = None, settings: Settings | None = None, which=which_all) -> ValidationEngine:
    if profile is None:
        profile = make_profile(test="python -m pytest -q", lint="ruff check .", typecheck="mypy .")
    return ValidationEngine(tmp_path, settings or Settings(), profile, runner=runner, which=which)


async def test_run_check_parses_and_records_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    runner = FakeRunner({"pytest": (1, PYTEST_FAIL)})
    settings = Settings()
    settings.validation.test_timeout_s = 77
    eng = engine(tmp_path, runner, settings=settings)
    result = await eng.run_check(CheckKind.TEST)
    assert result.status == "failed" and result.classification == "test_failure"
    assert result.failed == 1 and result.passed == 4
    assert result.duration_s == 1.5
    assert result.summary == "pytest: 1 failed, 4 passed (1.5s)"
    assert eng.history == [result] and eng.latest(CheckKind.TEST) is result
    (call,) = runner.calls
    assert call["command"] == "python -m pytest -q"
    assert call["cwd"] == tmp_path.resolve()
    assert call["timeout_s"] == 77
    assert call["max_output_chars"] == max(settings.terminal.max_output_chars, MIN_OUTPUT_CHARS)
    assert "ANTHROPIC_API_KEY" not in call["env"] and call["env"]["CI"] == "true"


async def test_checks_are_cached_and_refreshable(tmp_path: Path) -> None:
    eng = engine(tmp_path, FakeRunner())
    first = eng.checks()
    assert first[CheckKind.LINT].command == "ruff check ."  # type: ignore[union-attr]
    assert eng.checks() == first
    eng.refresh_checks(make_profile(lint="flake8"))
    assert eng.checks()[CheckKind.LINT].command == "flake8"  # type: ignore[union-attr]
    eng.refresh_checks(None)
    assert eng.checks()[CheckKind.TEST] is None


async def test_no_command_and_unavailable_command(tmp_path: Path) -> None:
    runner = FakeRunner()
    eng = engine(tmp_path, runner, which=lambda name: None if name == "mypy" else f"/bin/{name}")
    none = await eng.run_check(CheckKind.BUILD)
    assert none.status == "unavailable" and none.classification == "" and none.summary == "no build command detected"
    missing = await eng.run_check(CheckKind.TYPECHECK)
    assert missing.status == "unavailable" and missing.classification == "command_not_found"
    assert "mypy not found on PATH" in missing.summary
    assert runner.calls == []
    assert [r.kind for r in eng.history] == [CheckKind.BUILD, CheckKind.TYPECHECK]


async def test_targeted_files_use_targeted_commands(tmp_path: Path) -> None:
    runner = FakeRunner({"pytest": (0, "2 passed in 0.1s\n"), "ruff": (1, RUFF_FAIL)})
    eng = engine(tmp_path, runner)
    t = await eng.run_check(CheckKind.TEST, targeted_files=["tests/test_a.py"])
    assert t.ok() and runner.calls[-1]["command"] == "python -m pytest -q tests/test_a.py"
    lint = await eng.run_check(CheckKind.LINT, targeted_files=["src/a.py", "README.md"])
    assert runner.calls[-1]["command"] == "ruff check src/a.py"
    assert lint.diagnostics[0].code == "F401"
    # not derivable -> full command
    await eng.run_check(CheckKind.LINT, targeted_files=["README.md"])
    assert runner.calls[-1]["command"] == "ruff check ."
    assert eng.command_for(CheckKind.TEST, ["tests/test_b.py"]).scope == "targeted"  # type: ignore[union-attr]


async def test_run_many_is_sequential_in_order(tmp_path: Path) -> None:
    runner = FakeRunner({"pytest": (0, "3 passed in 0.1s\n"), "ruff": (1, RUFF_FAIL), "mypy": (0, "Success: no issues found in 3 source files\n")})
    eng = engine(tmp_path, runner)
    kinds = [CheckKind.TYPECHECK, CheckKind.LINT, CheckKind.TEST, CheckKind.BUILD]
    results = await eng.run_many(kinds)
    assert [r.kind for r in results] == kinds
    assert [r.status for r in results] == ["passed", "failed", "passed", "unavailable"]
    assert [c["command"] for c in runner.calls] == ["mypy .", "ruff check .", "python -m pytest -q"]
    assert len(eng.history) == 4


async def test_timeout_cancel_and_spawn_errors(tmp_path: Path) -> None:
    timed = await engine(tmp_path, FakeRunner({"pytest": (None, "...\n[command timed out after 5s and was terminated]")}, timed_out=True)).run_check(CheckKind.TEST)
    assert timed.status == "timeout" and timed.classification == "timeout"
    cancelled = await engine(tmp_path, FakeRunner({"pytest": (None, "..")}, cancelled=True)).run_check(CheckKind.TEST)
    assert cancelled.status == "cancelled" and not cancelled.ok()
    token = CancellationToken()
    token.cancel("stop")
    runner = FakeRunner()
    pre = await engine(tmp_path, runner).run_many([CheckKind.TEST, CheckKind.LINT], cancel=token)
    assert [r.status for r in pre] == ["cancelled", "cancelled"] and runner.calls == []
    broken = await engine(tmp_path, FakeRunner(raise_os=True)).run_check(CheckKind.TEST)
    assert broken.status == "error" and broken.classification == "environment"


async def test_cancel_token_is_passed_to_runner(tmp_path: Path) -> None:
    runner = FakeRunner()
    token = CancellationToken()
    await engine(tmp_path, runner).run_check(CheckKind.TEST, cancel=token)
    assert runner.calls[0]["cancel"] is token


async def test_missing_working_directory(tmp_path: Path) -> None:
    runner = FakeRunner()
    eng = engine(tmp_path, runner)
    vc = ValidationCommand(kind=CheckKind.TEST, command="pytest", source="x", cwd="nope")
    r = await eng.run_command(vc)
    assert r.status == "error" and r.classification == "environment" and runner.calls == []


async def test_docker_sandbox_wraps_command_and_ignores_host_path(tmp_path: Path) -> None:
    settings = Settings()
    settings.terminal.sandbox = "docker"
    runner = FakeRunner({"pytest": (0, "1 passed in 0.1s\n")})
    eng = engine(tmp_path, runner, settings=settings, which=lambda _: None)
    assert eng.checks()[CheckKind.TEST].available  # type: ignore[union-attr]
    r = await eng.run_check(CheckKind.TEST)
    cmd = runner.calls[0]["command"]
    assert cmd.startswith("docker run") and "python -m pytest -q" in cmd
    assert r.command == "python -m pytest -q" and r.ok()


async def test_absolute_paths_are_made_workspace_relative(tmp_path: Path) -> None:
    ws = tmp_path.resolve()
    out = f"{ws}/src/a.py:3: error: Name \"x\" is not defined  [name-defined]\n/elsewhere/lib.py:1: error: boom  [misc]\n"
    eng = engine(tmp_path, FakeRunner({"mypy": (1, out)}))
    r = await eng.run_check(CheckKind.TYPECHECK)
    assert [d.file for d in r.diagnostics] == ["src/a.py", "/elsewhere/lib.py"]


async def test_subdirectory_cwd_paths_are_rebased(tmp_path: Path) -> None:
    (tmp_path / "web").mkdir()
    eng = engine(tmp_path, FakeRunner({"eslint": (1, "src/a.js:1:1: 'x' is not defined [Error/no-undef]\n")}))
    vc = ValidationCommand(kind=CheckKind.LINT, command="npx eslint -f unix .", source="x", cwd="web")
    r = await eng.run_command(vc)
    assert r.diagnostics[0].file == "web/src/a.js"


async def test_format_write_command(tmp_path: Path) -> None:
    prof = make_profile(format="ruff format .")
    eng = engine(tmp_path, FakeRunner(), profile=prof)
    assert eng.checks()[CheckKind.FORMAT].command == "ruff format --check ."  # type: ignore[union-attr]
    w = eng.format_write_command(["src/a.py"])
    assert w is not None and w.command == "ruff format src/a.py" and w.scope == "targeted"
    assert eng.format_write_command().command == "ruff format ."  # type: ignore[union-attr]


# ------------------------------------------------------------------------------------- end to end


def _write_project(root: Path) -> None:
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    (tests / "test_bad.py").write_text(
        "from calc import add\n\n\ndef test_add_wrong():\n    assert add(2, 2) == 5\n\n\ndef test_fine():\n    assert True\n"
    )


async def test_end_to_end_real_pytest(tmp_path: Path) -> None:
    _write_project(tmp_path)
    settings = Settings()
    python = shlex.quote(sys.executable) if os.name != "nt" else f'"{sys.executable}"'
    settings.validation.test_command = f"{python} -m pytest -q -p no:cacheprovider"
    settings.validation.test_timeout_s = 120
    eng = ValidationEngine(tmp_path, settings, None)
    full = await eng.run_check(CheckKind.TEST)
    assert full.status == "failed", full.output_tail
    assert full.classification == "test_failure"
    assert (full.passed, full.failed) == (2, 1)
    (failure,) = full.failures
    assert failure.test_id == "tests/test_bad.py::test_add_wrong"
    assert (failure.file, failure.line, failure.failure_type) == ("tests/test_bad.py", 5, "assertion")
    assert full.signature()

    again = await eng.run_check(CheckKind.TEST)
    assert again.signature() == full.signature()

    targeted = await eng.run_check(CheckKind.TEST, targeted_files=["tests/test_ok.py"])
    assert targeted.ok(), targeted.output_tail
    assert targeted.passed == 1
    assert targeted.command.endswith("tests/test_ok.py")
    assert len(eng.history) == 3


def test_windows_style_relative_paths_are_normalized(tmp_path: Path) -> None:
    from ai_engineer.tester.models import CheckResult, Diagnostic, TestCaseFailure

    eng = ValidationEngine(tmp_path, Settings(), None)
    result = CheckResult(
        kind=CheckKind.TEST, command="pytest", status="failed",
        failures=[TestCaseFailure(test_id="tests/test_x.py::t", file="tests\\test_x.py", line=3)],
        diagnostics=[Diagnostic(file="src\\pkg\\a.py", line=1, message="m")],
    )
    eng._relativize(result, tmp_path)
    assert result.failures[0].file == "tests/test_x.py"
    assert result.diagnostics[0].file == "src/pkg/a.py"
