from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai_engineer.config.settings import Mode, PermissionLevel, Settings
from ai_engineer.core.types import ToolUseBlock
from ai_engineer.security.command_risk import Risk
from ai_engineer.tester.engine import ValidationEngine
from ai_engineer.tester.models import CheckKind
from ai_engineer.tools.approval import AllowAllBroker, DenyAllBroker
from ai_engineer.tools.base import SideEffect
from ai_engineer.tools.builtin.testing import (
    TESTING_TOOLS,
    RunBuildTool,
    RunFormatterTool,
    RunLinterTool,
    RunTestsTool,
    RunTypecheckTool,
)
from ai_engineer.tools.executor import ToolExecutor, ToolRegistry
from ai_engineer.tools.factory import builtin_tool_classes, make_context
from ai_engineer.tools.permissions import PermissionPolicy
from ai_engineer.tools.process import CommandOutcome

PYTEST_FAIL = """\
=================================== FAILURES ===================================
___________________________________ test_add ___________________________________

    def test_add():
>       assert add(1, 2) == 4
E       assert 3 == 4

tests/test_calc.py:5: AssertionError
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_add - assert 3 == 4
========================= 1 failed, 4 passed in 0.10s ==========================
"""


class FakeRunner:
    def __init__(self, outputs: dict[str, tuple[int, str]]) -> None:
        self.outputs = outputs
        self.commands: list[str] = []

    async def __call__(self, command, cwd, env, timeout_s, max_output_chars, cancel=None, shell=None) -> CommandOutcome:
        self.commands.append(command)
        exit_code, output = next((v for k, v in self.outputs.items() if k in command), (0, ""))
        return CommandOutcome(command=command, exit_code=exit_code, output=output, duration_s=0.4)


def setup(tmp_path: Path, outputs: dict[str, tuple[int, str]] | None = None, *, mode: Mode = Mode.DEVELOPER,
          validation: bool = True, profile: Any = None, broker: Any = None):
    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "src").mkdir()
    (ws / "tests" / "test_calc.py").write_text("def test_add():\n    assert 1\n")
    (ws / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (ws / "README.md").write_text("# demo\n")
    settings = Settings()
    settings.permissions.mode = mode
    profile = profile or SimpleNamespace(
        commands={
            "test": [SimpleNamespace(command="python -m pytest -q", source="pyproject.toml", confidence=0.9)],
            "lint": [SimpleNamespace(command="ruff check .", source="ruff", confidence=0.9)],
            "typecheck": [SimpleNamespace(command="mypy .", source="mypy", confidence=0.8)],
            "format": [SimpleNamespace(command="ruff format .", source="ruff", confidence=0.8)],
        },
        frameworks=["pytest"], package_managers=["pip"], primary_language="python", test_files=[], root=str(ws),
    )
    runner = FakeRunner(outputs or {})
    engine = ValidationEngine(ws, settings, profile, runner=runner, which=lambda name: f"/usr/bin/{name}")
    ctx = make_context(ws, settings, validation=engine if validation else None)
    executor = ToolExecutor(
        ToolRegistry([cls() for cls in TESTING_TOOLS]), PermissionPolicy(settings.permissions), broker or AllowAllBroker()
    )
    return ws, ctx, executor, runner, engine


def call(name: str, **args: Any) -> ToolUseBlock:
    return ToolUseBlock(id=f"call_{name}", name=name, input=args)


async def test_run_tests_reports_failures(tmp_path: Path) -> None:
    _, ctx, executor, runner, engine = setup(tmp_path, {"pytest": (1, PYTEST_FAIL)})
    block = await executor.execute(call("run_tests"), ctx)
    assert block.is_error
    assert "pytest: 1 failed, 4 passed" in block.content
    assert "tests/test_calc.py::test_add — assert 3 == 4 (tests/test_calc.py:5)" in block.content
    assert "classification: test_failure" in block.content
    assert "output (tail):" in block.content
    assert runner.commands == ["python -m pytest -q"]
    assert engine.history[-1].failed == 1


async def test_run_tests_result_data_and_ok(tmp_path: Path) -> None:
    _, ctx, _, runner, _ = setup(tmp_path, {"pytest": (0, "5 passed in 0.1s\n")})
    tool = RunTestsTool()
    result = await tool.run(RunTestsTool.Input(), ctx)
    assert result.ok and result.error == ""
    assert {"passed", "failed", "errors", "command", "status", "classification"} <= set(result.data)
    assert result.data["passed"] == 5 and result.data["status"] == "passed" and result.data["command"] == "python -m pytest -q"


async def test_run_tests_targeted_files_are_validated(tmp_path: Path) -> None:
    _, ctx, executor, runner, _ = setup(tmp_path, {"pytest": (0, "1 passed in 0.1s\n")})
    ok = await executor.execute(call("run_tests", files=["./tests/test_calc.py"]), ctx)
    assert not ok.is_error, ok.content
    assert runner.commands[-1] == "python -m pytest -q tests/test_calc.py"
    outside = await executor.execute(call("run_tests", files=["../escape.py"]), ctx)
    assert outside.is_error and "outside the workspace" in outside.content
    missing = await executor.execute(call("run_tests", files=["tests/test_nope.py"]), ctx)
    assert missing.is_error and "file not found" in missing.content
    assert len(runner.commands) == 1


async def test_assess_and_timeout(tmp_path: Path) -> None:
    _, ctx, _, _, engine = setup(tmp_path)
    tool = RunTestsTool()
    args = RunTestsTool.Input(files=["tests/test_calc.py"])
    a = tool.assess(args, ctx)
    assert a.level == PermissionLevel.DEVELOPMENT
    assert a.summary == "Running tests: python -m pytest -q tests/test_calc.py"
    assert a.command == "python -m pytest -q tests/test_calc.py"
    assert a.read_only is False and isinstance(a.risk, Risk)
    expected = engine.checks()[CheckKind.TEST].timeout_s + 30  # type: ignore[union-attr]
    assert tool.effective_timeout(args, ctx) == expected
    assert tool.effective_timeout(RunTestsTool.Input(timeout_s=10), ctx) == 40
    assert RunLinterTool().assess(RunLinterTool.Input(), ctx).summary == "Running linter: ruff check ."


async def test_validation_engine_missing(tmp_path: Path) -> None:
    _, ctx, executor, _, _ = setup(tmp_path, validation=False)
    for name in ("run_tests", "run_linter", "run_typecheck", "run_build", "run_formatter"):
        block = await executor.execute(call(name), ctx)
        assert block.is_error and "validation engine unavailable" in block.content


async def test_linter_typecheck_build(tmp_path: Path) -> None:
    outputs = {
        "ruff check": (1, "src/calc.py:1:1: F401 [*] `os` imported but unused\nFound 1 error.\n"),
        "mypy": (1, 'src/calc.py:2: error: Incompatible return value type (got "str", expected "int")  [return-value]\n'),
    }
    _, ctx, executor, runner, _ = setup(tmp_path, outputs)
    lint = await executor.execute(call("run_linter", files=["src/calc.py", "README.md"]), ctx)
    assert lint.is_error and "src/calc.py:1:1: `os` imported but unused [F401]" in lint.content
    assert runner.commands[-1] == "ruff check src/calc.py"
    tc = await executor.execute(call("run_typecheck"), ctx)
    assert tc.is_error and "[return-value]" in tc.content and "classification: type_errors" in tc.content
    build = await executor.execute(call("run_build"), ctx)
    assert build.is_error and "no build command detected" in build.content


async def test_formatter_check_and_write(tmp_path: Path) -> None:
    _, ctx, executor, runner, _ = setup(tmp_path, {"--check": (1, "Would reformat: src/calc.py\n1 file would be reformatted\n")})
    check = await executor.execute(call("run_formatter"), ctx)
    assert check.is_error and "src/calc.py" in check.content
    assert runner.commands[-1] == "ruff format --check ."
    write = await executor.execute(call("run_formatter", files=["src/calc.py"], check=False), ctx)
    assert not write.is_error, write.content
    assert runner.commands[-1] == "ruff format src/calc.py"
    assert "re-read" in write.content
    tool = RunFormatterTool()
    a = tool.assess(RunFormatterTool.Input(check=False), ctx)
    assert a.summary == "Formatting: ruff format ." and a.level == PermissionLevel.DEVELOPMENT
    assert RunFormatterTool.side_effect == SideEffect.WRITE
    no_target = await executor.execute(call("run_formatter", files=["README.md"], check=False), ctx)
    assert no_target.is_error and "cannot be limited" in no_target.content


async def test_formatter_never_rewrites_protected_files(tmp_path: Path) -> None:
    ws, ctx, executor, runner, _ = setup(tmp_path)
    (ws / ".env").write_text("X=1\n")
    block = await executor.execute(call("run_formatter", files=[".env"], check=False), ctx)
    assert block.is_error and "protected" in block.content
    assert runner.commands == []


async def test_policy_applies(tmp_path: Path) -> None:
    _, ctx, executor, runner, _ = setup(tmp_path, mode=Mode.SAFE)
    block = await executor.execute(call("run_tests"), ctx)
    assert block.is_error and "permission denied" in block.content
    assert runner.commands == []
    _, ctx2, executor2, runner2, _ = setup(tmp_path / "assisted", mode=Mode.ASSISTED, broker=DenyAllBroker())
    denied = await executor2.execute(call("run_linter"), ctx2)
    assert denied.is_error and "not approved" in denied.content and runner2.commands == []


def test_tool_registry_metadata() -> None:
    names = [cls.name for cls in TESTING_TOOLS]
    assert names == ["run_tests", "run_linter", "run_typecheck", "run_build", "run_formatter"]
    assert set(TESTING_TOOLS) <= set(builtin_tool_classes())
    for cls in TESTING_TOOLS:
        assert cls.level == PermissionLevel.DEVELOPMENT
        spec = cls().spec()
        assert spec.input_schema["type"] == "object"
    assert "files" in RunTestsTool().spec().input_schema["properties"]
    assert "files" not in RunBuildTool().spec().input_schema["properties"]
    assert "check" in RunFormatterTool().spec().input_schema["properties"]
    assert RunTypecheckTool.kind == CheckKind.TYPECHECK


@pytest.mark.parametrize("bad", [{"files": "tests/x.py"}, {"timeout_s": -1}, {"unknown": 1}])
async def test_invalid_arguments(tmp_path: Path, bad: dict[str, Any]) -> None:
    _, ctx, executor, runner, _ = setup(tmp_path)
    block = await executor.execute(ToolUseBlock(id="x", name="run_tests", input=bad), ctx)
    assert block.is_error and "invalid arguments" in block.content and runner.commands == []
