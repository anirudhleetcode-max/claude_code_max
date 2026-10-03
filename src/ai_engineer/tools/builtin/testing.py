"""Validation tools: run the project's tests, linters, formatters, type checker and build.

The concrete commands come from the :class:`~ai_engineer.tester.engine.ValidationEngine`
attached to the tool context (``ctx.validation``); results are parsed into structured
failures so the model sees *what* failed, not just raw output.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from ...config.settings import PermissionLevel
from ...core.errors import ToolError
from ...security.command_risk import classify_command
from ...tester.models import CheckKind, CheckResult, ValidationCommand
from ..base import ActionAssessment, SideEffect, Tool, ToolContext, ToolInput, ToolResult

MAX_LISTED = 20
TAIL_CHARS = 4000
PASSING_TAIL_CHARS = 1000
TIMEOUT_MARGIN_S = 30.0

_FILES_FIELD = "Workspace-relative files (or directories) to restrict the run to; empty = whole project"


def _engine(ctx: ToolContext) -> Any:
    if ctx.validation is None:
        raise ToolError("validation engine unavailable")
    return ctx.validation


def _resolve_files(ctx: ToolContext, files: list[str]) -> list[str]:
    """Validate paths against the workspace guard; return workspace-relative POSIX paths."""
    out: list[str] = []
    for raw in files:
        path = ctx.guard.resolve(raw)
        if not path.exists():
            raise ToolError(f"file not found: {raw}")
        rel = ctx.guard.relative(path)
        if rel not in out:
            out.append(rel)
    return out


def _location(file: str | None, line: int | None, column: int | None = None) -> str:
    if not file:
        return ""
    return file + (f":{line}" if line else "") + (f":{column}" if line and column else "")


def render_result(result: CheckResult) -> str:
    """Compact, model-facing rendering of a check result."""
    lines = [result.summary or f"{result.kind}: {result.status}"]
    if result.command:
        lines.append(f"command: {result.command}")
    if not result.ok() and result.classification:
        lines.append(f"classification: {result.classification}")
    if result.failures:
        lines.append("failures:")
        for f in result.failures[:MAX_LISTED]:
            msg = f.message.strip().split("\n")[0][:300]
            loc = _location(f.file, f.line)
            lines.append(f"- {f.test_id} — {msg}" + (f" ({loc})" if loc else ""))
        if len(result.failures) > MAX_LISTED:
            lines.append(f"... and {len(result.failures) - MAX_LISTED} more failures")
    if result.diagnostics:
        lines.append("diagnostics:")
        for d in result.diagnostics[:MAX_LISTED]:
            loc = _location(d.file, d.line, d.column)
            code = f" [{d.code}]" if d.code else ""
            sev = " (warning)" if d.severity == "warning" else ""
            lines.append(f"- {loc + ': ' if loc else ''}{d.message.strip()[:300]}{code}{sev}")
        if len(result.diagnostics) > MAX_LISTED:
            lines.append(f"... and {len(result.diagnostics) - MAX_LISTED} more diagnostics")
    tail = result.output_tail.strip()
    if tail:
        limit = PASSING_TAIL_CHARS if result.ok() else TAIL_CHARS
        lines.append("output (tail):")
        lines.append(tail[-limit:])
    return "\n".join(lines)


def to_tool_result(result: CheckResult) -> ToolResult:
    return ToolResult(
        ok=result.ok(),
        content=render_result(result),
        error="" if result.ok() else (result.summary or result.status),
        duration_s=result.duration_s,
        data={
            "kind": str(result.kind),
            "status": result.status,
            "classification": result.classification,
            "command": result.command,
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "skipped": result.skipped,
            "exit_code": result.exit_code,
        },
    )


class _CheckTool(Tool):
    """Shared behaviour: resolve the concrete command, assess it, run it through the engine."""

    kind: ClassVar[CheckKind]
    verb: ClassVar[str]
    level = PermissionLevel.DEVELOPMENT
    side_effect = SideEffect.EXECUTE
    timeout_s = 1800.0
    tags = frozenset({"validation"})

    def command(self, args: Any, ctx: ToolContext) -> ValidationCommand | None:
        engine = _engine(ctx)
        files = _resolve_files(ctx, list(getattr(args, "files", None) or []))
        vc: ValidationCommand | None = engine.command_for(self.kind, files or None)
        timeout = getattr(args, "timeout_s", None)
        if vc is not None and timeout:
            vc = vc.model_copy(update={"timeout_s": float(timeout)})
        return vc

    def assess(self, args: Any, ctx: ToolContext) -> ActionAssessment:
        vc = self.command(args, ctx)
        if vc is None:
            return ActionAssessment(
                level=PermissionLevel.DEVELOPMENT, summary=f"{self.verb}: no {self.kind} command detected", read_only=False
            )
        assessment = classify_command(vc.command, ctx.workspace)
        return ActionAssessment(
            level=PermissionLevel.DEVELOPMENT,
            summary=f"{self.verb}: {vc.command}",
            risk=assessment.risk,
            read_only=False,
            command=vc.command,
            details={"risk_reasons": assessment.reasons, "kind": str(self.kind), "scope": vc.scope},
        )

    def effective_timeout(self, args: Any, ctx: ToolContext) -> float:
        try:
            vc = self.command(args, ctx)
        except ToolError:
            return self.timeout_s
        return (vc.timeout_s if vc is not None else 60.0) + TIMEOUT_MARGIN_S

    async def run(self, args: Any, ctx: ToolContext) -> ToolResult:
        engine = _engine(ctx)
        vc = self.command(args, ctx)
        if vc is None:
            result: CheckResult = await engine.run_check(self.kind, cancel=ctx.cancel)
            return ToolResult.fail(
                f"{result.summary}. Configure validation.{self.kind}_command or use run_command.",
                kind=str(self.kind), status=result.status,
            )
        return to_tool_result(await engine.run_command(vc, ctx.cancel))


class RunTestsInput(ToolInput):
    files: list[str] = Field(default_factory=list, description="Test files to run; empty = the full test suite")
    timeout_s: float | None = Field(default=None, gt=0, description="Timeout in seconds (default from configuration)")


class RunTestsTool(_CheckTool):
    name = "run_tests"
    description = (
        "Run the project's test suite (or only the given test files) with the detected test runner. "
        "Returns pass/fail counts and each failing test with its message and location."
    )
    Input = RunTestsInput
    kind = CheckKind.TEST
    verb = "Running tests"


class RunLinterInput(ToolInput):
    files: list[str] = Field(default_factory=list, description=_FILES_FIELD)
    timeout_s: float | None = Field(default=None, gt=0, description="Timeout in seconds")


class RunLinterTool(_CheckTool):
    name = "run_linter"
    description = "Run the project's linter (whole project or the given files) and list the reported problems."
    Input = RunLinterInput
    kind = CheckKind.LINT
    verb = "Running linter"


class RunTypecheckInput(ToolInput):
    files: list[str] = Field(default_factory=list, description=_FILES_FIELD)
    timeout_s: float | None = Field(default=None, gt=0, description="Timeout in seconds")


class RunTypecheckTool(_CheckTool):
    name = "run_typecheck"
    description = "Run the project's type checker (e.g. mypy, tsc) and list type errors with their locations."
    Input = RunTypecheckInput
    kind = CheckKind.TYPECHECK
    verb = "Running type checker"


class RunBuildInput(ToolInput):
    timeout_s: float | None = Field(default=None, gt=0, description="Timeout in seconds")


class RunBuildTool(_CheckTool):
    name = "run_build"
    description = "Run the project's build/compile command and list compiler errors."
    Input = RunBuildInput
    kind = CheckKind.BUILD
    verb = "Running build"


class RunFormatterInput(ToolInput):
    files: list[str] = Field(default_factory=list, description=_FILES_FIELD)
    check: bool = Field(
        default=True, description="true: only report files that need formatting; false: rewrite files in place"
    )
    timeout_s: float | None = Field(default=None, gt=0, description="Timeout in seconds")


class RunFormatterTool(_CheckTool):
    name = "run_formatter"
    description = (
        "Check formatting with the project's formatter (check=true, the default, never modifies files) "
        "or apply it (check=false rewrites the given files, or the whole project when no files are given)."
    )
    Input = RunFormatterInput
    kind = CheckKind.FORMAT
    verb = "Checking formatting"
    side_effect = SideEffect.WRITE

    def command(self, args: Any, ctx: ToolContext) -> ValidationCommand | None:
        if args.check:
            return super().command(args, ctx)
        engine = _engine(ctx)
        files = _resolve_files(ctx, list(args.files or []))
        for rel in files:
            ctx.guard.resolve(rel, for_write=True)  # protected paths are never rewritten
        vc: ValidationCommand | None = engine.format_write_command(files or None)
        if vc is None and files:
            raise ToolError(
                "the project's formatter cannot be limited to these files; "
                "call run_formatter with check=false and no files to format the whole project"
            )
        if vc is not None and args.timeout_s:
            vc = vc.model_copy(update={"timeout_s": float(args.timeout_s)})
        return vc

    def assess(self, args: Any, ctx: ToolContext) -> ActionAssessment:
        assessment = super().assess(args, ctx)
        if not args.check:
            assessment.summary = assessment.summary.replace(self.verb, "Formatting", 1)
            assessment.paths = list(args.files or [])
        return assessment

    async def run(self, args: RunFormatterInput, ctx: ToolContext) -> ToolResult:
        if args.check:
            return await super().run(args, ctx)
        engine = _engine(ctx)
        vc = self.command(args, ctx)
        if vc is None:
            raise ToolError("no formatter detected for this project; configure validation.format_command")
        paths = [ctx.guard.resolve(f) for f in _resolve_files(ctx, list(args.files or []))]
        for path in paths:
            if path.is_file():
                ctx.files.before_write(path)
        result = await engine.run_command(vc, ctx.cancel)
        for path in paths:
            if path.is_file():
                ctx.files.after_write(path)
        tool_result = to_tool_result(result)
        if result.ok():
            tool_result.content += "\nFiles may have been rewritten; re-read them before editing."
        return tool_result


TESTING_TOOLS: list[type[Tool]] = [RunTestsTool, RunLinterTool, RunTypecheckTool, RunBuildTool, RunFormatterTool]
