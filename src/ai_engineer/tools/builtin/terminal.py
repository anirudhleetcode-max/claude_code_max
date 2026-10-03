"""Terminal tools: risk-classified command execution and background processes."""

from __future__ import annotations

from pydantic import Field

from ...config.settings import PermissionLevel
from ...core.errors import ToolError
from ...security.command_risk import classify_command
from ..base import ActionAssessment, SideEffect, Tool, ToolContext, ToolInput, ToolResult
from ..process import build_env, default_shell, docker_wrap, run_shell


class RunCommandInput(ToolInput):
    command: str = Field(description="Shell command to run (non-interactive; stdin is closed)")
    cwd: str = Field(default=".", description="Working directory relative to the workspace root")
    timeout_s: float | None = Field(default=None, gt=0, description="Timeout in seconds (default from configuration)")
    background: bool = Field(default=False, description="Start as a background process (servers, watchers) and return immediately")


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the workspace and return its exit code and combined output. Commands are "
        "risk-classified: destructive or outward-facing commands may be denied or need human approval. "
        "Never start interactive programs. Prefer the dedicated tools for reading/searching files and for git."
    )
    Input = RunCommandInput
    level = PermissionLevel.READ_ONLY  # actual level comes from the risk classification
    side_effect = SideEffect.EXECUTE
    timeout_s = 3600.0

    def assess(self, args: RunCommandInput, ctx: ToolContext) -> ActionAssessment:
        cwd = ctx.guard.resolve(args.cwd)
        assessment = classify_command(args.command, ctx.workspace)
        level = PermissionLevel.READ_ONLY if assessment.read_only else PermissionLevel.DEVELOPMENT
        if args.background:
            level = max(level, PermissionLevel.DEVELOPMENT)
        where = "" if cwd == ctx.guard.root else f" (in {ctx.guard.relative(cwd)})"
        return ActionAssessment(
            level=level,
            summary=f"Running: {args.command}{where}",
            risk=assessment.risk,
            read_only=assessment.read_only and not args.background,
            command=args.command,
            details={"risk_reasons": assessment.reasons},
        )

    def effective_timeout(self, args: RunCommandInput, ctx: ToolContext) -> float:
        term = ctx.settings.terminal
        requested = args.timeout_s or term.default_timeout_s
        # the outer timeout leaves room for the process-tree kill
        return min(requested, term.max_timeout_s) + 15

    async def run(self, args: RunCommandInput, ctx: ToolContext) -> ToolResult:
        term = ctx.settings.terminal
        cwd = ctx.guard.resolve(args.cwd)
        if not cwd.is_dir():
            raise ToolError(f"working directory does not exist: {args.cwd}")
        command = args.command
        if term.sandbox == "docker":
            command = docker_wrap(command, ctx.workspace, cwd, term)
        env = build_env(term)
        shell = default_shell(term)
        if args.background:
            proc = await ctx.processes.start(command, cwd, env, shell)
            return ToolResult(
                content=f"started background process {proc.id} (pid {proc.proc.pid}). "
                "Use process_output to read its output and process_stop to stop it.",
                data={"process_id": proc.id, "command": args.command},
            )
        timeout = min(args.timeout_s or term.default_timeout_s, term.max_timeout_s)
        outcome = await run_shell(command, cwd, env, timeout, term.max_output_chars, ctx.cancel, shell)
        status = "timed out" if outcome.timed_out else f"exit code {outcome.exit_code}"
        header = f"[{status}; {outcome.duration_s:.1f}s]"
        return ToolResult(
            ok=outcome.ok,
            content=f"{header}\n{outcome.output}".rstrip(),
            error="" if outcome.ok else status,
            truncated=outcome.truncated,
            data={"exit_code": outcome.exit_code, "command": args.command, "timed_out": outcome.timed_out},
        )


class ProcessListInput(ToolInput):
    pass


class ProcessListTool(Tool):
    name = "process_list"
    description = "List background processes started by the agent in this session."
    Input = ProcessListInput

    async def run(self, args: ProcessListInput, ctx: ToolContext) -> ToolResult:
        procs = [p.describe() for p in ctx.processes.list()]
        if not procs:
            return ToolResult(content="no background processes")
        lines = [f"{p['id']} pid={p['pid']} running={p['running']} exit={p['exit_code']} {p['command']}" for p in procs]
        return ToolResult(content="\n".join(lines), data={"processes": procs})


class ProcessOutputInput(ToolInput):
    process_id: str
    tail_chars: int = Field(default=5000, ge=100, le=50000)


class ProcessOutputTool(Tool):
    name = "process_output"
    description = "Read recent output of a background process."
    Input = ProcessOutputInput

    async def run(self, args: ProcessOutputInput, ctx: ToolContext) -> ToolResult:
        proc = ctx.processes.get(args.process_id)
        if proc is None:
            raise ToolError(f"no such process: {args.process_id}")
        text = proc.buffer.value()[-args.tail_chars :]
        state = "running" if proc.running else f"exited with {proc.proc.returncode}"
        return ToolResult(content=f"[{state}]\n{text}")


class ProcessStopInput(ToolInput):
    process_id: str


class ProcessStopTool(Tool):
    name = "process_stop"
    description = "Stop a background process started by the agent (terminates its whole process tree)."
    Input = ProcessStopInput
    level = PermissionLevel.DEVELOPMENT
    side_effect = SideEffect.EXECUTE

    async def run(self, args: ProcessStopInput, ctx: ToolContext) -> ToolResult:
        if not await ctx.processes.stop(args.process_id):
            raise ToolError(f"no such process: {args.process_id}")
        return ToolResult(content=f"stopped {args.process_id}")


TERMINAL_TOOLS: list[type[Tool]] = [RunCommandTool, ProcessListTool, ProcessOutputTool, ProcessStopTool]
