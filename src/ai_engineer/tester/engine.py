"""Runs validation checks (tests, lint, format, type checks, build, audit) and records them."""

from __future__ import annotations

import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path, PureWindowsPath
from typing import Any

from ..config.settings import Settings
from ..core.cancel import CancellationToken
from ..tools.process import CommandOutcome, build_env, default_shell, docker_wrap, run_shell
from .detect import (
    TIMEOUTS,
    plan_checks,
    targeted_lint_command,
    targeted_test_command,
    with_availability,
    write_format_command,
)
from .models import CheckKind, CheckResult, ValidationCommand
from .parsers import OUTPUT_TAIL_CHARS, parse_output

Runner = Callable[..., Awaitable[CommandOutcome]]  # same signature as tools.process.run_shell
MIN_OUTPUT_CHARS = 60000


def _is_abs(path: str) -> bool:
    return os.path.isabs(path) or PureWindowsPath(path).is_absolute()


class ValidationEngine:
    """Plans and runs validation commands for one workspace.

    Every result is appended to :attr:`history` (in memory; the caller persists it).
    """

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        profile: Any | None,
        runner: Runner = run_shell,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.settings = settings
        self.profile = profile
        self._runner = runner
        self._which = which
        self._checks: dict[CheckKind, ValidationCommand | None] | None = None
        self.history: list[CheckResult] = []

    # --- planning ------------------------------------------------------------------------------

    def checks(self) -> dict[CheckKind, ValidationCommand | None]:
        """The command chosen for each check kind (cached; see :meth:`refresh_checks`)."""
        if self._checks is None:
            plan = plan_checks(self.profile, self.settings.validation, self._which)
            if self.settings.terminal.sandbox == "docker":
                # tools live in the container image; host PATH says nothing about them
                plan = {
                    k: v.model_copy(update={"available": True, "unavailable_reason": ""}) if v else None
                    for k, v in plan.items()
                }
            self._checks = plan
        return dict(self._checks)

    def refresh_checks(self, profile: Any | None) -> None:
        self.profile = profile
        self._checks = None
        self.checks()

    def _frameworks(self) -> list[str]:
        return [str(f) for f in (getattr(self.profile, "frameworks", None) or [])]

    def command_for(self, kind: CheckKind, targeted_files: list[str] | None = None) -> ValidationCommand | None:
        """The command :meth:`run_check` would run (targeted when derivable, else full)."""
        kind = CheckKind(kind)
        vc = self.checks().get(kind)
        if vc is None or not targeted_files:
            return vc
        targeted: ValidationCommand | None = None
        if kind == CheckKind.TEST:
            targeted = targeted_test_command(vc, targeted_files, self._frameworks())
        elif kind in (CheckKind.LINT, CheckKind.TYPECHECK, CheckKind.FORMAT):
            targeted = targeted_lint_command(vc, targeted_files)
        return targeted or vc

    def format_write_command(self, files: list[str] | None = None) -> ValidationCommand | None:
        """The file-rewriting formatter (``ruff format <files>`` ...), or None if unknown."""
        command = write_format_command(self.profile, self.settings.validation, files)
        if command is None:
            return None
        source = "config" if self.settings.validation.format_command is not None else "profile"
        vc = ValidationCommand(
            kind=CheckKind.FORMAT, command=command, source=source, timeout_s=TIMEOUTS[CheckKind.FORMAT],
            scope="targeted" if files else "full",
        )
        if self.settings.terminal.sandbox == "docker":
            return vc
        root = getattr(self.profile, "root", None) or str(self.workspace)
        return with_availability(vc, self._which, str(root))

    # --- running --------------------------------------------------------------------------------

    def _record(self, result: CheckResult) -> CheckResult:
        self.history.append(result)
        return result

    def latest(self, kind: CheckKind) -> CheckResult | None:
        """Most recent recorded result for ``kind``."""
        return next((r for r in reversed(self.history) if r.kind == kind), None)

    def _relativize(self, result: CheckResult, cwd: Path) -> None:
        def rel(path: str | None) -> str | None:
            if not path:
                return path
            candidate = Path(path) if _is_abs(path) else cwd / path
            for p in (candidate, candidate.resolve() if candidate.exists() else candidate):
                try:
                    return p.relative_to(self.workspace).as_posix()
                except ValueError:
                    continue
            return path

        if cwd == self.workspace:
            for f in result.failures:
                if f.file and _is_abs(f.file):
                    f.file = rel(f.file)
            for d in result.diagnostics:
                if d.file and _is_abs(d.file):
                    d.file = rel(d.file)
            return
        for f in result.failures:
            f.file = rel(f.file)
        for d in result.diagnostics:
            d.file = rel(d.file)

    async def run_command(self, vc: ValidationCommand, cancel: CancellationToken | None = None) -> CheckResult:
        """Run one validation command and parse its output."""
        if not vc.available:
            return self._record(CheckResult(
                kind=vc.kind, command=vc.command, status="unavailable", classification="command_not_found",
                summary=f"{vc.kind}: {vc.unavailable_reason or 'command unavailable'} ({vc.command})",
            ))
        if cancel is not None and cancel.cancelled:
            return self._record(CheckResult(
                kind=vc.kind, command=vc.command, status="cancelled", summary=f"{vc.kind}: cancelled before start",
            ))
        term = self.settings.terminal
        cwd = (self.workspace / vc.cwd).resolve()
        if not cwd.is_dir():
            return self._record(CheckResult(
                kind=vc.kind, command=vc.command, status="error", classification="environment",
                summary=f"{vc.kind}: working directory does not exist: {vc.cwd}",
            ))
        command = vc.command
        if term.sandbox == "docker":
            command = docker_wrap(command, self.workspace, cwd, term)
        max_chars = max(term.max_output_chars, MIN_OUTPUT_CHARS)
        try:
            outcome = await self._runner(
                command, cwd, build_env(term), vc.timeout_s, max_chars, cancel, default_shell(term)
            )
        except OSError as exc:
            return self._record(CheckResult(
                kind=vc.kind, command=vc.command, status="error", classification="environment",
                summary=f"{vc.kind}: could not start command: {exc}",
            ))
        if outcome.cancelled:
            return self._record(CheckResult(
                kind=vc.kind, command=vc.command, status="cancelled", duration_s=round(outcome.duration_s, 3),
                summary=f"{vc.kind}: cancelled after {outcome.duration_s:.1f}s",
                output_tail=outcome.output[-OUTPUT_TAIL_CHARS:],
            ))
        result = parse_output(
            vc.kind, vc.command, outcome.output, outcome.exit_code, outcome.timed_out, duration_s=outcome.duration_s
        )
        self._relativize(result, cwd)
        return self._record(result)

    async def run_check(
        self,
        kind: CheckKind,
        *,
        targeted_files: list[str] | None = None,
        cancel: CancellationToken | None = None,
    ) -> CheckResult:
        """Run the planned command for ``kind`` (restricted to ``targeted_files`` when possible)."""
        kind = CheckKind(kind)
        vc = self.command_for(kind, targeted_files)
        if vc is None:
            return self._record(CheckResult(
                kind=kind, command="", status="unavailable", classification="", summary=f"no {kind} command detected",
            ))
        return await self.run_command(vc, cancel)

    async def run_many(
        self,
        kinds: list[CheckKind],
        *,
        targeted_files: list[str] | None = None,
        cancel: CancellationToken | None = None,
    ) -> list[CheckResult]:
        """Run checks sequentially in the given order."""
        results: list[CheckResult] = []
        for kind in kinds:
            results.append(await self.run_check(kind, targeted_files=targeted_files, cancel=cancel))
        return results
