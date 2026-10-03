"""Executes model tool calls safely: validate → assess → policy → approve → run → redact → audit."""

from __future__ import annotations

import asyncio
import difflib
import logging
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..core.cancel import run_cancellable
from ..core.errors import CancelledByUser, PermissionDenied, ToolError
from ..core.events import EventBus, EventType
from ..core.types import ToolResultBlock, ToolSpec, ToolUseBlock
from ..core.util import truncate_middle
from ..models.prompted_tools import INVALID_TOOL_CALL
from ..security.audit import AuditLog
from .approval import ApprovalBroker, ApprovalRequest, DenyAllBroker
from .base import Tool, ToolContext, ToolResult
from .permissions import Decision, PermissionPolicy

log = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in sorted(self._tools)]

    def specs(self, names: Iterable[str] | None = None) -> list[ToolSpec]:
        selected = self.names() if names is None else [n for n in names if n in self._tools]
        return [self._tools[n].spec() for n in selected]

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        return ToolRegistry(self._tools[n] for n in names if n in self._tools)


@dataclass
class ToolStats:
    calls: int = 0
    errors: int = 0
    denied: int = 0
    timeouts: int = 0
    total_s: float = 0.0
    by_tool: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, name: str, ok: bool, duration: float, denied: bool = False, timed_out: bool = False) -> None:
        self.calls += 1
        self.total_s += duration
        self.errors += 0 if ok else 1
        self.denied += 1 if denied else 0
        self.timeouts += 1 if timed_out else 0
        entry = self.by_tool.setdefault(name, {"calls": 0, "errors": 0, "total_s": 0.0})
        entry["calls"] += 1
        entry["errors"] += 0 if ok else 1
        entry["total_s"] += duration


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        approvals: ApprovalBroker | None = None,
        audit: AuditLog | None = None,
        bus: EventBus | None = None,
        max_result_chars: int = 20000,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.approvals = approvals or DenyAllBroker(bus)
        self.audit = audit or AuditLog(None)
        self.bus = bus
        self.max_result_chars = max_result_chars
        self.stats = ToolStats()

    def _emit(self, etype: EventType, message: str, **data: object) -> None:
        if self.bus is not None:
            self.bus.emit(etype, message, data=data)

    def _error_block(self, call: ToolUseBlock, message: str) -> ToolResultBlock:
        return ToolResultBlock(tool_use_id=call.id, name=call.name, content=f"ERROR: {message}", is_error=True)

    async def execute(self, call: ToolUseBlock, ctx: ToolContext, allowed: set[str] | None = None) -> ToolResultBlock:
        started = time.monotonic()
        if call.name == INVALID_TOOL_CALL:
            self.stats.record(call.name, False, 0.0)
            return self._error_block(call, f"malformed tool call: {call.input.get('error', '')}. Use valid JSON arguments.")
        tool = self.registry.get(call.name)
        if tool is None or (allowed is not None and call.name not in allowed):
            names = sorted(allowed) if allowed is not None else self.registry.names()
            close = difflib.get_close_matches(call.name, names, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            self.stats.record(call.name, False, 0.0)
            return self._error_block(call, f"unknown or unavailable tool '{call.name}'.{hint} Available: {', '.join(names)}")
        try:
            args = tool.Input.model_validate(call.input)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e.get('loc', ())) or '<args>'}: {e.get('msg')}" for e in exc.errors()[:8]
            )
            self.stats.record(call.name, False, 0.0)
            return self._error_block(call, f"invalid arguments for {call.name}: {problems}")

        try:
            assessment = tool.assess(args, ctx)
        except ToolError as exc:
            self.stats.record(call.name, False, 0.0)
            return self._error_block(call, str(exc))
        decision = self.policy.evaluate(assessment)
        risk = assessment.risk.name if assessment.risk is not None else ""
        if decision.decision == Decision.ASK:
            approval = await self.approvals.request(
                ApprovalRequest(
                    tool=tool.name, summary=assessment.summary, reason=decision.reason, risk=risk,
                    details={"command": assessment.command, "paths": assessment.paths, **assessment.details},
                    task_id=ctx.task_id,
                )
            )
            if not approval.approved:
                decision_reason = f"not approved ({approval.reason})"
                self._deny(call, tool.name, assessment.summary, decision_reason, ctx)
                return self._error_block(
                    call, f"permission denied: {decision_reason}. Choose a safer alternative or explain why it is necessary."
                )
        elif decision.decision == Decision.DENY:
            self._deny(call, tool.name, assessment.summary, decision.reason, ctx)
            return self._error_block(call, f"permission denied: {decision.reason}. Choose a safer alternative.")

        self._emit(EventType.TOOL_CALLED, assessment.summary, tool=tool.name, call_id=call.id, risk=risk)
        timeout = tool.effective_timeout(args, ctx)
        timed_out = False
        try:
            result = await run_cancellable(lambda: asyncio.wait_for(tool.run(args, ctx), timeout=timeout), ctx.cancel)
        except CancelledByUser:
            raise
        except TimeoutError:
            timed_out = True
            result = ToolResult.fail(f"{tool.name} timed out after {timeout:.0f}s")
        except PermissionDenied as exc:
            result = ToolResult.fail(f"permission denied: {exc}")
        except ToolError as exc:
            result = ToolResult.fail(str(exc))
        except Exception as exc:
            log.debug("tool %s crashed:\n%s", tool.name, traceback.format_exc())
            result = ToolResult.fail(f"internal error in {tool.name}: {type(exc).__name__}: {exc}")
        duration = time.monotonic() - started
        result.duration_s = duration

        content = ctx.redactor.redact_text(result.content or ("(no output)" if result.ok else result.error))
        limit = min(self.max_result_chars, ctx.settings.agent.tool_result_max_chars)
        if len(content) > limit:
            content = truncate_middle(content, limit)
            result.truncated = True
        self.stats.record(tool.name, result.ok, duration, timed_out=timed_out)
        self.audit.record(
            "tool", tool=tool.name, summary=assessment.summary, ok=result.ok, duration_s=round(duration, 3),
            task_id=ctx.task_id, error=result.error[:500] if result.error else "",
        )
        self._emit(
            EventType.TOOL_RESULT,
            f"{tool.name}: {'ok' if result.ok else 'failed'} ({duration:.1f}s)",
            tool=tool.name, call_id=call.id, ok=result.ok, duration_s=round(duration, 3),
            error=(result.error[:300] if result.error else ""), **_public_data(result.data),
        )
        if not result.ok and not content.startswith("ERROR"):
            content = f"ERROR: {content}"
        return ToolResultBlock(tool_use_id=call.id, name=call.name, content=content, is_error=not result.ok)

    def _deny(self, call: ToolUseBlock, tool: str, summary: str, reason: str, ctx: ToolContext) -> None:
        self.stats.record(tool, False, 0.0, denied=True)
        self.audit.record("tool_denied", tool=tool, summary=summary, reason=reason, task_id=ctx.task_id)
        self._emit(EventType.TOOL_DENIED, f"denied {summary}: {reason}", tool=tool, call_id=call.id, reason=reason)

    async def execute_many(self, calls: list[ToolUseBlock], ctx: ToolContext, allowed: set[str] | None = None) -> list[ToolResultBlock]:
        """Run independent read-only calls concurrently; anything with side effects runs in order."""
        tools = [self.registry.get(c.name) for c in calls]
        if len(calls) > 1 and all(t is not None and t.concurrency_safe for t in tools):
            return list(await asyncio.gather(*(self.execute(c, ctx, allowed) for c in calls)))
        results = []
        for call in calls:
            ctx.cancel.raise_if_cancelled()
            results.append(await self.execute(call, ctx, allowed))
        return results


def _public_data(data: dict) -> dict:
    """Small, UI-relevant fields from a tool result (never full outputs)."""
    keep = {}
    for key in ("path", "paths", "exit_code", "passed", "failed", "errors", "files_changed", "command", "matches"):
        if key in data:
            value = data[key]
            if isinstance(value, list):
                value = value[:20]
            if isinstance(value, str):
                value = value[:300]
            keep[key] = value
    return keep
