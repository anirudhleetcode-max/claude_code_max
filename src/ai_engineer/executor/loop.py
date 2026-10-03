"""The agent loop: model ⇄ tools until the model submits, a budget is exhausted, or no progress is made."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..core.errors import AllModelsFailedError, ContextLengthError
from ..core.events import EventType
from ..core.types import Message, StopReason, TextBlock, ToolResultBlock, ToolUseBlock
from ..core.util import sha1_text
from ..models.base import ModelRequest, request_tokens
from ..tools.base import ToolContext
from ..tools.executor import ToolExecutor


@dataclass
class LoopLimits:
    max_steps: int = 60
    max_seconds: float = 3600.0
    context_budget_tokens: int = 80000
    repeated_call_limit: int = 3
    max_nudges: int = 2


@dataclass
class LoopResult:
    status: str  # finished | finished_no_submit | max_steps | timeout | refused | model_unavailable | no_progress | error
    submission: dict[str, Any] | None = None
    final_text: str = ""
    steps: int = 0
    tool_calls: int = 0
    context_resets: int = 0
    error: str = ""
    notes: list[str] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.status in ("finished", "finished_no_submit")


@dataclass
class _Action:
    name: str
    summary: str
    result_head: str
    is_error: bool


class AgentLoop:
    def __init__(
        self,
        model: Any,
        executor: ToolExecutor,
        ctx: ToolContext,
        *,
        role: str,
        stage: str,
        system: str,
        tools: list[str],
        finish_tool: str,
        limits: LoopLimits,
    ) -> None:
        self.model = model
        self.executor = executor
        self.ctx = ctx
        self.role = role
        self.stage = stage
        self.system = system
        self.tool_names = [t for t in dict.fromkeys([*tools, finish_tool]) if executor.registry.get(t) is not None]
        self.finish_tool = finish_tool
        self.limits = limits
        self._actions: list[_Action] = []
        self._notes: list[str] = []

    # ---- helpers ----------------------------------------------------------------------

    def _emit(self, etype: EventType, message: str, **data: Any) -> None:
        self.ctx.bus.emit(etype, message, stage=self.stage, data={"role": self.role, **data})

    def _handoff(self, task_message: str) -> list[Message]:
        """Start a fresh conversation seeded with a deterministic progress summary.

        Conversations are never edited in place; a reset creates a new one.
        """
        changed = self.ctx.files.changed_paths()
        lines = [task_message, "", "## Progress so far", "The previous conversation was reset to stay within the context budget."]
        if changed:
            lines.append("Files you have changed: " + ", ".join(changed))
        if self._actions:
            lines.append("Most recent actions:")
            for action in self._actions[-12:]:
                status = "ERROR" if action.is_error else "ok"
                lines.append(f"- {action.summary} → {status}: {action.result_head}")
        if self._notes:
            lines.append("Your recent notes:")
            lines += [f"- {n[:400]}" for n in self._notes[-4:]]
        lines.append("Continue from here. Re-read any file before editing it (your earlier reads were discarded).")
        self.ctx.files.forget_reads()
        return [Message.user("\n".join(lines))]

    def _request(self, messages: list[Message]) -> ModelRequest:
        return ModelRequest(
            messages=messages,
            system=self.system,
            tools=self.executor.registry.specs(self.tool_names),
            metadata={"role": self.role, "stage": self.stage, "task_id": self.ctx.task_id},
        )

    # ---- main loop ------------------------------------------------------------------------

    async def run(self, task_message: str) -> LoopResult:
        messages: list[Message] = [Message.user(task_message)]
        allowed = set(self.tool_names)
        started = time.monotonic()
        repeats: Counter[str] = Counter()
        warned: set[str] = set()
        interventions = 0
        nudges = 0
        resets = 0
        tool_calls = 0
        steps = 0
        last_text = ""

        def result(status: str, **kw: Any) -> LoopResult:
            return LoopResult(
                status=status, steps=steps, tool_calls=tool_calls, context_resets=resets, notes=list(self._notes),
                transcript=list(messages), final_text=kw.pop("final_text", last_text), **kw,
            )

        budget = self.limits.context_budget_tokens
        # Size of a fresh conversation (system + tools + task). A reset only helps once the conversation
        # has grown well past it; otherwise an oversized task context would reset on every step.
        fresh_tokens = request_tokens(self._request(messages))
        oversize_warned = False
        while steps < self.limits.max_steps:
            self.ctx.cancel.raise_if_cancelled()
            if time.monotonic() - started > self.limits.max_seconds:
                return result("timeout", error=f"time budget of {self.limits.max_seconds:.0f}s exhausted")
            req = self._request(messages)
            tokens = request_tokens(req)
            if tokens > budget and len(messages) > 1 and tokens - fresh_tokens > budget // 2:
                messages = self._handoff(task_message)
                resets += 1
                self._emit(EventType.CONTEXT_COMPACTED, "context budget reached; continuing in a fresh conversation with a progress summary")
                req = self._request(messages)
                fresh_tokens = request_tokens(req)
            if fresh_tokens > budget and not oversize_warned:
                oversize_warned = True
                self._emit(EventType.WARNING, f"the task context alone (~{fresh_tokens} tokens) exceeds the context budget ({budget}); "
                           "continuing, but consider a model with a larger context window or a smaller max_output_tokens")
            try:
                resp = await self.model.generate(req, cancel=self.ctx.cancel)
            except ContextLengthError as exc:
                if len(messages) <= 1:
                    return result("error", error=f"task context alone exceeds the model's window: {exc}")
                messages = self._handoff(task_message)
                resets += 1
                self._emit(EventType.CONTEXT_COMPACTED, "model rejected the context size; reset with a progress summary")
                continue
            except AllModelsFailedError as exc:
                return result("model_unavailable", error=str(exc))
            steps += 1
            assistant = resp.to_message()
            if not assistant.content:
                assistant = Message.assistant("(empty response)")
            messages.append(assistant)
            text = resp.text().strip()
            if text:
                last_text = text
                self._notes.append(text)
                self._emit(EventType.INFO, text[:240])
            if resp.stop_reason == StopReason.REFUSAL:
                return result("refused", error=resp.refusal_detail or "the model declined the request")
            calls = resp.tool_uses()
            if calls:
                nudges = 0  # the nudge limit counts consecutive text-only turns, not all of them
            else:
                if resp.stop_reason == StopReason.MAX_TOKENS:
                    messages.append(Message.user("Your reply was cut off by the output limit. Continue, working in smaller steps (smaller edits, fewer files per call)."))
                    continue
                nudges += 1
                if nudges > self.limits.max_nudges:
                    return result("finished_no_submit")
                messages.append(Message.user(
                    f"Continue working using the tools. When the work is complete and verified, call {self.finish_tool} "
                    "(it is the only way to finish)."
                ))
                continue
            finish_calls = [c for c in calls if c.name == self.finish_tool]
            other_calls = [c for c in calls if c.name != self.finish_tool]
            results: list[ToolResultBlock] = []
            if other_calls:
                tool_calls += len(other_calls)
                results = await self.executor.execute_many(other_calls, self.ctx, allowed)
                for call, res in zip(other_calls, results, strict=True):
                    self._record_action(call, res)
            if finish_calls:
                submission, error = self._validate_finish(finish_calls[0])
                if submission is not None:
                    results.append(ToolResultBlock(tool_use_id=finish_calls[0].id, name=self.finish_tool, content="received"))
                    for extra in finish_calls[1:]:
                        results.append(ToolResultBlock(tool_use_id=extra.id, name=self.finish_tool, content="ignored duplicate"))
                    messages.append(Message(role="user", content=list(results)))
                    return result("finished", submission=submission)
                results.append(ToolResultBlock(tool_use_id=finish_calls[0].id, name=self.finish_tool, content=f"ERROR: {error}", is_error=True))
                for extra in finish_calls[1:]:
                    results.append(ToolResultBlock(tool_use_id=extra.id, name=self.finish_tool, content="ERROR: duplicate", is_error=True))
            content: list[Any] = list(results)
            for call, res in zip(other_calls, results[: len(other_calls)], strict=True):
                key = sha1_text(call.name + json.dumps(call.input, sort_keys=True) + sha1_text(res.content))
                repeats[key] += 1
                if repeats[key] >= self.limits.repeated_call_limit and key not in warned:
                    warned.add(key)
                    interventions += 1
                    self._emit(EventType.LOOP_DETECTED, f"repeated {call.name} call with identical result", tool=call.name)
                    content.append(TextBlock(text=(
                        f"Note: you have called {call.name} with the same arguments {repeats[key]} times and got the same "
                        "result. Repeating it will not help — change your approach or gather different information."
                    )))
            if interventions >= 3:
                messages.append(Message(role="user", content=content))
                return result("no_progress", error="the agent kept repeating identical actions")
            messages.append(Message(role="user", content=content))
        return result("max_steps", error=f"step budget of {self.limits.max_steps} exhausted")

    def _record_action(self, call: ToolUseBlock, res: ToolResultBlock) -> None:
        tool = self.executor.registry.get(call.name)
        summary = call.name
        if tool is not None:
            try:
                summary = tool.summarize(tool.Input.model_validate(call.input))
            except ValidationError:
                summary = call.name
        head = " ".join(res.content.split())[:200]
        self._actions.append(_Action(call.name, summary, head, res.is_error))

    def _validate_finish(self, call: ToolUseBlock) -> tuple[dict[str, Any] | None, str]:
        tool = self.executor.registry.get(self.finish_tool)
        if tool is None:
            return dict(call.input), ""
        try:
            return tool.Input.model_validate(call.input).model_dump(), ""
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}" for e in exc.errors()[:5])
            return None, f"invalid {self.finish_tool} arguments: {problems}"
