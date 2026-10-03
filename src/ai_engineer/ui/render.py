"""Terminal rendering of agent events: concise action summaries and evidence."""

from __future__ import annotations

import os
import sys
from typing import TextIO

from ..core.events import Event, EventType

_COLORS = {"dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36", "bold": "1"}

_STYLE: dict[EventType, tuple[str, str]] = {
    EventType.TASK_STARTED: ("▶", "bold"),
    EventType.TASK_RESUMED: ("▶", "bold"),
    EventType.STAGE_STARTED: ("●", "cyan"),
    EventType.UNDERSTANDING_CREATED: ("✎", "cyan"),
    EventType.PLAN_CREATED: ("☰", "cyan"),
    EventType.SUBTASK_STARTED: ("◆", "bold"),
    EventType.SUBTASK_COMPLETED: ("◆", "bold"),
    EventType.TOOL_CALLED: ("→", "dim"),
    EventType.TOOL_DENIED: ("✋", "yellow"),
    EventType.FILE_CHANGED: ("✏", "blue"),
    EventType.TEST_STARTED: ("⧗", "dim"),
    EventType.TEST_PASSED: ("✔", "green"),
    EventType.TEST_FAILED: ("✘", "red"),
    EventType.VALIDATION_RESULT: ("•", "dim"),
    EventType.FAILURE_RECORDED: ("⚠", "yellow"),
    EventType.FIX_ATTEMPTED: ("🔧", "yellow"),
    EventType.LOOP_DETECTED: ("↻", "yellow"),
    EventType.REVIEW_STARTED: ("🔍", "magenta"),
    EventType.REVIEW_COMPLETED: ("🔍", "magenta"),
    EventType.GATES_EVALUATED: ("☑", "bold"),
    EventType.CHECKPOINT_CREATED: ("⚑", "dim"),
    EventType.CHECKPOINT_RESTORED: ("⚑", "yellow"),
    EventType.COMMIT_CREATED: ("⎇", "green"),
    EventType.MODEL_RETRY: ("↺", "yellow"),
    EventType.MODEL_FALLBACK: ("⇄", "yellow"),
    EventType.CONTEXT_COMPACTED: ("⇲", "dim"),
    EventType.APPROVAL_REQUIRED: ("?", "yellow"),
    EventType.QUESTION_ASKED: ("?", "yellow"),
    EventType.TASK_COMPLETED: ("■", "green"),
    EventType.TASK_FAILED: ("■", "red"),
    EventType.TASK_BLOCKED: ("■", "yellow"),
    EventType.TASK_CANCELLED: ("■", "yellow"),
    EventType.TASK_INTERRUPTED: ("■", "yellow"),
    EventType.REPORT_CREATED: ("📄", "bold"),
    EventType.WARNING: ("!", "yellow"),
    EventType.ERROR: ("!", "red"),
    EventType.INFO: ("·", "dim"),
}

_VERBOSE_ONLY = {EventType.MODEL_CALLED, EventType.MODEL_RESULT, EventType.TOOL_RESULT, EventType.STAGE_COMPLETED, EventType.SESSION_STARTED, EventType.TASK_CREATED, EventType.MEMORY_UPDATED}


class TerminalRenderer:
    def __init__(self, stream: TextIO | None = None, verbose: bool = False, color: bool | None = None) -> None:
        self.stream = stream or sys.stderr
        self.verbose = verbose
        if color is None:
            color = self.stream.isatty() and os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb"
        self.color = color

    def _c(self, text: str, style: str) -> str:
        if not self.color or style not in _COLORS:
            return text
        return f"\033[{_COLORS[style]}m{text}\033[0m"

    def __call__(self, event: Event) -> None:
        if event.type in _VERBOSE_ONLY and not self.verbose:
            return
        if event.type == EventType.INFO and not self.verbose and event.data.get("role"):
            # model commentary between tool calls: shown only in verbose mode
            return
        icon, style = _STYLE.get(event.type, ("·", "dim"))
        indent = "    " if event.type in (EventType.TOOL_CALLED, EventType.FILE_CHANGED, EventType.TEST_STARTED, EventType.VALIDATION_RESULT, EventType.CHECKPOINT_CREATED, EventType.INFO, EventType.TOOL_RESULT) else "  "
        message = event.message.replace("\n", " ")
        if len(message) > 220 and not self.verbose:
            message = message[:217] + "..."
        line = f"{indent}{self._c(icon, style)} {self._c(message, style if style in ('red', 'green', 'yellow', 'bold') else '')}"
        try:
            print(line, file=self.stream, flush=True)
        except (OSError, UnicodeEncodeError):
            print(line.encode("ascii", "replace").decode(), file=self.stream, flush=True)
