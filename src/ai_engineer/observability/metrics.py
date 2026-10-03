"""Metrics derived from events: durations, latencies, tokens, retries, failures, test iterations."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from ..core.events import Event, EventBus, EventType


def _ts(event: Event) -> float:
    try:
        return datetime.fromisoformat(event.ts).timestamp()
    except ValueError:
        return time.time()


class MetricsCollector:
    """Aggregates per-task metrics from the event stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._stage_start: dict[tuple[str, str], float] = {}

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self)

    def _m(self, task_id: str) -> dict[str, Any]:
        if task_id not in self._tasks:
            self._tasks[task_id] = {
                "started_at": None,
                "finished_at": None,
                "duration_s": None,
                "events": Counter(),
                "model_calls": 0,
                "model_latency_s": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_estimated": False,
                "model_retries": 0,
                "model_fallbacks": 0,
                "tool_calls": 0,
                "tool_errors": 0,
                "tool_denied": 0,
                "tool_latency_s": defaultdict(float),
                "tool_counts": Counter(),
                "test_runs": 0,
                "test_failures": 0,
                "fix_attempts": 0,
                "loops_detected": 0,
                "context_resets": 0,
                "approvals_requested": 0,
                "human_interventions": 0,
                "files_changed": set(),
                "stage_durations_s": defaultdict(float),
            }
        return self._tasks[task_id]

    def __call__(self, event: Event) -> None:
        if not event.task_id:
            return
        with self._lock:
            m = self._m(event.task_id)
            m["events"][str(event.type)] += 1
            t = event.type
            data = event.data
            if t in (EventType.TASK_STARTED, EventType.TASK_RESUMED) and m["started_at"] is None:
                m["started_at"] = _ts(event)
            elif t in (EventType.TASK_COMPLETED, EventType.TASK_FAILED, EventType.TASK_BLOCKED, EventType.TASK_CANCELLED, EventType.TASK_INTERRUPTED):
                m["finished_at"] = _ts(event)
                if m["started_at"] is not None:
                    m["duration_s"] = round(m["finished_at"] - m["started_at"], 3)
            elif t == EventType.MODEL_RESULT:
                m["model_calls"] += 1
                m["model_latency_s"] += float(data.get("latency_s", 0) or 0)
                m["input_tokens"] += int(data.get("input_tokens", 0) or 0)
                m["output_tokens"] += int(data.get("output_tokens", 0) or 0)
                m["tokens_estimated"] = m["tokens_estimated"] or bool(data.get("estimated"))
            elif t == EventType.MODEL_RETRY:
                m["model_retries"] += 1
            elif t == EventType.MODEL_FALLBACK:
                m["model_fallbacks"] += 1
            elif t == EventType.TOOL_RESULT:
                m["tool_calls"] += 1
                tool = str(data.get("tool", "?"))
                m["tool_counts"][tool] += 1
                m["tool_latency_s"][tool] += float(data.get("duration_s", 0) or 0)
                if not data.get("ok", True):
                    m["tool_errors"] += 1
            elif t == EventType.TOOL_DENIED:
                m["tool_denied"] += 1
            elif t == EventType.TEST_STARTED:
                m["test_runs"] += 1
            elif t == EventType.TEST_FAILED:
                m["test_failures"] += 1
            elif t == EventType.FIX_ATTEMPTED:
                m["fix_attempts"] += 1
            elif t == EventType.LOOP_DETECTED:
                m["loops_detected"] += 1
            elif t == EventType.CONTEXT_COMPACTED:
                m["context_resets"] += 1
            elif t == EventType.APPROVAL_REQUIRED:
                m["approvals_requested"] += 1
            elif t == EventType.APPROVAL_RESOLVED and data.get("by") not in ("memory", "policy", "timeout", "allow-all", "system"):
                m["human_interventions"] += 1
            elif t == EventType.QUESTION_ASKED:
                m["human_interventions"] += 1
            elif t == EventType.FILE_CHANGED:
                path = data.get("path")
                if path:
                    m["files_changed"].add(path)
            elif t == EventType.STAGE_STARTED:
                self._stage_start[(event.task_id, event.stage)] = _ts(event)
            elif t == EventType.STAGE_COMPLETED:
                start = self._stage_start.pop((event.task_id, event.stage), None)
                if start is not None:
                    m["stage_durations_s"][event.stage] += _ts(event) - start

    def snapshot(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            m = self._tasks.get(task_id)
            if m is None:
                return {}
            out = dict(m)
            out["events"] = dict(m["events"])
            out["tool_counts"] = dict(m["tool_counts"])
            out["tool_latency_s"] = {k: round(v, 3) for k, v in m["tool_latency_s"].items()}
            out["stage_durations_s"] = {k: round(v, 3) for k, v in m["stage_durations_s"].items()}
            out["files_changed"] = sorted(m["files_changed"])
            out["model_latency_s"] = round(m["model_latency_s"], 3)
            if out["duration_s"] is None and m["started_at"] is not None:
                out["duration_s"] = round(time.time() - m["started_at"], 3)
            return out
