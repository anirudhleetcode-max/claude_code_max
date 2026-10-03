"""Post-task retrospective: deterministic analysis producing advisory improvement suggestions.

Suggestions are written to `.agent/reports/<task>.improvements.json` and summarised by
`aie improve`. They never modify the agent itself: changes to the agent must go
through normal development, testing and review.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def analyze(metrics: dict[str, Any], state: Any, failures: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    def add(kind: str, observation: str, suggestion: str) -> None:
        findings.append({"kind": kind, "observation": observation, "suggestion": suggestion})

    tool_counts: dict[str, int] = metrics.get("tool_counts", {}) or {}
    tool_calls = int(metrics.get("tool_calls", 0) or 0)
    if tool_calls and metrics.get("tool_errors", 0) / max(1, tool_calls) > 0.25:
        add("tool_errors", f"{metrics['tool_errors']} of {tool_calls} tool calls failed", "inspect the trace for recurring tool errors; improve tool descriptions or prompts for those tools")
    if metrics.get("tool_denied", 0):
        add("permissions", f"{metrics['tool_denied']} tool call(s) were denied by policy", "if these actions are routine for this project, add precise allow_commands patterns; otherwise keep them denied")
    if metrics.get("loops_detected", 0):
        add("loops", f"{metrics['loops_detected']} repetition loop(s) detected", "the model repeated identical actions; consider a stronger model for the coder/debugger role or more specific task descriptions")
    if metrics.get("model_fallbacks", 0) or metrics.get("model_retries", 0) > 3:
        add("reliability", f"{metrics.get('model_retries', 0)} retries and {metrics.get('model_fallbacks', 0)} fallbacks", "check provider health (`aie doctor`) and rate limits; reorder fallback chains if one provider is unreliable")
    if metrics.get("context_resets", 0) > 1:
        add("context", f"conversation was reset {metrics['context_resets']} times", "split the task into smaller subtasks or use a model with a larger context window")
    reads = tool_counts.get("read_file", 0)
    if reads > 40:
        add("efficiency", f"{reads} file reads", "use search_text/find_symbol/code_search before reading whole files")
    if metrics.get("fix_attempts", 0) >= 3:
        add("debugging", f"{metrics['fix_attempts']} fix attempts", "review failure records: recurring classifications may indicate missing environment setup or unclear requirements")
    classes = Counter(c for f in failures for c in str(f.get("classification", "")).split(",") if c)
    for cls, n in classes.most_common(3):
        if cls in ("command_not_found", "missing_dependency", "environment"):
            add("environment", f"{n} failure(s) classified as {cls}", "install the missing tools/dependencies or set explicit validation commands in .agent/config.toml")
    subtasks = getattr(state, "subtasks", {}) or {}
    unverified = [sid for sid, s in subtasks.items() if s.status == "completed_unverified"]
    if unverified:
        add("verification", f"{len(unverified)} subtask(s) could not be fully verified", "add tests or configure test/lint/typecheck commands so future work can be verified")
    stage_durations: dict[str, float] = metrics.get("stage_durations_s", {}) or {}
    if stage_durations:
        slowest = max(stage_durations.items(), key=lambda kv: kv[1])
        if slowest[1] > 600:
            add("performance", f"stage '{slowest[0]}' took {slowest[1]:.0f}s", "consider targeted test commands or disabling validation.full_suite_per_subtask for large suites")
    return {"findings": findings, "advisory_only": True} if findings else {}
