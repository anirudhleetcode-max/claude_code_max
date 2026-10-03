"""Final engineering report (Markdown + JSON). States only what was actually observed."""

from __future__ import annotations

from typing import Any

from ..core.util import human_duration, utcnow_iso
from ..tasks.store import CheckpointRecord, Task, TaskStatus
from .state import PipelineState


def _gate_rows(gates: dict[str, Any] | None) -> list[str]:
    if not gates or not gates.get("results"):
        return []
    rows = ["| Gate | Mode | Status | Detail |", "|---|---|---|---|"]
    for r in gates["results"]:
        detail = str(r.get("detail", "")).replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {r['name']} | {r['mode']} | **{r['status']}** | {detail} |")
    return rows


def _check_line(c: dict[str, Any]) -> str:
    counts = []
    for key in ("passed", "failed", "errors", "skipped"):
        if c.get(key) is not None:
            counts.append(f"{c[key]} {key}")
    count_text = f" ({', '.join(counts)})" if counts else ""
    cmd = f" — `{c['command']}`" if c.get("command") else ""
    return f"- **{c.get('kind')}**: {c.get('status')}{count_text}{cmd} — {c.get('summary', '')}"


def build_report(
    task: Task,
    state: PipelineState,
    status: TaskStatus,
    error: str | None,
    metrics: dict[str, Any],
    diffstat: str,
    failures: list[dict[str, Any]],
    checkpoints: list[CheckpointRecord],
    roles: dict[str, list[str]],
) -> tuple[str, dict[str, Any]]:
    u = state.understanding or {}
    md: list[str] = [f"# Engineering report: {task.title}", ""]
    md += [
        f"- **Task ID:** `{task.id}`",
        f"- **Status:** **{status}**" + (f" — {error}" if error else ""),
        f"- **Created:** {task.created}  **Report written:** {task.finished or utcnow_iso()}",
    ]
    if metrics.get("duration_s") is not None:
        md.append(f"- **Duration:** {human_duration(float(metrics['duration_s']))}")
    if roles:
        md.append("- **Models (role → fallback chain):** " + "; ".join(f"{r}: {' → '.join(c)}" for r, c in roles.items()))
    if state.branch:
        md.append(f"- **Branch:** `{state.branch}` (from `{state.original_branch}`)")
    md.append("")

    md += ["## Summary", u.get("summary", task.description[:500]), ""]
    if status == TaskStatus.COMPLETED:
        md.append("All configured quality gates ran and passed.")
    elif status == TaskStatus.COMPLETED_UNVERIFIED:
        md.append("The work is done, but some required checks could not be verified (see *Unverified items*).")
    elif status == TaskStatus.FAILED:
        md.append("The task did **not** pass its quality gates (see *Quality gates* and *Failures*).")
    elif status in (TaskStatus.BLOCKED, TaskStatus.INTERRUPTED):
        md.append(f"The task stopped before completion and can be resumed with `aie resume {task.id}`.")
    md.append("")

    if state.answer:
        md += ["## Answer", str(state.answer.get("answer", "")), ""]
        if state.answer.get("evidence"):
            md += ["**Evidence:**", *[f"- `{e}`" for e in state.answer["evidence"]], ""]
        md += [f"Confidence: {state.answer.get('confidence', 'n/a')}", ""]

    if u.get("requirements"):
        md += ["## Requirements", *[f"- {r}" for r in u["requirements"]], ""]
    criteria_status: dict[str, dict[str, Any]] = {}
    for c in (state.final_review or {}).get("requirements", []):
        criteria_status[c["criterion"]] = c
    if u.get("acceptance_criteria"):
        md.append("## Acceptance criteria")
        for c in u["acceptance_criteria"]:
            st = criteria_status.get(c)
            md.append(f"- {c}" + (f" — **{st['status']}** ({st.get('evidence', '')})" if st else ""))
        md.append("")
    if u.get("assumptions"):
        md += ["## Assumptions", *[f"- {a}" for a in u["assumptions"]], ""]
    if state.clarifications:
        md += ["## Clarifications", *[f"- {c['question']} → {c['answer']}" for c in state.clarifications], ""]

    if state.subtasks:
        md += ["## Plan and subtasks"]
        if state.plan:
            md.append(f"Approach: {state.plan.get('approach', '')}  (plan source: {state.plan_source})")
        md += ["", "| Subtask | Status | Repairs | Reviews | Files | Commit |", "|---|---|---|---|---|---|"]
        for sid in state.order:
            s = state.subtasks[sid]
            md.append(f"| {sid}: {s.title} | {s.status} | {s.repair_iterations} | {s.review_iterations} | {len(s.files_changed)} | {(s.commit or '')[:10]} |")
        md.append("")
        for sid in state.order:
            s = state.subtasks[sid]
            if s.submission:
                md += [f"**{sid} — agent summary (claims, independently verified below):** {s.submission.get('summary', '')}"]
                if s.submission.get("unresolved"):
                    md.append(f"  - Unresolved per agent: {s.submission['unresolved']}")
            for note in s.notes:
                md.append(f"  - Note: {note}")
        md.append("")

    if diffstat.strip():
        md += ["## Changes", "```", diffstat.strip()[:6000], "```", ""]

    if state.baseline:
        md += ["## Baseline (before any change)", *[_check_line(c) for c in state.baseline.values()], ""]
    if state.final_validation:
        md += ["## Final validation (commands actually executed)", *[_check_line(c) for c in state.final_validation], ""]

    gate_rows = _gate_rows(state.gates)
    if gate_rows:
        md += ["## Quality gates", *gate_rows, ""]

    review = state.final_review
    if review:
        md += ["## Review", f"Verdict: **{review.get('verdict')}** ({review.get('source')}{', ' + review['model'] if review.get('model') else ''}) — {review.get('summary', '')}"]
        for i in review.get("issues", [])[:40]:
            loc = f" `{i.get('file')}:{i.get('line')}`" if i.get("file") else ""
            md.append(f"- [{i['severity']}/{i['category']}]{loc} {i['description']}")
        md.append("")
    if state.security_findings:
        md += ["## Security findings (changed code)", *[f"- [{f['severity']}] `{f['path']}:{f['line']}` {f['rule']}: {f['message']}" for f in state.security_findings[:40]], ""]

    if failures:
        md.append("## Failures and fixes")
        for f in failures[-20:]:
            md += [
                f"### Attempt {f.get('attempt')} ({f.get('subtask_id') or 'final'}) — {f.get('result')}",
                f"- Error: {f.get('error', '')[:500]}",
                f"- Likely causes: {'; '.join(f.get('likely_causes', []))}",
                f"- Hypothesis: {f.get('hypothesis') or 'n/a'}",
                f"- Change: {f.get('change') or 'n/a'}",
            ]
        md.append("")

    unverified: list[str] = []
    for r in (state.gates or {}).get("results", []):
        if r["status"] in ("UNVERIFIED", "PRE-EXISTING"):
            unverified.append(f"{r['name']}: {r['detail']}")
    for sid in state.order:
        s = state.subtasks[sid]
        if s.status not in ("completed",):
            unverified.append(f"subtask {sid} ended as {s.status}")
    if state.understanding_source == "heuristic":
        unverified.append("task understanding came from deterministic heuristics (no model response)")
    md += ["## Unverified items and limitations", *([f"- {x}" for x in unverified] or ["- none"]), ""]
    if state.notes:
        md += ["## Notes", *[f"- {n}" for n in state.notes], ""]

    if checkpoints:
        md += ["## Checkpoints (rollback)", "Restore any checkpoint with `aie checkpoints restore <id>` (a safety checkpoint is taken first).", ""]
        for cp in checkpoints[-15:]:
            md.append(f"- `{cp.id}` — {cp.label}{' (verified)' if cp.verified else ''}")
        md.append("")

    if metrics:
        tokens = f"{metrics.get('input_tokens', 0)} in / {metrics.get('output_tokens', 0)} out" + (" (partly estimated)" if metrics.get("tokens_estimated") else "")
        md += [
            "## Metrics",
            f"- Model calls: {metrics.get('model_calls', 0)}; tokens: {tokens}; model latency: {metrics.get('model_latency_s', 0)}s",
            f"- Retries: {metrics.get('model_retries', 0)}; fallbacks: {metrics.get('model_fallbacks', 0)}",
            f"- Tool calls: {metrics.get('tool_calls', 0)} (errors {metrics.get('tool_errors', 0)}, denied {metrics.get('tool_denied', 0)})",
            f"- Test runs: {metrics.get('test_runs', 0)}; fix attempts: {metrics.get('fix_attempts', 0)}; loops detected: {metrics.get('loops_detected', 0)}",
            f"- Context resets: {metrics.get('context_resets', 0)}; human interventions: {metrics.get('human_interventions', 0)}",
            "",
        ]
    data = {
        "task": task.model_dump(exclude={"state", "lease_owner", "lease_expires"}),
        "status": str(status),
        "error": error,
        "understanding": u,
        "plan": state.plan,
        "subtasks": {k: v.model_dump() for k, v in state.subtasks.items()},
        "baseline": state.baseline,
        "final_validation": state.final_validation,
        "gates": state.gates,
        "review": state.final_review,
        "security_findings": state.security_findings,
        "failures": failures,
        "answer": state.answer,
        "checkpoints": [c.model_dump() for c in checkpoints],
        "metrics": metrics,
        "unverified": unverified,
        "diffstat": diffstat,
    }
    return "\n".join(md), data
