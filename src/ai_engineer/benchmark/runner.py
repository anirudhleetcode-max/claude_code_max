"""Benchmark runner.

Suites:
- ``harness``: scripted oracle transcripts (offline, deterministic). Verifies the agent
  *infrastructure* end to end. It is not a measure of model capability.
- ``model``: the same cases driven by your configured models (`--model provider:id`
  or AIE_MODEL). This measures real agent performance. Results depend on the model.

Every case is checked by hidden verification the agent never sees.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..config.settings import ModelsSettings
from ..core.util import atomic_write_json, utcnow_iso
from ..models.registry import ProviderRegistry
from ..tasks.store import TaskStatus
from .cases import ALL_CASES, BenchCase

PYTEST = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'


class CaseResult(BaseModel):
    id: str
    category: str
    title: str
    suite: str
    status: str
    completed: bool
    verified: bool
    verification: str
    duration_s: float
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    tool_calls: int = 0
    tool_errors: int = 0
    retries: int = 0
    fallbacks: int = 0
    test_runs: int = 0
    fix_attempts: int = 0
    recovered: bool = False
    human_interventions: int = 0
    context_resets: int = 0
    error: str | None = None


def _git(cwd: Path, *args: str) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@localhost", "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@localhost"}
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


def build_fixture(case: BenchCase, root: Path) -> Path:
    repo = root / "repo"
    for rel, content in case.files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


def run_hidden_checks(case: BenchCase, repo: Path, state: dict[str, Any], hidden_dir: Path) -> tuple[bool, str]:
    results: list[tuple[bool, str]] = []
    if case.hidden_tests:
        hidden_dir.mkdir(parents=True, exist_ok=True)
        for name, content in case.hidden_tests.items():
            (hidden_dir / name).write_text(content, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(hidden_dir), str(hidden_dir)],
            cwd=repo, env=env, capture_output=True, text=True, timeout=300, check=False,
        )
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
        results.append((proc.returncode == 0, f"hidden tests: {tail[0]}"))
    if case.verify is not None:
        results.append(case.verify(repo, state))
    if not results:
        return False, "no verification defined"
    return all(ok for ok, _ in results), "; ".join(msg for _, msg in results)


async def run_case(case: BenchCase, suite: str, work: Path, model: str | None = None) -> CaseResult:
    from ..providers.scripted import ScriptedProvider
    from ..runtime import Runtime

    root = work / case.id
    shutil.rmtree(root, ignore_errors=True)
    repo = build_fixture(case, root)
    overrides: dict[str, Any] = {
        "permissions": {"mode": "autonomous"},
        "validation": {"test_command": PYTEST, "dependency_audit": False},
        "observability": {"trace": True},
    }
    registry = None
    if suite == "harness":
        registry = ProviderRegistry(ModelsSettings())
        registry.register_instance("oracle", ScriptedProvider("oracle", by_role=case.oracle()))
        overrides["models"] = {"providers": {"oracle": {"type": "scripted"}}, "roles": {"default": ["oracle:transcript"]}}
    elif model:
        overrides["models"] = {"roles": {"default": [m.strip() for m in model.split(",")]}}
    started = time.monotonic()
    rt = Runtime.open(repo, overrides, registry=registry, use_global_config=suite != "harness")
    error = None
    status = TaskStatus.FAILED
    state: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    try:
        task = rt.create_task(case.prompt)
        result = await rt.run_task(task.id)
        status, error, state = result.status, result.error, result.state
        metrics = rt.metrics.snapshot(task.id)
    except Exception as exc:  # a crashing case is recorded, not fatal to the suite
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await rt.aclose()
    duration = time.monotonic() - started
    verified, detail = run_hidden_checks(case, repo, state, root / "hidden")
    completed = status in (TaskStatus.COMPLETED, TaskStatus.COMPLETED_UNVERIFIED)
    return CaseResult(
        id=case.id, category=case.category, title=case.title, suite=suite, status=str(status),
        completed=completed, verified=verified and completed, verification=detail, duration_s=round(duration, 2),
        model_calls=int(metrics.get("model_calls", 0)), input_tokens=int(metrics.get("input_tokens", 0)),
        output_tokens=int(metrics.get("output_tokens", 0)), tokens_estimated=bool(metrics.get("tokens_estimated")),
        tool_calls=int(metrics.get("tool_calls", 0)), tool_errors=int(metrics.get("tool_errors", 0)),
        retries=int(metrics.get("model_retries", 0)), fallbacks=int(metrics.get("model_fallbacks", 0)),
        test_runs=int(metrics.get("test_runs", 0)), fix_attempts=int(metrics.get("fix_attempts", 0)),
        recovered=int(metrics.get("fix_attempts", 0)) > 0 and verified,
        human_interventions=int(metrics.get("human_interventions", 0)),
        context_resets=int(metrics.get("context_resets", 0)), error=error,
    )


def summarize(results: list[CaseResult], suite: str, model: str | None) -> dict[str, Any]:
    n = len(results) or 1
    return {
        "suite": suite,
        "model": model if suite == "model" else "scripted oracle (no model)",
        "generated": utcnow_iso(),
        "cases": len(results),
        "completion_rate": round(sum(r.completed for r in results) / n, 3),
        "verified_completion_rate": round(sum(r.verified for r in results) / n, 3),
        "test_success_rate": round(sum(r.verified for r in results if r.category != "repository exploration") / max(1, sum(1 for r in results if r.category != "repository exploration")), 3),
        "total_retries": sum(r.retries for r in results),
        "total_fallbacks": sum(r.fallbacks for r in results),
        "failure_recoveries": sum(r.recovered for r in results),
        "human_interventions": sum(r.human_interventions for r in results),
        "mean_duration_s": round(sum(r.duration_s for r in results) / n, 2),
        "mean_tool_calls": round(sum(r.tool_calls for r in results) / n, 1),
        "mean_model_calls": round(sum(r.model_calls for r in results) / n, 1),
        "note": (
            "Harness suite: scripted transcripts exercise the agent infrastructure (tools, validation, repair, review, "
            "gates, git, reports) against hidden checks. It does NOT measure model capability."
            if suite == "harness" else "Model suite: results reflect the configured model(s) on these cases only."
        ),
    }


def render_markdown(summary: dict[str, Any], results: list[CaseResult]) -> str:
    lines = [
        f"# Benchmark results — {summary['suite']} suite",
        "",
        f"> {summary['note']}",
        "",
        f"- Model: {summary['model']}",
        f"- Generated: {summary['generated']}",
        f"- Completion rate: {summary['completion_rate']:.0%}; verified completion rate: {summary['verified_completion_rate']:.0%}",
        f"- Failure recoveries (fixed after a failed validation): {summary['failure_recoveries']}",
        f"- Retries: {summary['total_retries']}; fallbacks: {summary['total_fallbacks']}; human interventions: {summary['human_interventions']}",
        f"- Mean duration: {summary['mean_duration_s']}s; mean tool calls: {summary['mean_tool_calls']}; mean model calls: {summary['mean_model_calls']}",
        "",
        "| Case | Category | Status | Verified | Time (s) | Model calls | Tool calls | Test runs | Fix attempts | Verification |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.id} | {r.category} | {r.status} | {'yes' if r.verified else 'NO'} | {r.duration_s} | {r.model_calls} | "
            f"{r.tool_calls} | {r.test_runs} | {r.fix_attempts} | {r.verification.replace('|', '/')[:120]} |"
        )
    return "\n".join(lines) + "\n"


async def run_suite(suite: str, cases: list[BenchCase], out: Path, model: str | None = None) -> tuple[dict[str, Any], list[CaseResult]]:
    work = Path(tempfile.mkdtemp(prefix="aie-bench-"))
    results: list[CaseResult] = []
    try:
        for case in cases:
            print(f"[{case.id}] {case.title} ...", file=sys.stderr, flush=True)
            result = await run_case(case, suite, work, model)
            mark = "✔" if result.verified else "✘"
            print(f"  {mark} {result.status} verified={result.verified} ({result.duration_s}s) {result.verification}", file=sys.stderr, flush=True)
            results.append(result)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    summary = summarize(results, suite, model)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / f"{suite}-results.json", {"summary": summary, "results": [r.model_dump() for r in results]})
    (out / f"{suite}-results.md").write_text(render_markdown(summary, results), encoding="utf-8")
    return summary, results


def main(args: argparse.Namespace) -> int:
    cases = ALL_CASES
    if getattr(args, "only", None):
        wanted = {x.strip() for x in args.only.split(",")}
        cases = [c for c in cases if c.id in wanted]
    if args.bench_cmd == "list":
        for c in cases:
            print(f"{c.id:<14} {c.category:<32} {c.title}")
        return 0
    suite = args.suite
    if suite not in ("harness", "model"):
        print("suite must be 'harness' or 'model'", file=sys.stderr)
        return 64
    model = getattr(args, "model", None)
    if suite == "model" and not model and not os.environ.get("AIE_MODEL"):
        print("The model suite needs --model provider:model (or AIE_MODEL).", file=sys.stderr)
        return 64
    out = Path(args.out or "benchmark-results")
    summary, results = asyncio.run(run_suite(suite, cases, out, model))
    print(json.dumps(summary, indent=2))
    print(f"Results written to {out}/", file=sys.stderr)
    if suite == "harness":
        # the harness suite is a regression test of the infrastructure: everything must verify
        return 0 if all(r.verified for r in results) else 1
    return 0
