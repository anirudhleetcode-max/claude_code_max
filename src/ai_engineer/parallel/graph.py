"""Dependency-aware async job runner with concurrency limits, timeouts, resources and cancellation.

Jobs start once all their dependencies finished ``ok``; a job whose dependency did
not succeed is ``skipped``. A central scheduler admits a ready job only when a
concurrency slot is free *and* every named resource it needs is free, claiming
them all at once — so jobs sharing a resource never overlap, no slot is wasted
waiting on a lock, and lock-ordering deadlocks are impossible.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from ..core.cancel import CancellationToken
from ..core.errors import CancelledByUser

log = logging.getLogger(__name__)

T = TypeVar("T")

JobStatus = Literal["ok", "error", "timeout", "cancelled", "skipped"]

# How long to wait for cancelled jobs to unwind before abandoning them.
CANCEL_GRACE_S = 5.0


@dataclass
class Job:
    id: str
    fn: Callable[[], Awaitable[Any]]
    deps: set[str] = field(default_factory=set)
    timeout_s: float | None = None
    resources: set[str] = field(default_factory=set)  # exclusive named locks


@dataclass
class JobResult:
    id: str
    status: JobStatus
    value: Any = None
    error: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _find_cycle(jobs: dict[str, Job]) -> list[str] | None:
    """Return one dependency cycle as ``[a, b, ..., a]`` (each depends on the next), or None."""
    state: dict[str, int] = {}  # 1 = on the current path, 2 = fully explored
    for root in jobs:
        if root in state:
            continue
        path: list[str] = [root]
        iters = [iter(sorted(jobs[root].deps))]
        state[root] = 1
        while iters:
            dep = next(iters[-1], None)
            if dep is None:
                state[path.pop()] = 2
                iters.pop()
                continue
            if state.get(dep) == 1:
                return [*path[path.index(dep) :], dep]
            if dep not in state:
                state[dep] = 1
                path.append(dep)
                iters.append(iter(sorted(jobs[dep].deps)))
    return None


def validate_jobs(jobs: list[Job]) -> dict[str, Job]:
    """Check ids are unique, dependencies exist and there are no cycles (ValueError otherwise)."""
    by_id: dict[str, Job] = {}
    for job in jobs:
        if job.id in by_id:
            raise ValueError(f"duplicate job id {job.id!r}")
        by_id[job.id] = job
    for job in jobs:
        for dep in sorted(job.deps):
            if dep not in by_id:
                raise ValueError(f"job {job.id!r} depends on unknown job {dep!r}")
    cycle = _find_cycle(by_id)
    if cycle:
        raise ValueError(f"dependency cycle (each job depends on the next): {' -> '.join(cycle)}")
    return by_id


async def _await_with_timeout(awaitable: Awaitable[T], timeout_s: float | None) -> tuple[T | None, bool]:
    """Await ``awaitable``; returns (value, timed_out). Only our own deadline counts as a timeout."""
    if timeout_s is None:
        return await awaitable, False
    try:
        async with asyncio.timeout(timeout_s) as scope:
            return await awaitable, False
    except TimeoutError:
        if scope.expired():
            return None, True
        raise


async def _run_one(job: Job) -> JobResult:
    started = time.monotonic()
    try:
        value, timed_out = await _await_with_timeout(job.fn(), job.timeout_s)
    except CancelledByUser as exc:
        return JobResult(job.id, "cancelled", error=str(exc) or "cancelled", duration_s=time.monotonic() - started)
    except Exception as exc:
        return JobResult(job.id, "error", error=f"{type(exc).__name__}: {exc}", duration_s=time.monotonic() - started)
    elapsed = time.monotonic() - started
    if timed_out:
        return JobResult(job.id, "timeout", error=f"timed out after {job.timeout_s:g}s", duration_s=elapsed)
    return JobResult(job.id, "ok", value=value, duration_s=elapsed)


async def run_jobs(
    jobs: list[Job],
    *,
    max_concurrency: int = 4,
    cancel: CancellationToken | None = None,
    fail_fast: bool = False,
) -> dict[str, JobResult]:
    """Run ``jobs`` respecting dependencies; returns a result for every job, in input order."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    by_id = validate_jobs(jobs)
    results: dict[str, JobResult] = {}
    waiting: dict[str, set[str]] = {j.id: set(j.deps) for j in jobs}
    dependents: dict[str, list[str]] = {j.id: [] for j in jobs}
    for job in jobs:
        for dep in job.deps:
            dependents[dep].append(job.id)
    ready: list[str] = [j.id for j in jobs if not j.deps]
    busy: set[str] = set()
    running: dict[asyncio.Task[JobResult], str] = {}
    started_at: dict[str, float] = {}
    stop_reason = ""

    def skip_dependents(job_id: str) -> None:
        pending = [job_id]
        while pending:
            current = pending.pop()
            status = results[current].status
            for child in dependents[current]:
                if child in results:
                    continue
                results[child] = JobResult(child, "skipped", error=f"dependency {current!r} {_past(status)}")
                waiting.pop(child, None)
                if child in ready:
                    ready.remove(child)
                pending.append(child)

    def finish(result: JobResult) -> None:
        nonlocal stop_reason
        results[result.id] = result
        busy.difference_update(by_id[result.id].resources)
        if result.status != "ok":
            skip_dependents(result.id)
            if fail_fast and result.status in ("error", "timeout") and not stop_reason:
                stop_reason = f"fail_fast: job {result.id!r} {_past(result.status)}"
            return
        for child in dependents[result.id]:
            remaining = waiting.get(child)
            if remaining is None:
                continue
            remaining.discard(result.id)
            if not remaining:
                del waiting[child]
                ready.append(child)

    def launch() -> None:
        for job_id in list(ready):
            if len(running) >= max_concurrency:
                return
            job = by_id[job_id]
            if job.resources & busy:
                continue
            ready.remove(job_id)
            busy.update(job.resources)
            started_at[job_id] = time.monotonic()
            running[asyncio.ensure_future(_run_one(job))] = job_id

    cancel_waiter: asyncio.Future[str] | None = asyncio.ensure_future(cancel.wait()) if cancel is not None else None
    try:
        while True:
            if cancel is not None and cancel.cancelled and not stop_reason:
                stop_reason = f"cancelled: {cancel.reason or 'cancelled'}"
            if stop_reason:
                break
            launch()
            if not running:
                break
            waitables: set[asyncio.Future[Any]] = set(running)
            if cancel_waiter is not None:
                waitables.add(cancel_waiter)
            done, _ = await asyncio.wait(waitables, return_when=asyncio.FIRST_COMPLETED)
            for task in [t for t in running if t in done]:  # launch order keeps processing deterministic
                job_id = running.pop(task)
                if task.cancelled():
                    finish(JobResult(job_id, "cancelled", error="cancelled", duration_s=_since(started_at, job_id)))
                else:
                    finish(task.result())
    finally:
        if cancel_waiter is not None:
            cancel_waiter.cancel()
            await asyncio.wait([cancel_waiter])  # let it unregister from the token
        if running:
            reason = stop_reason or "cancelled"
            await _cancel_tasks(running)
            for task, job_id in running.items():
                if task.done() and not task.cancelled() and task.exception() is None:
                    results[job_id] = task.result()  # it finished before the cancellation landed
                else:
                    results[job_id] = JobResult(job_id, "cancelled", error=reason, duration_s=_since(started_at, job_id))
    for job in jobs:
        if job.id not in results:
            results[job.id] = JobResult(job.id, "cancelled", error=stop_reason or "cancelled")
    return {job.id: results[job.id] for job in jobs}


async def _cancel_tasks(running: dict[asyncio.Task[JobResult], str]) -> None:
    for task in running:
        task.cancel()
    _, pending = await asyncio.wait(running, timeout=CANCEL_GRACE_S)
    for task in pending:  # a job that ignores cancellation is abandoned rather than awaited forever
        log.warning("job %s did not stop within %.0fs of cancellation", running[task], CANCEL_GRACE_S)
    for task in running:
        if task.done() and not task.cancelled():
            with contextlib.suppress(BaseException):
                task.exception()  # mark retrieved so asyncio does not log it


def _since(started_at: dict[str, float], job_id: str) -> float:
    start = started_at.get(job_id)
    return time.monotonic() - start if start is not None else 0.0


def _past(status: str) -> str:
    return {"error": "failed", "timeout": "timed out", "cancelled": "was cancelled", "skipped": "was skipped"}.get(
        status, status
    )


class JobGraph:
    """Builder over :func:`run_jobs`."""

    def __init__(self) -> None:
        self.jobs: list[Job] = []

    def add(
        self,
        job_id: str,
        fn: Callable[[], Awaitable[Any]],
        *,
        deps: set[str] | None = None,
        timeout_s: float | None = None,
        resources: set[str] | None = None,
    ) -> Job:
        job = Job(job_id, fn, set(deps or ()), timeout_s, set(resources or ()))
        self.jobs.append(job)
        return job

    async def run(
        self, *, max_concurrency: int = 4, cancel: CancellationToken | None = None, fail_fast: bool = False
    ) -> dict[str, JobResult]:
        return await run_jobs(self.jobs, max_concurrency=max_concurrency, cancel=cancel, fail_fast=fail_fast)
