from __future__ import annotations

import asyncio
import time
from collections import defaultdict

import pytest

from ai_engineer.core.cancel import CancellationToken
from ai_engineer.core.errors import CancelledByUser
from ai_engineer.parallel import Job, JobGraph, JobResult, run_jobs, validate_jobs

GUARD_S = 10.0  # no test may hang: every run is bounded


async def run(jobs: list[Job], **kw) -> dict[str, JobResult]:
    return await asyncio.wait_for(run_jobs(jobs, **kw), GUARD_S)


def value(v, delay: float = 0.0):
    async def fn():
        if delay:
            await asyncio.sleep(delay)
        return v

    return fn


def raises(exc: BaseException, delay: float = 0.0):
    async def fn():
        if delay:
            await asyncio.sleep(delay)
        raise exc

    return fn


class Tracker:
    """Records overlapping execution, overall and per resource."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.by_resource: dict[str, int] = defaultdict(int)
        self.resource_peak: dict[str, int] = defaultdict(int)
        self.order: list[str] = []
        self.finished: list[str] = []

    def job(self, name: str, delay: float = 0.05, resources: tuple[str, ...] = ()):
        async def fn():
            self.order.append(name)
            self.active += 1
            self.peak = max(self.peak, self.active)
            for r in resources:
                self.by_resource[r] += 1
                self.resource_peak[r] = max(self.resource_peak[r], self.by_resource[r])
            try:
                await asyncio.sleep(delay)
            finally:
                self.active -= 1
                for r in resources:
                    self.by_resource[r] -= 1
            self.finished.append(name)
            return name

        return fn


# ---- validation ---------------------------------------------------------------------------------


async def test_empty_and_basic_results_in_input_order() -> None:
    assert await run([]) == {}
    results = await run([Job("b", value(2, 0.02)), Job("a", value(1)), Job("c", value(None))])
    assert list(results) == ["b", "a", "c"]
    assert {k: r.status for k, r in results.items()} == {"b": "ok", "a": "ok", "c": "ok"}
    assert results["b"].value == 2 and results["a"].value == 1 and results["c"].value is None
    assert results["b"].ok and results["b"].duration_s >= 0.015


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="duplicate job id 'a'"):
        validate_jobs([Job("a", value(1)), Job("a", value(2))])
    with pytest.raises(ValueError, match="'a' depends on unknown job 'zzz'"):
        validate_jobs([Job("a", value(1), deps={"zzz"})])
    with pytest.raises(ValueError, match=r"cycle.*a -> a"):
        validate_jobs([Job("a", value(1), deps={"a"})])


async def test_cycle_is_named() -> None:
    jobs = [
        Job("start", value(0)),
        Job("a", value(1), deps={"start", "c"}),
        Job("b", value(2), deps={"a"}),
        Job("c", value(3), deps={"b"}),
    ]
    with pytest.raises(ValueError, match="dependency cycle") as info:
        await run(jobs)
    message = str(info.value)
    assert all(name in message for name in ("a", "b", "c")) and "start" not in message.split(":", 1)[1]
    with pytest.raises(ValueError, match="max_concurrency"):
        await run([Job("a", value(1))], max_concurrency=0)


def test_deep_chain_validates_without_recursion_limit() -> None:
    jobs = [Job("j0", value(0))] + [Job(f"j{i}", value(i), deps={f"j{i - 1}"}) for i in range(1, 3000)]
    assert len(validate_jobs(jobs)) == 3000


# ---- concurrency ------------------------------------------------------------------------------------


async def test_concurrency_is_real() -> None:
    n = 6
    started = time.monotonic()
    results = await run([Job(f"j{i}", value(i, 0.2)) for i in range(n)], max_concurrency=n)
    elapsed = time.monotonic() - started
    assert all(r.status == "ok" for r in results.values())
    assert elapsed < n * 0.2 / 2, elapsed


async def test_max_concurrency_is_respected() -> None:
    t = Tracker()
    results = await run([Job(f"j{i}", t.job(f"j{i}", 0.03)) for i in range(8)], max_concurrency=2)
    assert all(r.ok for r in results.values())
    assert t.peak == 2
    assert t.order == [f"j{i}" for i in range(8)]  # ready jobs start in input order


async def test_resources_serialize_but_others_overlap() -> None:
    t = Tracker()
    jobs = [Job(f"db{i}", t.job(f"db{i}", 0.05, ("db",)), resources={"db"}) for i in range(4)]
    jobs += [Job(f"free{i}", t.job(f"free{i}", 0.05)) for i in range(3)]
    started = time.monotonic()
    results = await run(jobs, max_concurrency=8)
    elapsed = time.monotonic() - started
    assert all(r.ok for r in results.values())
    assert t.resource_peak["db"] == 1
    assert t.peak >= 4  # one db job plus the three free jobs ran together
    assert elapsed >= 4 * 0.05 * 0.9  # the db jobs ran one after another


async def test_deadlock_prone_resource_sets_complete() -> None:
    t = Tracker()
    jobs = [
        Job("xy", t.job("xy", 0.03, ("x", "y")), resources={"x", "y"}),
        Job("yx", t.job("yx", 0.03, ("y", "x")), resources={"y", "x"}),
        Job("yz", t.job("yz", 0.03, ("y", "z")), resources={"y", "z"}),
        Job("zx", t.job("zx", 0.03, ("z", "x")), resources={"z", "x"}),
        Job("x", t.job("x", 0.03, ("x",)), resources={"x"}),
        Job("z", t.job("z", 0.03, ("z",)), resources={"z"}),
    ]
    results = await run(jobs, max_concurrency=6)
    assert all(r.ok for r in results.values())
    assert max(t.resource_peak.values()) == 1


async def test_blocked_resource_does_not_starve_other_ready_jobs() -> None:
    t = Tracker()
    jobs = [
        Job("long", t.job("long", 0.2, ("r",)), resources={"r"}),
        Job("waits", t.job("waits", 0.01, ("r",)), resources={"r"}),
        Job("free", t.job("free", 0.01)),
    ]
    await run(jobs, max_concurrency=2)
    assert t.finished.index("free") < t.finished.index("long") < t.finished.index("waits")


# ---- dependencies -------------------------------------------------------------------------------


async def test_diamond_dependencies() -> None:
    t = Tracker()
    jobs = [
        Job("d", t.job("d", 0.01), deps={"b", "c"}),
        Job("b", t.job("b", 0.05), deps={"a"}),
        Job("c", t.job("c", 0.02), deps={"a"}),
        Job("a", t.job("a", 0.02)),
    ]
    results = await run(jobs, max_concurrency=4)
    assert all(r.ok for r in results.values())
    assert t.order[0] == "a" and t.order[-1] == "d"
    assert set(t.order[1:3]) == {"b", "c"}
    assert t.peak == 2  # b and c overlapped


async def test_failed_dependency_skips_dependents_transitively() -> None:
    ran: list[str] = []

    def mark(name: str):
        async def fn():
            ran.append(name)
            return name

        return fn

    jobs = [
        Job("bad", raises(ValueError("boom"))),
        Job("child", mark("child"), deps={"bad"}),
        Job("grandchild", mark("grandchild"), deps={"child", "ok"}),
        Job("ok", mark("ok")),
        Job("sibling", mark("sibling"), deps={"ok"}),
    ]
    results = await run(jobs)
    assert results["bad"].status == "error" and results["bad"].error == "ValueError: boom"
    assert results["child"].status == "skipped" and "'bad'" in results["child"].error
    assert results["grandchild"].status == "skipped" and "'child'" in results["grandchild"].error
    assert results["ok"].ok and results["sibling"].ok
    assert sorted(ran) == ["ok", "sibling"]


async def test_dependency_timeout_skips_dependent() -> None:
    results = await run([Job("slow", value(1, 5), timeout_s=0.05), Job("after", value(2), deps={"slow"})])
    assert results["slow"].status == "timeout" and "0.05" in results["slow"].error
    assert results["after"].status == "skipped" and "timed out" in results["after"].error


# ---- failures, timeouts -------------------------------------------------------------------------


async def test_errors_and_timeouts_are_isolated() -> None:
    async def own_timeout():
        raise TimeoutError("upstream deadline")

    def not_async():
        return 42

    jobs = [
        Job("err", raises(KeyError("k"))),
        Job("slow", value(1, 5), timeout_s=0.1),
        Job("own_timeout", own_timeout, timeout_s=5),
        Job("sync", not_async),  # type: ignore[arg-type]
        Job("fine", value("done", 0.05), timeout_s=5),
    ]
    started = time.monotonic()
    results = await run(jobs)
    assert time.monotonic() - started < 2
    assert results["err"].status == "error" and results["err"].error == "KeyError: 'k'"
    assert results["slow"].status == "timeout" and results["slow"].duration_s < 1
    assert results["own_timeout"].status == "error" and "TimeoutError" in results["own_timeout"].error
    assert results["sync"].status == "error" and "TypeError" in results["sync"].error
    assert results["fine"].status == "ok" and results["fine"].value == "done"


async def test_job_raising_cancelled_by_user_is_cancelled() -> None:
    results = await run([Job("a", raises(CancelledByUser("stop requested"))), Job("b", value(1), deps={"a"})])
    assert results["a"].status == "cancelled" and results["a"].error == "stop requested"
    assert results["b"].status == "skipped"


async def test_fail_fast_cancels_the_rest() -> None:
    cleaned: list[str] = []

    async def long_job():
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.append("long")

    jobs = [
        Job("bad", raises(RuntimeError("nope"), 0.05)),
        Job("long", long_job),
        Job("queued", value(1)),
        Job("dependent", value(2), deps={"bad"}),
    ]
    started = time.monotonic()
    results = await run(jobs, max_concurrency=2, fail_fast=True)
    assert time.monotonic() - started < 2
    assert results["bad"].status == "error"
    assert results["long"].status == "cancelled" and "fail_fast" in results["long"].error
    assert results["queued"].status == "cancelled"  # never got a slot before the failure
    assert results["dependent"].status == "skipped"
    assert cleaned == ["long"]


async def test_without_fail_fast_other_jobs_continue() -> None:
    results = await run([Job("bad", raises(RuntimeError("x"))), Job("slow", value(1, 0.1))], max_concurrency=1)
    assert results["bad"].status == "error" and results["slow"].status == "ok"


async def test_fail_fast_on_timeout() -> None:
    results = await run(
        [Job("t", value(1, 5), timeout_s=0.05), Job("next", value(2, 0.01), deps=set()), Job("later", value(3))],
        max_concurrency=1,
        fail_fast=True,
    )
    assert results["t"].status == "timeout"
    assert results["next"].status == "cancelled" and results["later"].status == "cancelled"


# ---- cancellation -------------------------------------------------------------------------------


async def test_cancellation_mid_run() -> None:
    token = CancellationToken()
    cleaned: list[str] = []

    def sleeper(name: str, delay: float):
        async def fn():
            try:
                await asyncio.sleep(delay)
            finally:
                cleaned.append(name)
            return name

        return fn

    jobs = [
        Job("quick", sleeper("quick", 0.01)),
        Job("slow1", sleeper("slow1", 10)),
        Job("slow2", sleeper("slow2", 10)),
        Job("queued", sleeper("queued", 10)),
        Job("dependent", sleeper("dependent", 0), deps={"slow1"}),
    ]
    asyncio.get_running_loop().call_later(0.15, token.cancel, "user pressed stop")
    started = time.monotonic()
    results = await run(jobs, max_concurrency=3, cancel=token)
    assert time.monotonic() - started < 2
    assert results["quick"].status == "ok"
    assert results["slow1"].status == "cancelled" and "user pressed stop" in results["slow1"].error
    assert results["slow2"].status == "cancelled"
    assert results["queued"].status == "cancelled"  # started after quick finished, then cancelled
    assert results["dependent"].status == "cancelled" and results["dependent"].duration_s == 0
    assert set(cleaned) == {"quick", "slow1", "slow2", "queued"}
    assert not token._waiters  # the runner unregistered from the token


async def test_pre_cancelled_token_runs_nothing() -> None:
    token = CancellationToken()
    token.cancel("too late")
    ran: list[str] = []

    async def fn():
        ran.append("x")

    results = await run([Job("a", fn), Job("b", fn, deps={"a"})], cancel=token)
    assert {r.status for r in results.values()} == {"cancelled"}
    assert "too late" in results["a"].error and ran == []


async def test_outer_task_cancellation_cancels_jobs() -> None:
    cleaned: list[str] = []

    async def long_job():
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.append("long")

    task = asyncio.ensure_future(run_jobs([Job("a", long_job), Job("b", long_job)]))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, GUARD_S)
    assert cleaned == ["long", "long"]


async def test_job_graph_builder() -> None:
    graph = JobGraph()
    graph.add("fetch", value("data"))
    graph.add("parse", value("ast"), deps={"fetch"}, timeout_s=1, resources={"cpu"})
    results = await asyncio.wait_for(graph.run(max_concurrency=2), GUARD_S)
    assert [r.status for r in results.values()] == ["ok", "ok"]
    assert graph.jobs[1].resources == {"cpu"}
