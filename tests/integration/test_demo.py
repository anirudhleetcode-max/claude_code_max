"""The `aie demo` run: every step it claims to demonstrate actually happens."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ai_engineer.core.events import EventType
from ai_engineer.demo import TASK as DEMO_TASK
from ai_engineer.demo import create_demo_repo, demo_runtime
from ai_engineer.tasks.store import TaskStatus


async def test_demo_runs_the_whole_pipeline(tmp_path: Path) -> None:
    root = create_demo_repo(tmp_path / "shop")
    rt = demo_runtime(root)
    events: list = []
    rt.bus.subscribe(events.append)
    try:
        task = rt.create_task(DEMO_TASK, title="demo")
        result = await rt.run_task(task.id)
        checkpoints = rt.store.list_checkpoints(task.id)
    finally:
        await rt.aclose()
    state = result.state
    types = [e.type for e in events]
    assert result.status == TaskStatus.COMPLETED, result.error
    # 1 discovery, 2 planning, 3 implementation, 4 testing
    assert EventType.UNDERSTANDING_CREATED in types and EventType.PLAN_CREATED in types
    assert [s["status"] for s in state["subtasks"].values()] == ["completed", "completed"]
    assert EventType.FILE_CHANGED in types and EventType.TEST_STARTED in types
    # 5 intentional failure, 6 diagnosis, 7 repair, 8 re-test
    assert state["subtasks"]["s1"]["repair_iterations"] == 1
    assert state["subtasks"]["s1"]["failures"][0]["result"] == "fixed"
    assert EventType.TEST_FAILED in types and EventType.FAILURE_RECORDED in types and EventType.FIX_ATTEMPTED in types
    first_fail = types.index(EventType.FIX_ATTEMPTED)
    assert EventType.TEST_PASSED in types[first_fail:]
    # 9 review, 10 checkpoint, 11 final verification, 12 report
    assert types.count(EventType.REVIEW_COMPLETED) >= 3
    assert len(checkpoints) >= 5
    gates = {g["name"]: g["status"] for g in state["gates"]["results"]}
    assert gates["tests"] == gates["review"] == gates["implementation"] == gates["git_state"] == "PASSED"
    report = Path(state["report_path"]).read_text()
    assert "## Quality gates" in report and "## Failures and fixes" in report and "COMPLETED" in report
    # the work is really there and really committed
    stock = (root / "inventory" / "stock.py").read_text()
    assert "qty > self.count(sku)" in stock and "def total_units" in stock
    log = subprocess.run(["git", "log", "--format=%s"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    assert log[:2] == ["aie: Add total_units() with a test", "aie: Reject removing more stock than available"]


def test_demo_cli(tmp_path: Path, capsys) -> None:
    from ai_engineer.ui.cli import main

    code = main(["demo", "--dir", str(tmp_path / "shop")])
    out = capsys.readouterr().out
    assert code == 0 and "COMPLETED" in out and "repair iterations 1" in out
    assert main(["demo", "--dir", str(tmp_path / "shop")]) == 64  # refuses a non-empty directory
