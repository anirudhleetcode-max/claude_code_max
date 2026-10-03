from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ai_engineer.core.errors import StateError
from ai_engineer.core.events import EventBus, EventType
from ai_engineer.git.repo import GitRepo
from ai_engineer.tasks.checkpoints import CheckpointManager
from ai_engineer.tasks.store import Schedule, StateStore, Task, TaskStatus
from ai_engineer.tools.file_state import FileStateTracker


def test_task_crud_and_prefix_lookup(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.db")
    t = store.create_task(Task(title="t", description="do x"))
    assert store.require_task(t.id[:12]).id == t.id
    t2 = store.update_task(t.id, status=TaskStatus.RUNNING, state={"stage": "plan"}, artifacts={"a": 1})
    assert t2.status == TaskStatus.RUNNING and t2.state == {"stage": "plan"}
    with pytest.raises(StateError):
        store.update_task(t.id, nonsense=1)
    with pytest.raises(StateError):
        store.require_task("task_missing")
    store.close()
    # persistence across reopen
    store = StateStore(tmp_path / "s.db")
    assert store.require_task(t.id).artifacts == {"a": 1}


def test_queue_respects_priority_and_dependencies(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.db")
    a = store.create_task(Task(title="a", description="a", status=TaskStatus.QUEUED, priority=10))
    b = store.create_task(Task(title="b", description="b", status=TaskStatus.QUEUED, priority=1, depends_on=[a.id]))
    c = store.create_task(Task(title="c", description="c", status=TaskStatus.QUEUED, priority=20))
    assert store.next_queued().id == a.id  # b has higher priority but depends on a
    store.update_task(a.id, status=TaskStatus.COMPLETED)
    assert store.next_queued().id == b.id
    store.update_task(b.id, status=TaskStatus.FAILED)
    assert store.next_queued().id == c.id


def test_leases_and_crash_recovery(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.db")
    t = store.create_task(Task(title="t", description="d"))
    assert store.acquire_lease(t.id, "host:1:a", ttl_s=60)
    assert not store.acquire_lease(t.id, "host:2:b", ttl_s=60)  # held by someone else
    store.update_task(t.id, status=TaskStatus.RUNNING)
    assert store.recover_interrupted() == []  # lease alive
    store.update_task(t.id, lease_expires=time.time() - 1)
    recovered = store.recover_interrupted()
    assert [r.id for r in recovered] == [t.id]
    assert store.require_task(t.id).status == TaskStatus.INTERRUPTED
    assert store.acquire_lease(t.id, "host:2:b")


def test_recovery_detects_dead_local_pid(tmp_path: Path) -> None:
    import socket

    store = StateStore(tmp_path / "s.db")
    t = store.create_task(Task(title="t", description="d", status=TaskStatus.RUNNING))
    # a pid that certainly does not exist, lease not yet expired
    store.update_task(t.id, lease_owner=f"{socket.gethostname()}:999999999:x", lease_expires=time.time() + 600)
    assert [r.id for r in store.recover_interrupted()] == [t.id]


def test_control_requests_schedules_metrics_events(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.db")
    t = store.create_task(Task(title="t", description="d"))
    store.request_control(t.id, "stop")
    assert store.pop_control(t.id) == "stop" and store.pop_control(t.id) is None
    s = store.add_schedule(Schedule(description="audit deps", interval_s=60, next_run=0))
    assert [x.id for x in store.due_schedules(now=1)] == [s.id]
    store.mark_schedule_run(s.id, t.id, now=100)
    assert store.due_schedules(now=120) == [] and store.due_schedules(now=161)
    store.save_metrics(t.id, {"x": 1})
    assert store.get_metrics(t.id) == {"x": 1}
    bus = EventBus()
    bus.subscribe(store.add_event)
    bus.emit(EventType.INFO, "hello", task_id=t.id)
    bus.emit(EventType.INFO, "no task")  # ignored
    assert [e["message"] for e in store.events(t.id)] == ["hello"]
    store.export_tasks_json(tmp_path / "tasks.json")
    assert "audit" not in (tmp_path / "tasks.json").read_text() and t.id in (tmp_path / "tasks.json").read_text()


async def test_file_checkpoints_restore_multiple_epochs(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("a0")
    (ws / "b.txt").write_text("b0")
    store = StateStore(tmp_path / "s.db")
    files = FileStateTracker(ws)
    mgr = CheckpointManager(ws, store, files, tmp_path / "state")

    def write(rel: str, text: str) -> None:
        p = ws / rel
        files.before_write(p)
        p.write_text(text)
        files.after_write(p)

    cp1 = await mgr.create("one")
    write("a.txt", "a1")
    write("new.txt", "n1")
    cp2 = await mgr.create("two")
    write("a.txt", "a2")
    write("b.txt", "b2")
    assert sorted(await mgr.changed_files_since(cp1.id)) == ["a.txt", "b.txt", "new.txt"]
    assert sorted(await mgr.changed_files_since(cp2.id)) == ["a.txt", "b.txt"]
    diff = await mgr.diff_since(cp1.id)
    assert "-a0" in diff and "+a2" in diff
    safety, touched = await mgr.restore(cp1.id)
    assert (ws / "a.txt").read_text() == "a0" and (ws / "b.txt").read_text() == "b0"
    assert not (ws / "new.txt").exists()
    assert "new.txt" in touched
    # the restore itself is reversible
    await mgr.restore(safety.id)
    assert (ws / "a.txt").read_text() == "a2" and (ws / "new.txt").read_text() == "n1"


async def test_git_checkpoints_track_untracked_and_restore(git_repo: Path, tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.db")
    files = FileStateTracker(git_repo)
    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    mgr = CheckpointManager(git_repo, store, files, tmp_path / "state", GitRepo(git_repo), bus)
    cp = await mgr.create("start", task_id="t1", verified=True)
    (git_repo / "README.md").write_text("changed\n")
    (git_repo / "pkg").mkdir()
    (git_repo / "pkg" / "new.py").write_text("x = 1\n")
    assert sorted(await mgr.changed_files_since(cp.id)) == ["README.md", "pkg/new.py"]
    assert "pkg/new.py" in await mgr.diff_since(cp.id, stat=True)
    await mgr.restore(cp.id)
    assert (git_repo / "README.md").read_text() == "# demo\n"
    assert not (git_repo / "pkg").exists()
    assert store.last_verified_checkpoint("t1").id == cp.id
    assert {e.type for e in events} >= {EventType.CHECKPOINT_CREATED, EventType.CHECKPOINT_RESTORED}


@pytest.mark.skipif(os.name == "nt", reason="permission semantics differ on Windows")
def test_newer_schema_is_refused(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "s.db")
    store._exec("UPDATE meta SET value='999' WHERE key='schema_version'")
    store.close()
    with pytest.raises(StateError):
        StateStore(tmp_path / "s.db")
