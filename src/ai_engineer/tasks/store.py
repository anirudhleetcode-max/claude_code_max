"""SQLite-backed persistent state: tasks, events, checkpoints, test history, failures, schedules, metrics.

All writes are small and synchronous; the store is safe to share between the
asyncio loop and worker threads (a lock serializes connection use).
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import StateError
from ..core.events import Event
from ..core.ids import new_id
from ..core.util import atomic_write_json, utcnow_iso

SCHEMA_VERSION = 1


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_UNVERIFIED = "COMPLETED_UNVERIFIED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.COMPLETED_UNVERIFIED, TaskStatus.FAILED, TaskStatus.CANCELLED}
RESUMABLE_STATUSES = {TaskStatus.INTERRUPTED, TaskStatus.BLOCKED, TaskStatus.PENDING, TaskStatus.QUEUED}


class Task(BaseModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    parent_id: str | None = None
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 50  # lower runs first
    depends_on: list[str] = Field(default_factory=list)
    mode: str | None = None
    stage: str | None = None
    checkpoint_id: str | None = None
    attempts: int = 0
    artifacts: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created: str = Field(default_factory=utcnow_iso)
    updated: str = Field(default_factory=utcnow_iso)
    started: str | None = None
    finished: str | None = None
    lease_owner: str | None = None
    lease_expires: float | None = None


class CheckpointRecord(BaseModel):
    id: str
    task_id: str | None = None
    subtask_id: str | None = None
    label: str
    kind: str  # "git" | "files"
    ref: str | None = None
    tree: str | None = None
    created: str = Field(default_factory=utcnow_iso)
    verified: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class Schedule(BaseModel):
    id: str = Field(default_factory=lambda: new_id("sched"))
    description: str
    interval_s: float
    next_run: float
    enabled: bool = True
    last_task_id: str | None = None
    created: str = Field(default_factory=utcnow_iso)
    mode: str | None = None


_JSON_FIELDS = {"depends_on", "artifacts", "verification", "state"}
_TASK_COLUMNS = list(Task.model_fields)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY, parent_id TEXT, title TEXT NOT NULL, description TEXT NOT NULL,
    status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 50, depends_on TEXT NOT NULL DEFAULT '[]',
    mode TEXT, stage TEXT, checkpoint_id TEXT, attempts INTEGER NOT NULL DEFAULT 0,
    artifacts TEXT NOT NULL DEFAULT '{}', verification TEXT NOT NULL DEFAULT '{}', state TEXT NOT NULL DEFAULT '{}',
    error TEXT, created TEXT, updated TEXT, started TEXT, finished TEXT, lease_owner TEXT, lease_expires REAL
);
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status, priority, created);
CREATE INDEX IF NOT EXISTS tasks_parent ON tasks(parent_id);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, ts TEXT, task_id TEXT, subtask_id TEXT, type TEXT, stage TEXT, level TEXT, message TEXT, data TEXT
);
CREATE INDEX IF NOT EXISTS events_task ON events(task_id, ts);
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY, task_id TEXT, subtask_id TEXT, label TEXT, kind TEXT, ref TEXT, tree TEXT,
    created TEXT, verified INTEGER, meta TEXT
);
CREATE INDEX IF NOT EXISTS checkpoints_task ON checkpoints(task_id, created);
CREATE TABLE IF NOT EXISTS test_runs (
    id TEXT PRIMARY KEY, task_id TEXT, subtask_id TEXT, ts TEXT, kind TEXT, command TEXT, status TEXT,
    passed INTEGER, failed INTEGER, errors INTEGER, duration_s REAL, signature TEXT, summary TEXT, classification TEXT
);
CREATE INDEX IF NOT EXISTS test_runs_task ON test_runs(task_id, ts);
CREATE TABLE IF NOT EXISTS failures (id TEXT PRIMARY KEY, task_id TEXT, subtask_id TEXT, ts TEXT, signature TEXT, record TEXT);
CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY, description TEXT, interval_s REAL, next_run REAL, enabled INTEGER, last_task_id TEXT, created TEXT, mode TEXT
);
CREATE TABLE IF NOT EXISTS metrics (task_id TEXT PRIMARY KEY, data TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS control (task_id TEXT PRIMARY KEY, request TEXT, ts TEXT);
"""


def lease_owner_id(session: str = "") -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{session}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class StateStore:
    def __init__(self, db_path: Path) -> None:
        self.path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._con = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30, isolation_level=None)
            self._con.row_factory = sqlite3.Row
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._con.execute("PRAGMA busy_timeout=30000")
            self._con.executescript(_SCHEMA)
        except sqlite3.DatabaseError as exc:
            raise StateError(f"cannot open state database {db_path}: {exc}") from exc
        with self._lock:
            row = self._con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self._con.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row[0]) > SCHEMA_VERSION:
                raise StateError(f"{db_path} was written by a newer version (schema {row[0]}); upgrade ai-engineer")

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._con.execute(sql, tuple(params))

    # ---- tasks ----------------------------------------------------------------------

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        data = dict(row)
        for key in _JSON_FIELDS:
            data[key] = json.loads(data[key]) if data.get(key) else ([] if key == "depends_on" else {})
        return Task.model_validate(data)

    def create_task(self, task: Task) -> Task:
        values = task.model_dump()
        for key in _JSON_FIELDS:
            values[key] = json.dumps(values[key], default=str)
        cols = ", ".join(_TASK_COLUMNS)
        marks = ", ".join("?" for _ in _TASK_COLUMNS)
        self._exec(f"INSERT INTO tasks ({cols}) VALUES ({marks})", [values[c] for c in _TASK_COLUMNS])  # noqa: S608 - column names are constants
        return task

    def get_task(self, task_id: str) -> Task | None:
        row = self._exec("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def require_task(self, task_id: str) -> Task:
        task = self.get_task(task_id)
        if task is None:
            # allow unique prefixes for convenience in the CLI
            rows = self._exec("SELECT id FROM tasks WHERE id LIKE ?", (task_id + "%",)).fetchall()
            if len(rows) == 1:
                task = self.get_task(rows[0][0])
        if task is None:
            raise StateError(f"no such task: {task_id}")
        return task

    def update_task(self, task_id: str, **fields: Any) -> Task:
        if not fields:
            return self.require_task(task_id)
        unknown = set(fields) - set(_TASK_COLUMNS)
        if unknown:
            raise StateError(f"unknown task fields: {sorted(unknown)}")
        fields["updated"] = utcnow_iso()
        assignments = []
        params: list[Any] = []
        for key, value in fields.items():
            if key in _JSON_FIELDS:
                value = json.dumps(value, default=str)
            elif isinstance(value, StrEnum):
                value = value.value
            assignments.append(f"{key} = ?")
            params.append(value)
        params.append(task_id)
        cur = self._exec(f"UPDATE tasks SET {', '.join(assignments)} WHERE id = ?", params)  # noqa: S608 - keys validated above
        if cur.rowcount == 0:
            raise StateError(f"no such task: {task_id}")
        return self.require_task(task_id)

    def list_tasks(self, *, status: Iterable[TaskStatus] | None = None, parent_id: str | None = None, top_level: bool = False, limit: int = 200) -> list[Task]:
        sql = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            statuses = [str(s) for s in status]
            sql += f" AND status IN ({', '.join('?' for _ in statuses)})"
            params += statuses
        if parent_id is not None:
            sql += " AND parent_id = ?"
            params.append(parent_id)
        elif top_level:
            sql += " AND parent_id IS NULL"
        sql += " ORDER BY created DESC LIMIT ?"
        params.append(limit)
        return [self._row_to_task(r) for r in self._exec(sql, params).fetchall()]

    def subtasks(self, parent_id: str) -> list[Task]:
        rows = self._exec("SELECT * FROM tasks WHERE parent_id = ? ORDER BY created, id", (parent_id,)).fetchall()
        return [self._row_to_task(r) for r in rows]

    def next_queued(self) -> Task | None:
        """Highest-priority queued top-level task whose dependencies are complete."""
        rows = self._exec(
            "SELECT * FROM tasks WHERE status = ? AND parent_id IS NULL ORDER BY priority, created", (TaskStatus.QUEUED.value,)
        ).fetchall()
        for row in rows:
            task = self._row_to_task(row)
            deps = [self.get_task(d) for d in task.depends_on]
            if all(d is not None and d.status in (TaskStatus.COMPLETED, TaskStatus.COMPLETED_UNVERIFIED) for d in deps):
                return task
        return None

    # ---- leases & recovery -----------------------------------------------------------------

    def acquire_lease(self, task_id: str, owner: str, ttl_s: float = 60.0) -> bool:
        now = time.time()
        cur = self._exec(
            "UPDATE tasks SET lease_owner = ?, lease_expires = ? WHERE id = ? AND (lease_owner IS NULL OR lease_owner = ? OR lease_expires < ?)",
            (owner, now + ttl_s, task_id, owner, now),
        )
        return cur.rowcount == 1

    def heartbeat(self, task_id: str, owner: str, ttl_s: float = 60.0) -> bool:
        cur = self._exec("UPDATE tasks SET lease_expires = ? WHERE id = ? AND lease_owner = ?", (time.time() + ttl_s, task_id, owner))
        return cur.rowcount == 1

    def release_lease(self, task_id: str, owner: str) -> None:
        self._exec("UPDATE tasks SET lease_owner = NULL, lease_expires = NULL WHERE id = ? AND lease_owner = ?", (task_id, owner))

    def recover_interrupted(self) -> list[Task]:
        """Mark RUNNING tasks whose owner is gone (expired lease or dead local pid) as INTERRUPTED."""
        recovered = []
        now = time.time()
        host = socket.gethostname()
        for task in self.list_tasks(status=[TaskStatus.RUNNING], limit=1000):
            dead = task.lease_expires is None or task.lease_expires < now
            if not dead and task.lease_owner:
                owner_host, _, rest = task.lease_owner.partition(":")
                pid_text = rest.split(":", 1)[0]
                if owner_host == host and pid_text.isdigit() and not _pid_alive(int(pid_text)):
                    dead = True
            if dead:
                recovered.append(
                    self.update_task(task.id, status=TaskStatus.INTERRUPTED, lease_owner=None, lease_expires=None,
                                     error=(task.error or "") + " [interrupted: agent process stopped unexpectedly]")
                )
        return recovered

    # ---- control (stop requests from other processes) -------------------------------------

    def request_control(self, task_id: str, request: str) -> None:
        self._exec("INSERT OR REPLACE INTO control(task_id, request, ts) VALUES (?, ?, ?)", (task_id, request, utcnow_iso()))

    def pop_control(self, task_id: str) -> str | None:
        with self._lock:
            row = self._con.execute("SELECT request FROM control WHERE task_id = ?", (task_id,)).fetchone()
            if row:
                self._con.execute("DELETE FROM control WHERE task_id = ?", (task_id,))
            return row[0] if row else None

    # ---- events ----------------------------------------------------------------------------

    def add_event(self, event: Event) -> None:
        if not event.task_id:
            return
        self._exec(
            "INSERT OR IGNORE INTO events(id, ts, task_id, subtask_id, type, stage, level, message, data) VALUES (?,?,?,?,?,?,?,?,?)",
            (event.id, event.ts, event.task_id, event.subtask_id, str(event.type), event.stage, event.level, event.message,
             json.dumps(event.data, default=str)[:20000]),
        )

    def events(self, task_id: str, limit: int = 500, types: Iterable[str] | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE task_id = ?"
        params: list[Any] = [task_id]
        if types:
            tlist = list(types)
            sql += f" AND type IN ({', '.join('?' for _ in tlist)})"
            params += tlist
        sql += " ORDER BY ts, id LIMIT ?"
        params.append(limit)
        out = []
        for row in self._exec(sql, params).fetchall():
            item = dict(row)
            item["data"] = json.loads(item["data"] or "{}")
            out.append(item)
        return out

    # ---- checkpoints -------------------------------------------------------------------------

    def add_checkpoint(self, cp: CheckpointRecord) -> CheckpointRecord:
        self._exec(
            "INSERT INTO checkpoints(id, task_id, subtask_id, label, kind, ref, tree, created, verified, meta) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cp.id, cp.task_id, cp.subtask_id, cp.label, cp.kind, cp.ref, cp.tree, cp.created, int(cp.verified), json.dumps(cp.meta)),
        )
        return cp

    def _row_to_cp(self, row: sqlite3.Row) -> CheckpointRecord:
        data = dict(row)
        data["verified"] = bool(data["verified"])
        data["meta"] = json.loads(data["meta"] or "{}")
        return CheckpointRecord.model_validate(data)

    def get_checkpoint(self, cp_id: str) -> CheckpointRecord | None:
        row = self._exec("SELECT * FROM checkpoints WHERE id = ?", (cp_id,)).fetchone()
        if row is None:
            rows = self._exec("SELECT * FROM checkpoints WHERE id LIKE ?", (cp_id + "%",)).fetchall()
            row = rows[0] if len(rows) == 1 else None
        return self._row_to_cp(row) if row else None

    def list_checkpoints(self, task_id: str | None = None, limit: int = 200) -> list[CheckpointRecord]:
        if task_id:
            rows = self._exec("SELECT * FROM checkpoints WHERE task_id = ? ORDER BY created, id LIMIT ?", (task_id, limit)).fetchall()
        else:
            rows = self._exec("SELECT * FROM checkpoints ORDER BY created DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_cp(r) for r in rows]

    def last_verified_checkpoint(self, task_id: str) -> CheckpointRecord | None:
        row = self._exec(
            "SELECT * FROM checkpoints WHERE task_id = ? AND verified = 1 ORDER BY created DESC, id DESC LIMIT 1", (task_id,)
        ).fetchone()
        return self._row_to_cp(row) if row else None

    # ---- test history and failures ---------------------------------------------------------------

    def add_test_run(self, task_id: str, subtask_id: str | None, result: Any) -> None:
        self._exec(
            "INSERT INTO test_runs(id, task_id, subtask_id, ts, kind, command, status, passed, failed, errors, duration_s, signature, summary, classification) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (new_id("run"), task_id, subtask_id, utcnow_iso(), str(result.kind), result.command, result.status, result.passed,
             result.failed, result.errors, result.duration_s, result.signature(), result.summary, result.classification),
        )

    def test_runs(self, task_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if task_id:
            rows = self._exec("SELECT * FROM test_runs WHERE task_id = ? ORDER BY ts LIMIT ?", (task_id, limit)).fetchall()
        else:
            rows = self._exec("SELECT * FROM test_runs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def add_failure(self, task_id: str, subtask_id: str | None, signature: str, record: dict[str, Any]) -> str:
        fid = record.get("id") or new_id("fail")
        self._exec(
            "INSERT OR REPLACE INTO failures(id, task_id, subtask_id, ts, signature, record) VALUES (?,?,?,?,?,?)",
            (fid, task_id, subtask_id, utcnow_iso(), signature, json.dumps(record, default=str)),
        )
        return str(fid)

    def failures(self, task_id: str) -> list[dict[str, Any]]:
        rows = self._exec("SELECT record FROM failures WHERE task_id = ? ORDER BY ts", (task_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    # ---- schedules ----------------------------------------------------------------------------------

    def add_schedule(self, sched: Schedule) -> Schedule:
        self._exec(
            "INSERT INTO schedules(id, description, interval_s, next_run, enabled, last_task_id, created, mode) VALUES (?,?,?,?,?,?,?,?)",
            (sched.id, sched.description, sched.interval_s, sched.next_run, int(sched.enabled), sched.last_task_id, sched.created, sched.mode),
        )
        return sched

    def schedules(self) -> list[Schedule]:
        rows = self._exec("SELECT * FROM schedules ORDER BY next_run").fetchall()
        return [Schedule.model_validate({**dict(r), "enabled": bool(r["enabled"])}) for r in rows]

    def due_schedules(self, now: float | None = None) -> list[Schedule]:
        now = time.time() if now is None else now
        return [s for s in self.schedules() if s.enabled and s.next_run <= now]

    def mark_schedule_run(self, sched_id: str, task_id: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        row = self._exec("SELECT interval_s FROM schedules WHERE id = ?", (sched_id,)).fetchone()
        if row is None:
            raise StateError(f"no such schedule {sched_id}")
        self._exec("UPDATE schedules SET next_run = ?, last_task_id = ? WHERE id = ?", (now + float(row[0]), task_id, sched_id))

    def set_schedule_enabled(self, sched_id: str, enabled: bool) -> bool:
        return self._exec("UPDATE schedules SET enabled = ? WHERE id = ?", (int(enabled), sched_id)).rowcount == 1

    def delete_schedule(self, sched_id: str) -> bool:
        return self._exec("DELETE FROM schedules WHERE id = ?", (sched_id,)).rowcount == 1

    # ---- metrics -------------------------------------------------------------------------------------

    def save_metrics(self, task_id: str, data: dict[str, Any]) -> None:
        self._exec(
            "INSERT OR REPLACE INTO metrics(task_id, data, updated) VALUES (?, ?, ?)", (task_id, json.dumps(data, default=str), utcnow_iso())
        )

    def get_metrics(self, task_id: str) -> dict[str, Any]:
        row = self._exec("SELECT data FROM metrics WHERE task_id = ?", (task_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    # ---- snapshots ----------------------------------------------------------------------------------

    def export_tasks_json(self, path: Path) -> None:
        tasks = []
        for task in self.list_tasks(limit=1000):
            data = task.model_dump(exclude={"state", "lease_owner", "lease_expires"})
            tasks.append(data)
        atomic_write_json(path, {"generated": utcnow_iso(), "tasks": tasks})
