from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from ai_engineer.config.settings import Mode, PermissionLevel, Settings
from ai_engineer.core.cancel import CancellationToken
from ai_engineer.core.errors import CancelledByUser
from ai_engineer.core.events import EventBus, EventType
from ai_engineer.core.types import ToolUseBlock
from ai_engineer.git.repo import GitRepo
from ai_engineer.security.command_risk import Risk
from ai_engineer.security.secrets import Redactor
from ai_engineer.tools.approval import (
    AllowAllBroker,
    ApprovalDecision,
    CallbackBroker,
    DenyAllBroker,
    QueueBroker,
)
from ai_engineer.tools.base import ActionAssessment, Tool, ToolInput, ToolResult
from ai_engineer.tools.executor import ToolExecutor, ToolRegistry
from ai_engineer.tools.factory import default_tools, make_context
from ai_engineer.tools.permissions import Decision, PermissionPolicy
from ai_engineer.tools.process import build_env

FAKE_TOKEN = "ghp_" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0Pp9Oo8"


def settings_for(mode: Mode = Mode.DEVELOPER, **perm) -> Settings:
    s = Settings()
    s.permissions.mode = mode
    for k, v in perm.items():
        setattr(s.permissions, k, v)
    return s


def harness(tmp_path: Path, mode: Mode = Mode.DEVELOPER, broker=None, **perm):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    settings = settings_for(mode, **perm)
    bus = EventBus()
    events: list = []
    bus.subscribe(events.append)
    ctx = make_context(ws, settings, bus=bus, redactor=Redactor(environ={}))
    executor = ToolExecutor(ToolRegistry(default_tools()), PermissionPolicy(settings.permissions), broker or DenyAllBroker(bus), bus=bus)
    return ws, ctx, executor, events


def call(name: str, **args) -> ToolUseBlock:
    return ToolUseBlock(id=f"t_{name}", name=name, input=args)


# ---- policy ---------------------------------------------------------------------------------


def test_policy_matrix() -> None:
    def decide(mode: Mode, level: PermissionLevel, risk: Risk | None = None, read_only: bool = True, command: str | None = None, **perm):
        policy = PermissionPolicy(settings_for(mode, **perm).permissions)
        return policy.evaluate(ActionAssessment(level=level, summary="x", risk=risk, read_only=read_only, command=command)).decision

    assert decide(Mode.SAFE, PermissionLevel.READ_ONLY) == Decision.ALLOW
    assert decide(Mode.SAFE, PermissionLevel.SAFE_WRITE) == Decision.DENY
    assert decide(Mode.SAFE, PermissionLevel.READ_ONLY, Risk.LOW, read_only=False, command="pytest") == Decision.DENY
    assert decide(Mode.DEVELOPER, PermissionLevel.SAFE_WRITE) == Decision.ALLOW
    assert decide(Mode.DEVELOPER, PermissionLevel.READ_ONLY, Risk.MEDIUM, False, "pip install x") == Decision.ALLOW
    assert decide(Mode.DEVELOPER, PermissionLevel.READ_ONLY, Risk.HIGH, False, "git push") == Decision.ASK
    assert decide(Mode.ASSISTED, PermissionLevel.SAFE_WRITE) == Decision.ASK
    assert decide(Mode.ASSISTED, PermissionLevel.READ_ONLY) == Decision.ALLOW
    assert decide(Mode.AUTONOMOUS, PermissionLevel.READ_ONLY, Risk.CRITICAL, False, "rm -rf /") == Decision.DENY
    # allow-list pre-approves high risk but never critical
    assert decide(Mode.DEVELOPER, PermissionLevel.READ_ONLY, Risk.HIGH, False, "git push origin main", allow_commands=["git push origin *"]) == Decision.ALLOW
    assert decide(Mode.DEVELOPER, PermissionLevel.READ_ONLY, Risk.CRITICAL, False, "rm -rf /", allow_commands=["*"]) == Decision.DENY
    assert decide(Mode.DEVELOPER, PermissionLevel.READ_ONLY, Risk.LOW, True, "ls", deny_commands=["ls*"]) == Decision.DENY
    # a lowered ceiling denies instead of asking
    assert decide(Mode.DEVELOPER, PermissionLevel.READ_ONLY, Risk.HIGH, False, "git push", max_level=PermissionLevel.DEVELOPMENT) == Decision.DENY


# ---- executor -------------------------------------------------------------------------------


async def test_unknown_tool_and_invalid_args(tmp_path: Path) -> None:
    _, ctx, ex, _ = harness(tmp_path)
    r = await ex.execute(call("read_flie", path="a"), ctx)
    assert r.is_error and "read_file" in r.content
    r = await ex.execute(call("read_file", nope=1), ctx)
    assert r.is_error and "invalid arguments" in r.content
    r = await ex.execute(ToolUseBlock(id="x", name="read_file", input={"path": "a"}), ctx, allowed={"list_directory"})
    assert r.is_error and "unavailable" in r.content


async def test_denied_and_approved_flows(tmp_path: Path) -> None:
    ws, ctx, ex, events = harness(tmp_path)
    r = await ex.execute(call("run_command", command="git push origin main"), ctx)
    assert r.is_error and "not approved" in r.content
    assert any(e.type == EventType.TOOL_DENIED for e in events)
    assert ex.stats.denied == 1

    seen = []

    async def approve(req):
        seen.append(req)
        return ApprovalDecision(approved=True, reason="ok", remember=True)

    ex.approvals = CallbackBroker(approve)
    r = await ex.execute(call("run_command", command="echo pushed >/dev/null; rm -r tmpdir || true"), ctx)
    assert not r.is_error, r.content
    assert seen and seen[0].risk == "HIGH"
    # remembered: identical request is not asked again
    await ex.execute(call("run_command", command="echo pushed >/dev/null; rm -r tmpdir || true"), ctx)
    assert len(seen) == 1


async def test_safe_mode_blocks_writes_and_execution(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path, mode=Mode.SAFE)
    (ws / "a.txt").write_text("hello\n")
    assert not (await ex.execute(call("read_file", path="a.txt"), ctx)).is_error
    assert (await ex.execute(call("write_file", path="b.txt", content="x"), ctx)).is_error
    assert (await ex.execute(call("run_command", command="python -c 'print(1)'"), ctx)).is_error
    assert not (await ex.execute(call("run_command", command="ls"), ctx)).is_error
    assert not (ws / "b.txt").exists()


async def test_tool_timeout_and_crash_are_reported(tmp_path: Path) -> None:
    class Slow(Tool):
        name = "slow"
        description = "slow"
        Input = ToolInput
        timeout_s = 0.05

        async def run(self, args, ctx):
            await asyncio.sleep(5)
            return ToolResult()

    class Boom(Tool):
        name = "boom"
        description = "boom"
        Input = ToolInput

        async def run(self, args, ctx):
            raise RuntimeError("kaboom")

    _, ctx, _, _ = harness(tmp_path)
    ex = ToolExecutor(ToolRegistry([Slow(), Boom()]), PermissionPolicy(ctx.settings.permissions))
    r = await ex.execute(call("slow"), ctx)
    assert r.is_error and "timed out" in r.content
    r = await ex.execute(call("boom"), ctx)
    assert r.is_error and "kaboom" in r.content


async def test_cancellation_propagates(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    ctx.cancel = CancellationToken()
    task = asyncio.create_task(ex.execute(call("run_command", command="sleep 30"), ctx))
    await asyncio.sleep(0.3)
    ctx.cancel.cancel("user stop")
    with pytest.raises(CancelledByUser):
        await asyncio.wait_for(task, 10)


async def test_output_is_redacted_and_truncated(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    (ws / "notes.txt").write_text(f"token={FAKE_TOKEN}\n" + "x" * 50000 + "\nEND\n")
    ctx.settings.agent.tool_result_max_chars = 2000
    r = await ex.execute(call("read_file", path="notes.txt"), ctx)
    assert FAKE_TOKEN not in r.content
    assert "END" in r.content and "omitted" in r.content


async def test_read_only_calls_run_concurrently(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    for i in range(3):
        (ws / f"f{i}.txt").write_text(str(i))
    results = await ex.execute_many([call("read_file", path=f"f{i}.txt") for i in range(3)], ctx)
    assert [r.content.split("\t")[-1] for r in results] == ["0", "1", "2"]


# ---- file tools -----------------------------------------------------------------------------


async def test_write_requires_read_and_detects_external_change(tmp_path: Path) -> None:
    ws, ctx, ex, events = harness(tmp_path)
    (ws / "a.py").write_text("x = 1\n")
    r = await ex.execute(call("write_file", path="a.py", content="x = 2\n"), ctx)
    assert r.is_error and "has not been read" in r.content
    await ex.execute(call("read_file", path="a.py"), ctx)
    (ws / "a.py").write_text("x = 99\n")  # someone else edits the file
    r = await ex.execute(call("write_file", path="a.py", content="x = 2\n"), ctx)
    assert r.is_error and "changed on disk" in r.content
    await ex.execute(call("read_file", path="a.py"), ctx)
    r = await ex.execute(call("write_file", path="a.py", content="x = 2\n"), ctx)
    assert not r.is_error
    assert (ws / "a.py").read_text() == "x = 2\n"
    r = await ex.execute(call("write_file", path="pkg/new.py", content="y = 1\n"), ctx)
    assert not r.is_error and (ws / "pkg" / "new.py").exists()
    assert ctx.files.changed_paths() == ["a.py", "pkg/new.py"]
    assert any(e.type == EventType.FILE_CHANGED for e in events)


async def test_edit_file_uniqueness_crlf_and_hints(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    (ws / "w.txt").write_bytes(b"alpha\r\nbeta\r\nbeta\r\ngamma\r\n")
    await ex.execute(call("read_file", path="w.txt"), ctx)
    r = await ex.execute(call("edit_file", path="w.txt", old_string="beta", new_string="BETA"), ctx)
    assert r.is_error and "matches 2 places" in r.content
    r = await ex.execute(call("edit_file", path="w.txt", old_string="alpha\nbeta", new_string="ALPHA\nbeta"), ctx)
    assert not r.is_error, r.content
    assert (ws / "w.txt").read_bytes() == b"ALPHA\r\nbeta\r\nbeta\r\ngamma\r\n"
    r = await ex.execute(call("edit_file", path="w.txt", old_string="gama", new_string="x"), ctx)
    assert r.is_error and "Closest line 4" in r.content
    r = await ex.execute(call("edit_file", path="w.txt", old_string="beta", new_string="B", replace_all=True), ctx)
    assert not r.is_error and (ws / "w.txt").read_bytes().count(b"B\r\n") == 2


async def test_protected_and_outside_paths(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    for path in (".git/config", ".env", "../escape.txt", ".agent/state.db"):
        r = await ex.execute(call("write_file", path=path, content="x"), ctx)
        assert r.is_error, path
        assert "outside" in r.content or "protected" in r.content, r.content
    assert not (tmp_path / "escape.txt").exists()


async def test_env_file_values_are_redacted_on_read(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    (ws / ".env").write_text("DEBUG=1\nDB_URL=postgres://u:pw@h/db\n")
    r = await ex.execute(call("read_file", path=".env"), ctx)
    assert "DEBUG=[REDACTED]" in r.content and "pw@h" not in r.content


async def test_delete_and_list(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    (ws / "src").mkdir()
    (ws / "src" / "m.py").write_text("pass\n")
    (ws / "node_modules").mkdir()
    r = await ex.execute(call("list_directory", path=".", depth=2), ctx)
    assert "src/" in r.content and "m.py" in r.content and "node_modules/ (skipped)" in r.content
    r = await ex.execute(call("delete_file", path="src/m.py"), ctx)
    assert not r.is_error and not (ws / "src" / "m.py").exists()
    assert ctx.files.changes()[0].deleted


# ---- terminal -------------------------------------------------------------------------------


async def test_run_command_exit_codes_and_timeout(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    r = await ex.execute(call("run_command", command=f"{sys.executable} -c \"print('hi')\""), ctx)
    assert not r.is_error and "hi" in r.content and "exit code 0" in r.content
    r = await ex.execute(call("run_command", command=f"{sys.executable} -c \"import sys; sys.exit(3)\""), ctx)
    assert r.is_error and "exit code 3" in r.content
    r = await ex.execute(call("run_command", command="sleep 20", timeout_s=0.5), ctx)
    assert r.is_error and "timed out" in r.content


async def test_run_command_bounds_output(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    ctx.settings.terminal.max_output_chars = 1000
    r = await ex.execute(call("run_command", command=f"{sys.executable} -c \"print('a'*100000); print('TAIL')\""), ctx)
    assert "TAIL" in r.content and "omitted" in r.content and len(r.content) < 3000


def test_child_env_strips_secrets() -> None:
    s = Settings()
    env = build_env(s.terminal, base={"PATH": "/bin", "OPENAI_API_KEY": "sk-x", "DB_PASSWORD": "p", "KEEP_TOKEN": "t"})
    assert "OPENAI_API_KEY" not in env and "DB_PASSWORD" not in env and env["PATH"] == "/bin"
    s.terminal.env_allow = ["KEEP_TOKEN"]
    assert build_env(s.terminal, base={"KEEP_TOKEN": "t"})["KEEP_TOKEN"] == "t"
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["CI"] == "true"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"  # stale-bytecode regression guard


async def test_background_process_lifecycle(tmp_path: Path) -> None:
    ws, ctx, ex, _ = harness(tmp_path)
    script = f"{sys.executable} -c \"import time; print('ready', flush=True); time.sleep(30)\""
    r = await ex.execute(call("run_command", command=script, background=True), ctx)
    assert not r.is_error
    proc_id = ctx.processes.list()[0].id
    await asyncio.sleep(0.5)
    out = await ex.execute(call("process_output", process_id=proc_id), ctx)
    assert "ready" in out.content
    stop = await ex.execute(call("process_stop", process_id=proc_id), ctx)
    assert not stop.is_error
    assert not ctx.processes.get(proc_id).running


# ---- git ------------------------------------------------------------------------------------


async def test_git_tools(git_repo: Path, tmp_path: Path) -> None:
    settings = settings_for()
    ctx = make_context(git_repo, settings, redactor=Redactor(environ={}), git=GitRepo(git_repo))
    ex = ToolExecutor(ToolRegistry(default_tools()), PermissionPolicy(settings.permissions), AllowAllBroker())
    r = await ex.execute(call("git_status"), ctx)
    assert "working tree clean" in r.content
    await ex.execute(call("write_file", path="new.py", content="print(1)\n"), ctx)
    r = await ex.execute(call("git_status"), ctx)
    assert "?? new.py" in r.content
    r = await ex.execute(call("git_checkout", ref="main"), ctx)
    assert r.is_error and "uncommitted" in r.content
    r = await ex.execute(call("git_commit", message="add new module"), ctx)
    assert not r.is_error, r.content
    assert "add new module" in (await ex.execute(call("git_log", n=1), ctx)).content
    # secrets block commits
    await ex.execute(call("write_file", path="cfg.py", content=f"TOKEN = '{FAKE_TOKEN}'\n"), ctx)
    r = await ex.execute(call("git_commit", message="add config", paths=["cfg.py"]), ctx)
    assert r.is_error and "secrets" in r.content
    status = await ctx.git.status()
    assert "cfg.py" in status.untracked  # unstaged again after refusal
    r = await ex.execute(call("git_branch", create="--evil"), ctx)
    assert r.is_error


async def test_git_snapshot_and_restore_leave_index_untouched(git_repo: Path) -> None:
    repo = GitRepo(git_repo)
    (git_repo / "README.md").write_text("# changed\n")
    (git_repo / "staged.txt").write_text("staged\n")
    await repo.run("add", "staged.txt")
    commit, tree = await repo.snapshot("ck1", "checkpoint 1")
    assert await repo.snapshot_ref("ck1") == commit
    before_status = await repo.status()
    # agent makes further changes
    (git_repo / "README.md").write_text("# broken\n")
    (git_repo / "extra.py").write_text("x")
    diff = await repo.diff_since(tree)
    assert "broken" in diff and "extra.py" in diff
    assert set(await repo.changed_files_since(tree)) == {"README.md", "extra.py"}
    removed = await repo.restore_tree(tree)
    assert removed == ["extra.py"]
    assert (git_repo / "README.md").read_text() == "# changed\n"
    after_status = await repo.status()
    assert after_status.staged == before_status.staged == ["staged.txt"]
    assert await repo.current_branch() == "main"


# ---- database & web parsing ---------------------------------------------------------------


async def test_db_tools(tmp_path: Path) -> None:
    import sqlite3

    ws, ctx, ex, _ = harness(tmp_path)
    con = sqlite3.connect(ws / "app.db")
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("INSERT INTO users (name) VALUES ('ada'), ('bob')")
    con.commit()
    con.close()
    r = await ex.execute(call("db_schema", database="app.db"), ctx)
    assert "CREATE TABLE users" in r.content
    r = await ex.execute(call("db_query", database="app.db", sql="SELECT name FROM users WHERE id = ?", params=[2]), ctx)
    assert r.content.splitlines() == ["name", "bob"]
    r = await ex.execute(call("db_query", database="app.db", sql="INSERT INTO users (name) VALUES ('cy')"), ctx)
    assert not r.is_error
    r = await ex.execute(call("db_query", database="app.db", sql="DROP TABLE users"), ctx)
    assert r.is_error and "not approved" in r.content  # high risk needs a human


def test_html_to_text_and_url_checks() -> None:
    from ai_engineer.core.errors import ToolError
    from ai_engineer.tools.builtin.web import check_url, html_to_text

    title, text = html_to_text("<html><head><title>Docs</title><script>evil()</script></head><body><h1>API</h1><p>Use <b>foo</b>.</p><ul><li>a</li></ul></body></html>")
    assert title == "Docs" and "evil" not in text and "# API" in text and "- a" in text
    for bad in ("file:///etc/passwd", "http://169.254.169.254/latest", "http://user:pw@example.com", "http://localhost:8000"):
        with pytest.raises(ToolError):
            check_url(bad, [], [])
    with pytest.raises(ToolError):
        check_url("https://evil.example.com/x", [], ["example.com"])
    with pytest.raises(ToolError):
        check_url("https://other.org", ["docs.python.org"], [])


async def test_queue_broker_resolution_and_timeout() -> None:
    from ai_engineer.tools.approval import ApprovalRequest

    broker = QueueBroker(timeout_s=5)
    task = asyncio.create_task(broker.request(ApprovalRequest(tool="t", summary="s", reason="r")))
    await asyncio.sleep(0.01)
    pending = broker.pending()
    assert len(pending) == 1
    assert broker.resolve(pending[0].id, approved=True)
    assert (await task).approved
    quick = QueueBroker(timeout_s=0.05)
    decision = await quick.request(ApprovalRequest(tool="t", summary="s2", reason="r"))
    assert not decision.approved and decision.by == "timeout"


async def test_snapshot_detects_same_size_edit_in_same_second(git_repo: Path) -> None:
    """Regression: git's stat cache must not hide a same-size edit made in the same second.

    mtimes are pinned so the race is reproduced deterministically: the file and the index
    share one timestamp (as when an edit lands in the same second as the last index write).
    """
    import os
    import time

    from tests.conftest import git

    repo = GitRepo(git_repo)
    target = git_repo / "m.py"
    target.write_text("x = a - b\n")
    stamp = time.time() - 30
    os.utime(target, (stamp, stamp))
    git(git_repo, "add", "m.py")
    git(git_repo, "commit", "-q", "-m", "m")
    index = git_repo / ".git" / "index"
    os.utime(index, (stamp, stamp))
    _, tree = await repo.snapshot("c1", "before")
    target.write_text("x = a * b\n")  # same size
    os.utime(target, (stamp, stamp))  # and indistinguishable mtime
    os.utime(index, (stamp, stamp))
    assert await repo.changed_files_since(tree) == ["m.py"]
