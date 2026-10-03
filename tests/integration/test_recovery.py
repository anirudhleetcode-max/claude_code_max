"""Recovery and robustness: unrelated user changes, git edge cases, failed and interrupted restores,
agent-loop failure modes, model switching across a long task, and a hard process kill.

The model is scripted; these tests verify the harness, never model quality.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ai_engineer.core.errors import StateError
from ai_engineer.git.repo import GitRepo
from ai_engineer.providers.scripted import ScriptedProvider
from ai_engineer.tasks.store import TaskStatus
from tests.conftest import git
from tests.integration.helpers import APPROVE, call, j, make_calc_repo, open_runtime, understanding

FIX = {"old_string": "return a - b\n\n\ndef sub", "new_string": "return a + b\n\n\ndef sub"}
FIXED_CALC = '"""Tiny calculator."""\n\n\ndef add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n'


def fix_script() -> dict[str, list]:
    return {
        "classifier": [understanding()],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            call("edit_file", path="calc/__init__.py", **FIX),
            call("submit_work", summary="add() now returns a + b", files_changed=["calc/__init__.py"]),
        ],
        "reviewer": [APPROVE],
    }


def tree_state(repo: Path) -> dict[str, str]:
    """Contents of every non-ignored file plus git's view of the index and worktree."""
    files = {}
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo).as_posix()
        if path.is_file() and not rel.startswith((".git/", ".agent/")) and "__pycache__" not in rel and ".pytest_cache" not in rel:
            files[rel] = path.read_text(errors="replace")
    files["<status>"] = git(repo, "status", "--porcelain")
    return files


# ---- unrelated user changes and git edge cases ----------------------------------------------


async def test_user_changes_unstaged_staged_and_untracked_are_preserved(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    (repo / "pyproject.toml").write_text((repo / "pyproject.toml").read_text() + "# user's unstaged edit\n")
    (repo / "NOTES.md").write_text("user's staged notes\n")
    git(repo, "add", "NOTES.md")
    (repo / "scratch.py").write_text("print('user scratch, untracked')\n")
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=fix_script())})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
        start = result.state["start_checkpoint"]
        assert result.status == TaskStatus.COMPLETED, result.error
        assert result.state["auto_commit"] is False  # dirty start: nothing is committed
        assert git(repo, "log", "--format=%s").splitlines() == ["initial"]
        assert "# user's unstaged edit" in (repo / "pyproject.toml").read_text()
        assert git(repo, "diff", "--cached", "--name-only").split() == ["NOTES.md"]  # still staged
        assert "scratch.py" in git(repo, "status", "--porcelain")
        assert "return a + b" in (repo / "calc" / "__init__.py").read_text()
        # rolling the task back removes only the agent's change
        await rt.checkpoints.restore(start)
    finally:
        await rt.aclose()
    assert "return a - b" in (repo / "calc" / "__init__.py").read_text()
    assert "# user's unstaged edit" in (repo / "pyproject.toml").read_text()
    assert (repo / "scratch.py").exists() and (repo / "NOTES.md").read_text() == "user's staged notes\n"
    assert git(repo, "diff", "--cached", "--name-only").split() == ["NOTES.md"]


async def test_merge_in_progress_blocks_without_touching_the_repository(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    git(repo, "checkout", "-q", "-b", "other")
    (repo / "calc" / "__init__.py").write_text(FIXED_CALC.replace("a + b", "b + a"))
    git(repo, "commit", "-qam", "other")
    git(repo, "checkout", "-q", "main")
    (repo / "calc" / "__init__.py").write_text(FIXED_CALC)
    git(repo, "commit", "-qam", "main")
    subprocess.run(["git", "merge", "other"], cwd=repo, capture_output=True, check=False)
    assert (repo / ".git" / "MERGE_HEAD").exists()
    before = tree_state(repo)
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=fix_script())})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    assert result.status == TaskStatus.BLOCKED and ("merge" in (result.error or "") or "conflict" in (result.error or ""))
    assert tree_state(repo) == before and (repo / ".git" / "MERGE_HEAD").exists()


async def test_failed_implementation_rolls_back_to_the_exact_start(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    (repo / "scratch.py").write_text("user = 1\n")  # untracked user file present at the start
    before = tree_state(repo)
    script = {
        "classifier": [understanding()],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            call("edit_file", path="calc/__init__.py", old_string="return a - b\n\n\ndef sub", new_string="return a * b\n\n\ndef sub"),
            call("write_file", path="calc/helpers.py", content="BROKEN = (\n"),
            call("submit_work", summary="tried", files_changed=["calc/__init__.py", "calc/helpers.py"]),
        ],
        "debugger": [call("submit_work", summary="could not find the cause")] * 4,
        "reviewer": [APPROVE] * 3,
    }
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)}, agent={"max_repair_iterations": 1})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
        assert result.status == TaskStatus.FAILED
        await rt.checkpoints.restore(result.state["start_checkpoint"])
    finally:
        await rt.aclose()
    after = tree_state(repo)
    assert after == before, {k: (before.get(k), after.get(k)) for k in set(before) | set(after) if before.get(k) != after.get(k)}


async def test_interrupted_restore_names_the_safety_checkpoint_and_can_be_undone(git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rt = open_runtime(git_repo, {"s": ScriptedProvider("s")})
    try:
        (git_repo / "a.txt").write_text("state one\n")
        cp = await rt.checkpoints.create("state one")
        (git_repo / "a.txt").write_text("state two\n")
        (git_repo / "b.txt").write_text("new in state two\n")
        two = tree_state(git_repo)
        real_run = GitRepo.run

        async def dying_run(self, *args, **kw):
            if args and args[0] == "checkout-index":
                raise OSError("simulated crash while writing files")
            return await real_run(self, *args, **kw)

        monkeypatch.setattr(GitRepo, "run", dying_run)
        with pytest.raises(StateError, match="aie restore ") as info:
            await rt.checkpoints.restore(cp.id)
        monkeypatch.setattr(GitRepo, "run", real_run)
        assert tree_state(git_repo) != two  # the failed restore left a mixed tree (b.txt was removed)
        safety_id = str(info.value).rsplit("aie restore ", 1)[1].strip()
        await rt.checkpoints.restore(safety_id)
        assert tree_state(git_repo) == two
    finally:
        await rt.aclose()


# ---- agent-loop failure modes ---------------------------------------------------------------


async def test_file_deleted_under_the_agent_is_recovered_without_corruption(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")

    def delete_then_edit(req):
        (repo / "calc" / "__init__.py").unlink()  # someone deletes the file after the agent read it
        return call("edit_file", path="calc/__init__.py", **FIX)

    script = {
        "classifier": [understanding()],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            delete_then_edit,
            call("write_file", path="calc/__init__.py", content=FIXED_CALC),
            call("submit_work", summary="file was deleted; recreated it with the fix", files_changed=["calc/__init__.py"]),
        ],
        "reviewer": [APPROVE],
    }
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)})
    events: list = []
    rt.bus.subscribe(events.append)
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    assert result.status == TaskStatus.COMPLETED, result.error
    assert (repo / "calc" / "__init__.py").read_text() == FIXED_CALC
    failed_edits = [e for e in events if e.type == "TOOL_RESULT" and e.data.get("tool") == "edit_file" and not e.data.get("ok", True)]
    assert failed_edits  # the edit of the vanished file failed visibly instead of silently writing something


async def test_model_refusal_fails_honestly_and_leaves_the_repository_intact(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    before = tree_state(repo)
    refusal = {"text": "I can't help with that.", "stop_reason": "refusal"}
    script = {"classifier": [understanding()], "coder": [refusal], "reviewer": [APPROVE]}
    rt = open_runtime(repo, {"s": ScriptedProvider("s", by_role=script)})
    task = rt.create_task("Fix the add function in calc so the tests pass")
    try:
        result = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    assert result.status == TaskStatus.FAILED
    gates = {g["name"]: g["status"] for g in result.state["gates"]["results"]}
    assert gates["implementation"] == "FAILED"
    assert tree_state(repo) == before


# ---- a long task: several subtasks, model switching, outage, resume ------------------------


def three_step_plan() -> str:
    def sub(i: int, name: str, op: str) -> dict:
        return {"id": f"s{i}", "title": f"Add {name}()", "description": f"Add {name}(a, b) returning a {op} b to calc", "acceptance_criteria": [f"{name}() works"]}

    return j({
        "goal": "extend calc", "approach": "one function per subtask",
        "subtasks": [sub(1, "mul", "*"), {**sub(2, "div", "/"), "depends_on": ["s1"]}, {**sub(3, "mod", "%"), "depends_on": ["s2"]}],
    })


def append(name: str, op: str) -> list:
    line = f"\n\ndef {name}(a, b):\n    return a {op} b\n"
    return [
        call("read_file", path="calc/__init__.py"),
        call("edit_file", path="calc/__init__.py", old_string="def sub(a, b):\n    return a - b\n", new_string=f"def sub(a, b):\n    return a - b\n{line}"),
        call("submit_work", summary=f"added {name}", files_changed=["calc/__init__.py"]),
    ]


def approve(name: str) -> str:
    return j({"verdict": "approve", "summary": f"{name} added", "issues": [], "requirements": [{"criterion": f"{name}() works", "status": "met", "evidence": "calc/__init__.py"}]})


async def test_long_task_survives_model_switching_and_an_outage(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo", buggy=False)
    u = understanding(complexity="medium", summary="Add mul, div and mod to calc", requirements=["mul, div and mod exist"], acceptance_criteria=["mul() works", "div() works", "mod() works"])
    primary = ScriptedProvider("a", by_role={"classifier": [u], "planner": [three_step_plan()], "coder": append("mul", "*"), "reviewer": [approve("mul()")]})
    backup = ScriptedProvider("b", by_role={"coder": append("div", "/"), "reviewer": [approve("div()")]})  # nothing scripted for s3: outage
    roles = {"default": ["a:m", "b:m"]}
    rt = open_runtime(repo, {"a": primary, "b": backup}, roles=roles)
    events: list = []
    rt.bus.subscribe(events.append)
    task = rt.create_task("Add mul, div and mod functions to calc")
    try:
        first = await rt.run_task(task.id)
    finally:
        await rt.aclose()
    assert first.status == TaskStatus.BLOCKED  # every model failed during s3: state saved, nothing lost
    subs = first.state["subtasks"]
    assert subs["s1"]["status"] == "completed" and subs["s2"]["status"] == "completed" and subs["s3"]["status"] == "running"
    assert any(e.type == "MODEL_FALLBACK" for e in events)  # the primary ran out and the backup took over
    assert "def div" in (repo / "calc" / "__init__.py").read_text() and "def mod" not in (repo / "calc" / "__init__.py").read_text()
    commits_before = git(repo, "log", "--format=%s").splitlines()
    assert len([c for c in commits_before if c.startswith("aie: ")]) == 2  # one verified commit per finished subtask

    # the network comes back; a new process resumes and must finish s3 itself
    backup2 = ScriptedProvider("b", by_role={"coder": append("mod", "%"), "reviewer": [approve("mod()"), APPROVE]})
    rt2 = open_runtime(repo, {"a": ScriptedProvider("a"), "b": backup2}, roles=roles)
    try:
        final = await rt2.run_task(task.id)
        checkpoints = rt2.store.list_checkpoints(task.id)
    finally:
        await rt2.aclose()
    assert final.status == TaskStatus.COMPLETED, final.error
    assert "def mod" in (repo / "calc" / "__init__.py").read_text()
    assert len(checkpoints) >= 6  # task start + before/after each of the three subtasks


# ---- hard process termination ---------------------------------------------------------------


def _task_row(db: Path) -> tuple[str, str] | None:
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute("SELECT id, status FROM tasks WHERE parent_id IS NULL ORDER BY created DESC LIMIT 1").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return (row[0], row[1]) if row else None


def _configure_scripted(repo: Path, by_role: dict) -> None:
    script = repo / ".agent" / "script.json"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(json.dumps({"by_role": by_role}), encoding="utf-8")
    python = Path(sys.executable).as_posix()
    (repo / ".agent" / "config.toml").write_text(
        f'[models.providers.s]\ntype = "scripted"\noptions = {{ script_file = "{script.as_posix()}" }}\n\n'
        '[models.roles]\ndefault = ["s:m"]\n\n'
        f"[validation]\ntest_command = '\"{python}\" -m pytest -q -p no:cacheprovider'\ndependency_audit = false\n",
        encoding="utf-8",
    )


async def test_killed_process_is_recovered_and_resumed_to_a_verified_finish(tmp_path: Path) -> None:
    repo = make_calc_repo(tmp_path / "repo")
    _configure_scripted(repo, {
        "classifier": [understanding()],
        "coder": [
            call("read_file", path="calc/__init__.py"),
            call("edit_file", path="calc/__init__.py", **FIX),
            {"delay_s": 120, "text": "still thinking"},  # the process is killed while waiting here
        ],
    })
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[2] / "src"), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.Popen([sys.executable, "-m", "ai_engineer.ui.cli", "-C", str(repo), "run", "-q", "--no-interactive", "Fix the add function in calc"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    db = repo / ".agent" / "state.db"
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if "return a + b" in (repo / "calc" / "__init__.py").read_text() and (_task_row(db) or ("", ""))[1] == "RUNNING":
                break
            assert proc.poll() is None, "the agent exited before it could be killed"
            time.sleep(0.2)
        else:
            pytest.fail("the agent never reached the edit")
        time.sleep(0.5)
        proc.kill()  # no cleanup, no lease release: a crash
        proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()
    row = _task_row(db)
    assert row is not None and row[1] == "RUNNING"  # the crash left the row as it was
    task_id = row[0]
    # the next process notices the dead owner, and resume re-verifies instead of assuming success
    _configure_scripted(repo, {
        "coder": [call("read_file", path="calc/__init__.py"), call("submit_work", summary="the fix from the interrupted attempt is complete", files_changed=["calc/__init__.py"])],
        "reviewer": [APPROVE],
    })
    from ai_engineer.runtime import Runtime

    rt = Runtime.open(repo, use_global_config=False)
    try:
        assert rt.store.require_task(task_id).status == TaskStatus.INTERRUPTED
        resumed = await rt.run_task(task_id)
    finally:
        await rt.aclose()
    assert resumed.status == TaskStatus.COMPLETED, resumed.error
    sub = resumed.state["subtasks"]["s1"]
    assert sub["loop_status"] == "finished" and sub["validation"]
    assert any(v["kind"] == "test" and v["status"] == "passed" for v in sub["validation"])
