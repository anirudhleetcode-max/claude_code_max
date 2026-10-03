"""Async Git wrapper with non-destructive snapshots.

Snapshots capture the working tree (tracked + untracked, honouring .gitignore)
through a *temporary index file*, so the user's index, HEAD and branch are never
touched. Snapshot commits are kept alive by refs under ``refs/ai-engineer/``.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import AIEngineerError

SNAPSHOT_REF_PREFIX = "refs/ai-engineer/checkpoints/"


class GitError(AIEngineerError):
    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


@dataclass
class StatusEntry:
    path: str
    index: str  # X in porcelain XY
    worktree: str  # Y in porcelain XY
    orig_path: str | None = None

    @property
    def untracked(self) -> bool:
        return self.index == "?"

    @property
    def conflicted(self) -> bool:
        return "U" in (self.index + self.worktree) or (self.index + self.worktree) in ("AA", "DD")


@dataclass
class GitStatus:
    branch: str | None
    head: str | None
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    entries: list[StatusEntry] = field(default_factory=list)
    in_progress: str | None = None  # merge / rebase / cherry-pick / revert / bisect

    @property
    def clean(self) -> bool:
        return not self.entries

    @property
    def conflicts(self) -> list[str]:
        return [e.path for e in self.entries if e.conflicted]

    @property
    def staged(self) -> list[str]:
        return [e.path for e in self.entries if e.index not in (".", "?", "!")]

    @property
    def unstaged(self) -> list[str]:
        return [e.path for e in self.entries if e.worktree not in (".", "?", "!") and not e.untracked]

    @property
    def untracked(self) -> list[str]:
        return [e.path for e in self.entries if e.untracked]

    def summary(self) -> str:
        parts = [f"branch {self.branch or '(detached)'}"]
        if self.upstream:
            parts.append(f"upstream {self.upstream} (+{self.ahead}/-{self.behind})")
        if self.in_progress:
            parts.append(f"{self.in_progress} in progress")
        if self.clean:
            parts.append("working tree clean")
        else:
            parts.append(
                f"{len(self.staged)} staged, {len(self.unstaged)} modified, {len(self.untracked)} untracked"
                + (f", {len(self.conflicts)} CONFLICTED" if self.conflicts else "")
            )
        return "; ".join(parts)


def git_available() -> bool:
    return shutil.which("git") is not None


class GitRepo:
    def __init__(self, root: Path, author_name: str | None = None, author_email: str | None = None) -> None:
        self.root = root
        self.author_name = author_name
        self.author_email = author_email

    def _env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "PAGER": "cat", "LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0"})
        if self.author_name:
            env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = self.author_name
        if self.author_email:
            env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = self.author_email
        if extra:
            env.update(extra)
        return env

    async def run(self, *args: str, env: dict[str, str] | None = None, check: bool = True, input_text: str | None = None, timeout: float = 120) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git", "-c", "core.quotepath=off", *args,
            cwd=str(self.root),
            env=self._env(env),
            stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(input_text.encode() if input_text is not None else None), timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise GitError(f"git {' '.join(args[:3])} timed out") from None
        stdout = out.decode("utf-8", errors="replace")
        stderr = err.decode("utf-8", errors="replace")
        if check and proc.returncode != 0:
            raise GitError(f"git {' '.join(args[:4])} failed: {stderr.strip()[:500]}", returncode=proc.returncode, stderr=stderr)
        return stdout

    # ---- queries -------------------------------------------------------------------

    async def is_repo(self) -> bool:
        if not git_available():
            return False
        try:
            out = await self.run("rev-parse", "--is-inside-work-tree")
        except GitError:
            return False
        return out.strip() == "true"

    async def toplevel(self) -> Path:
        return Path((await self.run("rev-parse", "--show-toplevel")).strip())

    async def head(self) -> str | None:
        try:
            return (await self.run("rev-parse", "--verify", "-q", "HEAD")).strip() or None
        except GitError:
            return None

    async def current_branch(self) -> str | None:
        out = (await self.run("symbolic-ref", "--short", "-q", "HEAD", check=False)).strip()
        return out or None

    async def status(self) -> GitStatus:
        out = await self.run("status", "--porcelain=v2", "--branch", "-z", "--untracked-files=all")
        status = GitStatus(branch=None, head=None)
        records = out.split("\0")
        i = 0
        while i < len(records):
            rec = records[i]
            i += 1
            if not rec:
                continue
            if rec.startswith("# branch.oid "):
                oid = rec.split()[-1]
                status.head = None if oid == "(initial)" else oid
            elif rec.startswith("# branch.head "):
                head = rec.split(" ", 2)[-1]
                status.branch = None if head == "(detached)" else head
            elif rec.startswith("# branch.upstream "):
                status.upstream = rec.split(" ", 2)[-1]
            elif rec.startswith("# branch.ab "):
                m = re.match(r"# branch\.ab \+(\d+) -(\d+)", rec)
                if m:
                    status.ahead, status.behind = int(m.group(1)), int(m.group(2))
            elif rec.startswith("1 "):
                parts = rec.split(" ", 8)
                status.entries.append(StatusEntry(parts[8], parts[1][0], parts[1][1]))
            elif rec.startswith("2 "):
                parts = rec.split(" ", 9)
                orig = records[i] if i < len(records) else None
                i += 1
                status.entries.append(StatusEntry(parts[9], parts[1][0], parts[1][1], orig))
            elif rec.startswith("u "):
                parts = rec.split(" ", 10)
                status.entries.append(StatusEntry(parts[10], parts[1][0], parts[1][1]))
            elif rec.startswith("? "):
                status.entries.append(StatusEntry(rec[2:], "?", "?"))
        git_dir = Path((await self.run("rev-parse", "--git-dir")).strip())
        if not git_dir.is_absolute():
            git_dir = self.root / git_dir
        for marker, name in (("MERGE_HEAD", "merge"), ("rebase-merge", "rebase"), ("rebase-apply", "rebase"),
                             ("CHERRY_PICK_HEAD", "cherry-pick"), ("REVERT_HEAD", "revert"), ("BISECT_LOG", "bisect")):
            if (git_dir / marker).exists():
                status.in_progress = name
                break
        return status

    async def diff(self, *, staged: bool = False, base: str | None = None, paths: list[str] | None = None, stat: bool = False, context: int = 3) -> str:
        args = ["diff", f"-U{context}", "--no-color", "--no-ext-diff"]
        if stat:
            args.append("--stat")
        if staged:
            args.append("--cached")
        if base:
            args.append(base)
        if paths:
            args += ["--", *paths]
        return await self.run(*args)

    async def log(self, n: int = 20, path: str | None = None, ref: str | None = None) -> list[dict[str, str]]:
        args = ["log", f"-n{n}", "--format=%H%x1f%an%x1f%ad%x1f%s", "--date=iso-strict"]
        if ref:
            args.append(ref)
        if path:
            args += ["--", path]
        out = await self.run(*args, check=False)
        entries = []
        for line in out.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                entries.append({"sha": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]})
        return entries

    async def branches(self) -> list[str]:
        out = await self.run("for-each-ref", "--format=%(refname:short)", "refs/heads")
        return [b for b in out.splitlines() if b]

    async def list_worktree_files(self) -> list[str]:
        out = await self.run("ls-files", "-co", "--exclude-standard", "-z")
        return [p for p in out.split("\0") if p]

    # ---- mutations ---------------------------------------------------------------------

    async def create_branch(self, name: str, checkout: bool = True, start: str | None = None) -> None:
        if checkout:
            await self.run("switch", "-c", name, *([start] if start else []))
        else:
            await self.run("branch", name, *([start] if start else []))

    async def checkout(self, ref: str, allow_dirty: bool = False) -> None:
        status = await self.status()
        if not status.clean and not allow_dirty:
            raise GitError("working tree has uncommitted changes; refusing to switch branches (commit, stash or checkpoint first)")
        if ref.startswith("-"):
            raise GitError(f"invalid ref: {ref}")
        await self.run("switch", ref)

    async def add(self, paths: list[str]) -> None:
        if paths:
            await self.run("add", "--", *paths)

    async def commit(self, message: str, paths: list[str] | None = None, allow_empty: bool = False) -> str:
        if paths is not None:
            await self.add(paths)
        args = ["commit", "-q", "-F", "-"]
        if allow_empty:
            args.append("--allow-empty")
        await self.run(*args, input_text=message)
        return (await self.run("rev-parse", "HEAD")).strip()

    # ---- snapshots ------------------------------------------------------------------------

    async def _temp_index(self) -> str:
        fd, path = tempfile.mkstemp(prefix="aie-index-")
        os.close(fd)
        os.unlink(path)  # git creates it
        return path

    async def snapshot_tree(self) -> str:
        """Write the current working tree (honouring .gitignore) to a tree object."""
        index = await self._temp_index()
        env = {"GIT_INDEX_FILE": index}
        try:
            # Seed from a copy of the real index: its stat cache means only changed
            # files are re-hashed. The user's own index is never modified.
            real_index = Path((await self.run("rev-parse", "--git-path", "index")).strip())
            if not real_index.is_absolute():
                real_index = self.root / real_index
            if real_index.is_file():
                shutil.copyfile(real_index, index)
            else:
                head = await self.head()
                if head:
                    await self.run("read-tree", head, env=env)
            await self.run("add", "-A", "--", ".", env=env)
            return (await self.run("write-tree", env=env)).strip()
        finally:
            if os.path.exists(index):
                os.unlink(index)

    async def snapshot(self, checkpoint_id: str, message: str) -> tuple[str, str]:
        """Create a snapshot commit referenced by ``refs/ai-engineer/checkpoints/<id>``. Returns (commit, tree)."""
        tree = await self.snapshot_tree()
        head = await self.head()
        args = ["commit-tree", tree, "-m", message]
        if head:
            args[2:2] = ["-p", head]
        env = {
            "GIT_AUTHOR_NAME": self.author_name or "ai-engineer",
            "GIT_AUTHOR_EMAIL": self.author_email or "ai-engineer@localhost",
            "GIT_COMMITTER_NAME": self.author_name or "ai-engineer",
            "GIT_COMMITTER_EMAIL": self.author_email or "ai-engineer@localhost",
        }
        commit = (await self.run(*args, env=env)).strip()
        await self.run("update-ref", SNAPSHOT_REF_PREFIX + checkpoint_id, commit)
        return commit, tree

    async def snapshot_ref(self, checkpoint_id: str) -> str | None:
        out = (await self.run("rev-parse", "--verify", "-q", SNAPSHOT_REF_PREFIX + checkpoint_id, check=False)).strip()
        return out or None

    async def tree_files(self, treeish: str) -> list[str]:
        out = await self.run("ls-tree", "-r", "-z", "--name-only", treeish)
        return [p for p in out.split("\0") if p]

    async def diff_trees(self, a: str, b: str, stat: bool = False, paths: list[str] | None = None) -> str:
        args = ["diff", "--no-color", "--no-ext-diff", "-M"]
        if stat:
            args.append("--stat")
        args += [a, b]
        if paths:
            args += ["--", *paths]
        return await self.run(*args)

    async def diff_since(self, treeish: str, stat: bool = False, paths: list[str] | None = None) -> str:
        """Diff from a commit/tree to the *current working tree*, including untracked files."""
        current = await self.snapshot_tree()
        return await self.diff_trees(treeish, current, stat=stat, paths=paths)

    async def changed_files_since(self, treeish: str) -> list[str]:
        current = await self.snapshot_tree()
        out = await self.run("diff", "--name-only", "-z", "--no-renames", treeish, current)
        return [p for p in out.split("\0") if p]

    async def restore_tree(self, treeish: str) -> list[str]:
        """Make the working tree match ``treeish`` without touching HEAD or the user's index.

        Returns paths deleted because they did not exist in the snapshot.
        Callers must snapshot the current state first so the restore is reversible.
        """
        wanted = set(await self.tree_files(treeish))
        present = await self.list_worktree_files()
        removed = []
        for rel in present:
            if rel not in wanted:
                target = self.root / rel
                if target.is_file() or target.is_symlink():
                    target.unlink()
                    removed.append(rel)
        index = await self._temp_index()
        env = {"GIT_INDEX_FILE": index}
        try:
            await self.run("read-tree", treeish, env=env)
            await self.run("checkout-index", "-a", "-f", env=env)
        finally:
            if os.path.exists(index):
                os.unlink(index)
        # prune now-empty directories left behind by removed files
        for rel in removed:
            parent = (self.root / rel).parent
            while parent != self.root and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
        return removed

    async def delete_snapshot(self, checkpoint_id: str) -> None:
        await self.run("update-ref", "-d", SNAPSHOT_REF_PREFIX + checkpoint_id, check=False)
