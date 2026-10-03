"""Git tools. History-destroying operations are intentionally not exposed."""

from __future__ import annotations

import re

from pydantic import Field

from ...config.settings import PermissionLevel
from ...core.errors import ToolError
from ...core.events import EventType
from ...core.util import truncate_middle
from ...git.repo import GitError, GitRepo
from ...security.paths import PathGuard
from ...security.secrets import redact_secret_files_in_diff, scan_text
from ..base import ActionAssessment, SideEffect, Tool, ToolContext, ToolInput, ToolResult

_BRANCH_RE = re.compile(r"^(?!-)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/\-]{1,100}(?<!\.lock)(?<!/)$")


def _git(ctx: ToolContext) -> GitRepo:
    if ctx.git is None:
        raise ToolError("this workspace is not a git repository")
    return ctx.git


async def stage_for_commit(git: GitRepo, guard: PathGuard, paths: list[str]) -> list[str]:
    """Stage exactly ``paths`` and return the staged file list, or unstage them and raise ToolError when
    they include secret/protected files or the added lines contain possible secrets."""
    if not paths:
        return []
    await git.run("add", "-A", "--", *paths)
    staged_files = [f for f in (await git.run("diff", "--cached", "--name-only", "-z", "--", *paths)).split("\0") if f]
    blocked = [f for f in staged_files if guard.is_secret_file(f) or (guard.is_protected(f) and not f.startswith(".agent/"))]
    if blocked:
        await git.run("reset", "-q", "--", *paths)
        raise ToolError(
            f"refusing to commit secret or protected files: {', '.join(blocked[:10])}; "
            "the agent does not commit these — leave them out (the user can commit them deliberately)"
        )
    staged = await git.diff(staged=True, paths=paths)
    added = "\n".join(line[1:] for line in staged.splitlines() if line.startswith("+") and not line.startswith("+++"))
    secrets = scan_text(added)
    if secrets:
        await git.run("reset", "-q", "--", *paths)
        kinds = ", ".join(sorted({s.kind for s in secrets}))
        raise ToolError(f"refusing to commit: staged changes contain possible secrets ({kinds}); remove them first")
    return staged_files


class GitStatusInput(ToolInput):
    pass


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show branch, upstream, and staged/modified/untracked/conflicted files."
    Input = GitStatusInput

    async def run(self, args: GitStatusInput, ctx: ToolContext) -> ToolResult:
        status = await _git(ctx).status()
        lines = [status.summary()]
        for entry in status.entries[:300]:
            code = "??" if entry.untracked else f"{entry.index}{entry.worktree}"
            lines.append(f"  {code} {entry.path}" + (f" (from {entry.orig_path})" if entry.orig_path else ""))
        if len(status.entries) > 300:
            lines.append(f"  ... {len(status.entries) - 300} more")
        return ToolResult(content="\n".join(lines), data={"clean": status.clean, "conflicts": status.conflicts})


class GitDiffInput(ToolInput):
    staged: bool = Field(default=False, description="Show staged changes instead of unstaged")
    base: str | None = Field(default=None, description="Compare against this commit/branch")
    paths: list[str] = Field(default_factory=list)
    stat: bool = Field(default=False, description="Only show a summary of changed files")


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Show a unified diff of working-tree, staged, or commit changes."
    Input = GitDiffInput

    async def run(self, args: GitDiffInput, ctx: ToolContext) -> ToolResult:
        if args.base and args.base.startswith("-"):
            raise ToolError("invalid base ref")
        for p in args.paths:
            ctx.guard.resolve(p)
        diff = await _git(ctx).diff(staged=args.staged, base=args.base, paths=args.paths or None, stat=args.stat)
        diff = redact_secret_files_in_diff(diff, ctx.guard.is_secret_file)
        return ToolResult(content=truncate_middle(diff, 60000) or "(no differences)")


class GitLogInput(ToolInput):
    n: int = Field(default=15, ge=1, le=200)
    path: str | None = None
    ref: str | None = None


class GitLogTool(Tool):
    name = "git_log"
    description = "Show recent commits (sha, author, date, subject), optionally for one path."
    Input = GitLogInput

    async def run(self, args: GitLogInput, ctx: ToolContext) -> ToolResult:
        if args.ref and args.ref.startswith("-"):
            raise ToolError("invalid ref")
        if args.path:
            ctx.guard.resolve(args.path)
        entries = await _git(ctx).log(args.n, args.path, args.ref)
        lines = [f"{e['sha'][:10]} {e['date'][:10]} {e['author']}: {e['subject']}" for e in entries]
        return ToolResult(content="\n".join(lines) or "(no commits)")


class GitBranchInput(ToolInput):
    create: str | None = Field(default=None, description="Name of a new branch to create and switch to")


class GitBranchTool(Tool):
    name = "git_branch"
    description = "List local branches, or create and switch to a new branch (requires a clean working tree)."
    Input = GitBranchInput

    def assess(self, args: GitBranchInput, ctx: ToolContext) -> ActionAssessment:
        if args.create:
            return ActionAssessment(level=PermissionLevel.DEVELOPMENT, summary=f"Creating branch {args.create}", read_only=False)
        return ActionAssessment(level=PermissionLevel.READ_ONLY, summary="Listing branches")

    async def run(self, args: GitBranchInput, ctx: ToolContext) -> ToolResult:
        git = _git(ctx)
        if args.create:
            if not _BRANCH_RE.match(args.create):
                raise ToolError(f"invalid branch name: {args.create}")
            await git.create_branch(args.create)
            return ToolResult(content=f"created and switched to {args.create}")
        current = await git.current_branch()
        branches = await git.branches()
        return ToolResult(content="\n".join(("* " if b == current else "  ") + b for b in branches))


class GitCheckoutInput(ToolInput):
    ref: str = Field(description="Existing branch to switch to")


class GitCheckoutTool(Tool):
    name = "git_checkout"
    description = "Switch to an existing branch. Refuses when there are uncommitted changes (never discards work)."
    Input = GitCheckoutInput
    level = PermissionLevel.DEVELOPMENT
    side_effect = SideEffect.WRITE

    def assess(self, args: GitCheckoutInput, ctx: ToolContext) -> ActionAssessment:
        return ActionAssessment(level=self.level, summary=f"Switching to branch {args.ref}", read_only=False)

    async def run(self, args: GitCheckoutInput, ctx: ToolContext) -> ToolResult:
        if not _BRANCH_RE.match(args.ref):
            raise ToolError(f"invalid ref: {args.ref}")
        try:
            await _git(ctx).checkout(args.ref)
        except GitError as exc:
            raise ToolError(str(exc)) from exc
        ctx.files.forget_reads()  # files may differ on the new branch
        return ToolResult(content=f"switched to {args.ref}")


class GitCommitInput(ToolInput):
    message: str = Field(min_length=3, description="Commit message (first line: concise summary)")
    paths: list[str] = Field(default_factory=list, description="Files to commit; default: files changed by the agent in this session")


class GitCommitTool(Tool):
    name = "git_commit"
    description = (
        "Commit specific files (by default only the files the agent changed in this session). Only those paths "
        "are committed; other staged changes are left alone. Secret/protected files and diffs containing "
        "secrets are refused."
    )
    Input = GitCommitInput
    level = PermissionLevel.DEVELOPMENT
    side_effect = SideEffect.WRITE

    def assess(self, args: GitCommitInput, ctx: ToolContext) -> ActionAssessment:
        paths = args.paths or ctx.files.changed_paths()
        return ActionAssessment(
            level=self.level, summary=f"Committing {len(paths)} file(s): {args.message.splitlines()[0][:80]}", read_only=False, paths=paths
        )

    async def run(self, args: GitCommitInput, ctx: ToolContext) -> ToolResult:
        git = _git(ctx)
        status = await git.status()
        if status.conflicts:
            raise ToolError(f"cannot commit with unresolved conflicts: {', '.join(status.conflicts)}")
        paths = args.paths or ctx.files.changed_paths()
        if not paths:
            raise ToolError("nothing to commit: no paths given and the agent has not changed any files")
        for p in paths:
            rel = ctx.guard.relative(ctx.guard.resolve(p))
            if ctx.guard.is_protected(rel) and not rel.startswith(".agent/"):
                raise ToolError(f"refusing to commit protected path {rel}")
        staged_files = await stage_for_commit(git, ctx.guard, paths)
        if not staged_files:
            return ToolResult(content="nothing to commit (no changes in the given paths)")
        # Commit only these paths: changes the user staged elsewhere are neither committed nor scanned.
        sha = await git.commit(args.message, only=paths)
        ctx.bus.emit(EventType.COMMIT_CREATED, f"committed {sha[:10]}: {args.message.splitlines()[0]}", data={"sha": sha, "paths": staged_files})
        return ToolResult(content=f"committed {sha[:10]} ({len(staged_files)} file(s))", data={"sha": sha, "files_changed": staged_files})


GIT_TOOLS: list[type[Tool]] = [GitStatusTool, GitDiffTool, GitLogTool, GitBranchTool, GitCheckoutTool, GitCommitTool]
