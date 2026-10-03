"""File system tools."""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

from pydantic import Field

from ...config.settings import PermissionLevel
from ...core.errors import ToolError
from ...core.events import EventType
from ...core.util import atomic_write_text, is_binary_bytes
from ...security.secrets import redact_secret_file_text
from ..base import ActionAssessment, SideEffect, Tool, ToolContext, ToolInput, ToolResult

MAX_READ_BYTES = 5_000_000


def _read_text(path: Path) -> tuple[str, str, bytes]:
    if not path.exists():
        raise ToolError(f"file not found: {path.name}")
    if path.is_dir():
        raise ToolError(f"{path.name} is a directory; use list_directory")
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(f"file is too large to read whole ({size} bytes); use search_text or read a range")
    data = path.read_bytes()
    if is_binary_bytes(data):
        raise ToolError(f"{path.name} is a binary file ({size} bytes)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    newline = "\r\n" if "\r\n" in text else "\n"
    return text, newline, data


def _emit_change(ctx: ToolContext, rel: str, action: str, lines: int) -> None:
    ctx.bus.emit(EventType.FILE_CHANGED, f"{action} {rel}", data={"path": rel, "action": action, "lines": lines})


class ReadFileInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root")
    offset: int = Field(default=1, ge=1, description="First line to return (1-based)")
    limit: int = Field(default=2000, ge=1, le=5000, description="Maximum number of lines to return")


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a text file. Returns lines prefixed with line numbers (e.g. '  12\\tcode'). "
        "Use offset/limit for large files. You must read a file before overwriting it."
    )
    Input = ReadFileInput

    def summarize(self, args: ReadFileInput) -> str:
        return f"Reading {args.path}" + (f" from line {args.offset}" if args.offset > 1 else "")

    async def run(self, args: ReadFileInput, ctx: ToolContext) -> ToolResult:
        path = ctx.guard.resolve(args.path)
        text, _, data = _read_text(path)
        rel = ctx.guard.relative(path)
        ctx.files.record_read(path, data)
        if ctx.guard.is_secret_file(rel):
            text = redact_secret_file_text(text)
        lines = text.splitlines()
        total = len(lines)
        start = args.offset - 1
        chunk = lines[start : start + args.limit]
        width = len(str(start + len(chunk)))
        body = "\n".join(f"{i + start + 1:>{width}}\t{line}" for i, line in enumerate(chunk))
        footer = ""
        if start + len(chunk) < total:
            footer = f"\n[showing lines {args.offset}-{start + len(chunk)} of {total}; use offset={start + len(chunk) + 1} to continue]"
        if total == 0:
            body = "(empty file)"
        return ToolResult(content=body + footer, data={"path": rel, "lines": total})


class WriteFileInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root")
    content: str = Field(description="Complete new file content")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely replace an existing one. To change part of an existing file "
        "prefer edit_file. Existing files must be read first."
    )
    Input = WriteFileInput
    level = PermissionLevel.SAFE_WRITE
    side_effect = SideEffect.WRITE

    def assess(self, args: WriteFileInput, ctx: ToolContext) -> ActionAssessment:
        path = ctx.guard.resolve(args.path, for_write=True)
        verb = "Updating" if path.exists() else "Creating"
        return ActionAssessment(level=self.level, summary=f"{verb} {ctx.guard.relative(path)}", read_only=False, paths=[ctx.guard.relative(path)])

    async def run(self, args: WriteFileInput, ctx: ToolContext) -> ToolResult:
        path = ctx.guard.resolve(args.path, for_write=True)
        rel = ctx.guard.relative(path)
        if path.is_dir():
            raise ToolError(f"{rel} is a directory")
        existed = path.exists()
        newline = "\n"
        if existed:
            ctx.files.check_write(path)
            _, newline, _ = _read_text(path)
        content = args.content
        if newline == "\r\n" and "\r\n" not in content:
            content = content.replace("\n", "\r\n")
        ctx.files.before_write(path)
        atomic_write_text(path, content, newline="")
        ctx.files.after_write(path)
        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        action = "updated" if existed else "created"
        _emit_change(ctx, rel, action, lines)
        return ToolResult(content=f"{action} {rel} ({lines} lines)", data={"path": rel, "action": action})


class EditFileInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root")
    old_string: str = Field(description="Exact text to replace (must match the file exactly, including indentation)")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace every occurrence instead of requiring a unique match")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must match exactly once (include surrounding lines "
        "to make it unique) unless replace_all is true. The file must have been read first. Do not include "
        "the line-number prefixes shown by read_file."
    )
    Input = EditFileInput
    level = PermissionLevel.SAFE_WRITE
    side_effect = SideEffect.WRITE

    def assess(self, args: EditFileInput, ctx: ToolContext) -> ActionAssessment:
        path = ctx.guard.resolve(args.path, for_write=True)
        rel = ctx.guard.relative(path)
        return ActionAssessment(level=self.level, summary=f"Editing {rel}", read_only=False, paths=[rel])

    async def run(self, args: EditFileInput, ctx: ToolContext) -> ToolResult:
        path = ctx.guard.resolve(args.path, for_write=True)
        rel = ctx.guard.relative(path)
        if args.old_string == args.new_string:
            raise ToolError("old_string and new_string are identical; nothing to change")
        if not args.old_string:
            raise ToolError("old_string must not be empty; use write_file to create files")
        text, newline, _ = _read_text(path)
        ctx.files.check_write(path)
        old, new = args.old_string, args.new_string
        if newline == "\r\n":
            old = old.replace("\r\n", "\n").replace("\n", "\r\n")
            new = new.replace("\r\n", "\n").replace("\n", "\r\n")
        count = text.count(old)
        if count == 0:
            raise ToolError(_no_match_hint(text, args.old_string, rel))
        if count > 1 and not args.replace_all:
            raise ToolError(f"old_string matches {count} places in {rel}; add surrounding context to make it unique or set replace_all")
        updated = text.replace(old, new) if args.replace_all else text.replace(old, new, 1)
        ctx.files.before_write(path)
        atomic_write_text(path, updated, newline="")
        ctx.files.after_write(path)
        diff = "".join(
            difflib.unified_diff(
                text.splitlines(keepends=True), updated.splitlines(keepends=True), f"a/{rel}", f"b/{rel}", n=2
            )
        )
        if len(diff) > 3000:
            diff = diff[:3000] + "\n... (diff truncated)"
        _emit_change(ctx, rel, "edited", updated.count("\n"))
        return ToolResult(content=f"edited {rel} ({count if args.replace_all else 1} replacement(s))\n{diff}", data={"path": rel})


def _no_match_hint(text: str, old: str, rel: str) -> str:
    first = old.strip().splitlines()[0] if old.strip() else old
    lines = text.splitlines()
    close = difflib.get_close_matches(first.strip(), [line.strip() for line in lines], n=1, cutoff=0.6)
    msg = f"old_string not found in {rel}."
    if re.match(r"^\s*\d+\t", old):
        msg += " It seems to include read_file line-number prefixes; remove them."
    if close:
        idx = [line.strip() for line in lines].index(close[0])
        msg += f" Closest line {idx + 1}: {lines[idx].strip()[:200]!r}. Re-read the file and copy the exact text."
    else:
        msg += " Re-read the file and copy the exact text, including whitespace."
    return msg


class DeleteFileInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root")


class DeleteFileTool(Tool):
    name = "delete_file"
    description = "Delete a single file (not directories). The original is backed up by the checkpoint system."
    Input = DeleteFileInput
    level = PermissionLevel.SAFE_WRITE
    side_effect = SideEffect.WRITE

    def assess(self, args: DeleteFileInput, ctx: ToolContext) -> ActionAssessment:
        path = ctx.guard.resolve(args.path, for_write=True)
        rel = ctx.guard.relative(path)
        return ActionAssessment(level=self.level, summary=f"Deleting {rel}", read_only=False, paths=[rel])

    async def run(self, args: DeleteFileInput, ctx: ToolContext) -> ToolResult:
        path = ctx.guard.resolve(args.path, for_write=True)
        rel = ctx.guard.relative(path)
        if not path.exists():
            raise ToolError(f"file not found: {rel}")
        if path.is_dir():
            raise ToolError(f"{rel} is a directory; only single files can be deleted")
        ctx.files.before_write(path)
        path.unlink()
        ctx.files.after_write(path, deleted=True)
        _emit_change(ctx, rel, "deleted", 0)
        return ToolResult(content=f"deleted {rel}", data={"path": rel})


class ListDirectoryInput(ToolInput):
    path: str = Field(default=".", description="Directory relative to the workspace root")
    depth: int = Field(default=1, ge=1, le=4, description="How many levels to descend")
    include_hidden: bool = False


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List files and directories (with sizes). Skips dependency/build directories such as node_modules."
    Input = ListDirectoryInput

    def summarize(self, args: ListDirectoryInput) -> str:
        return f"Listing {args.path}"

    async def run(self, args: ListDirectoryInput, ctx: ToolContext) -> ToolResult:
        from ...repo.files import DEFAULT_IGNORE_DIRS

        root = ctx.guard.resolve(args.path)
        if not root.is_dir():
            raise ToolError(f"not a directory: {args.path}")
        lines: list[str] = []
        limit = 500

        def walk(directory: Path, level: int) -> None:
            try:
                entries = sorted(os.scandir(directory), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
            except OSError as exc:
                lines.append(f"{'  ' * level}[unreadable: {exc.strerror}]")
                return
            for entry in entries:
                if len(lines) >= limit:
                    return
                if not args.include_hidden and entry.name.startswith(".") and entry.name not in (".github", ".gitignore", ".env.example"):
                    continue
                indent = "  " * level
                if entry.is_dir(follow_symlinks=False):
                    skipped = entry.name in DEFAULT_IGNORE_DIRS
                    lines.append(f"{indent}{entry.name}/" + (" (skipped)" if skipped else ""))
                    if level + 1 < args.depth and not skipped:
                        walk(Path(entry.path), level + 1)
                else:
                    try:
                        size = entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        size = 0
                    lines.append(f"{indent}{entry.name} ({size} B)")

        walk(root, 0)
        if len(lines) >= limit:
            lines.append(f"[truncated at {limit} entries]")
        return ToolResult(content="\n".join(lines) or "(empty directory)", data={"path": ctx.guard.relative(root)})


FS_TOOLS: list[type[Tool]] = [ReadFileTool, WriteFileTool, EditFileTool, DeleteFileTool, ListDirectoryTool]
