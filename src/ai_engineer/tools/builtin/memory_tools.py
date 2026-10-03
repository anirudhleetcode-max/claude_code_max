"""Agent tools for searching and recording long-lived memory."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import Field

from ...config.settings import PermissionLevel
from ...core.errors import ToolError
from ..base import SideEffect, Tool, ToolContext, ToolInput, ToolResult

LayerName = Literal["session", "project", "engineering", "command", "decision"]
RecordKind = Literal["fact", "decision", "lesson", "convention", "known_issue"]

_LAYER_FOR_KIND: dict[str, str] = {"decision": "decision", "lesson": "engineering"}


def _memory(ctx: ToolContext) -> Any:
    if ctx.memory is None:
        raise ToolError("memory is not available")
    return ctx.memory


class MemorySearchInput(ToolInput):
    query: str = Field(description="What to look for (keywords work best)")
    layers: list[LayerName] = Field(
        default_factory=list, description="Restrict to these memory layers (empty = all layers)"
    )
    limit: int = Field(default=8, ge=1, le=30, description="Maximum number of memories to return")


class MemorySearchTool(Tool):
    name = "memory_search"
    description = (
        "Search memory from earlier work: project facts, conventions, decisions, known commands and "
        "cross-project lessons. Results are hints, not truth: verify against the repository. Items marked "
        "STALE reference files that changed since they were recorded."
    )
    Input = MemorySearchInput

    def summarize(self, args: ToolInput) -> str:
        assert isinstance(args, MemorySearchInput)
        return f"Searching memory for {args.query[:80]!r}"

    async def run(self, args: MemorySearchInput, ctx: ToolContext) -> ToolResult:
        memory = _memory(ctx)
        items = await asyncio.to_thread(memory.search, args.query, layers=list(args.layers) or None, limit=args.limit)
        if not items:
            return ToolResult(content="No matching memories.", data={"count": 0, "ids": []})
        lines = [f"{i}. {item.render()}" for i, item in enumerate(items, 1)]
        lines.append("(Memories are hints from earlier work; the current repository state wins.)")
        return ToolResult(content="\n".join(lines), data={"count": len(items), "ids": [item.id for item in items]})


class MemoryRecordInput(ToolInput):
    kind: RecordKind = Field(
        description=(
            "fact: something true about this project; decision: a design choice and why; lesson: a reusable, "
            "cross-project engineering insight; convention: a project style/workflow rule; known_issue: a problem "
            "to watch for"
        )
    )
    content: str = Field(min_length=5, description="The memory, as one self-contained statement")
    key: str | None = Field(
        default=None, description="Stable identifier; recording again with the same key and kind replaces it"
    )
    files: list[str] = Field(
        default_factory=list, description="Workspace files this depends on (it is flagged stale when they change)"
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="How sure you are (0-1)")


class MemoryRecordTool(Tool):
    name = "memory_record"
    description = (
        "Record something worth remembering for future tasks: a verified project fact, a decision with its "
        "rationale, a convention, a known issue, or a reusable engineering lesson. Record only what you have "
        "verified; never record secrets. Reference the files it depends on so it can be flagged when they change."
    )
    Input = MemoryRecordInput
    level = PermissionLevel.SAFE_WRITE
    side_effect = SideEffect.WRITE

    def summarize(self, args: ToolInput) -> str:
        assert isinstance(args, MemoryRecordInput)
        return f"Remembering {args.kind}: {' '.join(args.content.split())[:100]}"

    async def run(self, args: MemoryRecordInput, ctx: ToolContext) -> ToolResult:
        memory = _memory(ctx)
        refs: list[str] = []
        for raw in args.files:
            path = ctx.guard.resolve(raw)
            rel = ctx.guard.relative(path)
            if not path.is_file():
                raise ToolError(f"file not found: {rel} (file references must name existing files)")
            if rel not in refs:
                refs.append(rel)
        layer = _LAYER_FOR_KIND.get(args.kind, "project")
        item = await asyncio.to_thread(
            memory.add,
            layer=layer,
            kind=args.kind,
            content=args.content,
            key=args.key,
            source=f"agent:{ctx.task_id or 'session'}",
            confidence=args.confidence,
            file_refs=refs,
        )
        where = f"{layer} memory" + (f" (version {item.version})" if item.version > 1 else "")
        return ToolResult(
            content=f"Recorded {args.kind} in {where}.",
            data={"id": item.id, "layer": layer, "version": item.version, "files": refs},
        )


MEMORY_TOOLS: list[type[Tool]] = [MemorySearchTool, MemoryRecordTool]
