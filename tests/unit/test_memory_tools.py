from __future__ import annotations

from pathlib import Path

import pytest

from ai_engineer.config.settings import Mode, Settings
from ai_engineer.core.types import ToolUseBlock
from ai_engineer.memory import MemoryManager, MemoryStore
from ai_engineer.security.secrets import Redactor
from ai_engineer.tools.approval import AllowAllBroker
from ai_engineer.tools.builtin.memory_tools import MEMORY_TOOLS, MemoryRecordTool, MemorySearchTool
from ai_engineer.tools.executor import ToolExecutor, ToolRegistry
from ai_engineer.tools.factory import make_context
from ai_engineer.tools.permissions import PermissionPolicy

FAKE_SECRET = "ghp_" + "B" * 36


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("PORT = 8000\n")
    (tmp_path / "outside.py").write_text("x = 1\n")
    return root


@pytest.fixture
def manager(tmp_path: Path, ws: Path):
    mgr = MemoryManager(
        MemoryStore(ws / ".agent" / "state.db", workspace=ws, redactor=Redactor(environ={})),
        MemoryStore(tmp_path / "home" / "memory.db", redactor=Redactor(environ={})),
    )
    yield mgr
    mgr.close()


def setup(ws: Path, memory: MemoryManager | None, mode: Mode = Mode.DEVELOPER, task_id: str = ""):
    settings = Settings()
    settings.permissions.mode = mode
    ctx = make_context(ws, settings, memory=memory, task_id=task_id)
    executor = ToolExecutor(
        ToolRegistry([cls() for cls in MEMORY_TOOLS]), PermissionPolicy(settings.permissions), AllowAllBroker()
    )
    return ctx, executor


def call(name: str, **args) -> ToolUseBlock:
    return ToolUseBlock(id=f"t_{name}", name=name, input=args)


async def test_record_and_search_roundtrip(ws: Path, manager: MemoryManager) -> None:
    ctx, ex = setup(ws, manager, task_id="task_7")
    res = await ex.execute(
        call("memory_record", kind="fact", content="The app listens on PORT 8000", key="port", files=["src/app.py"]),
        ctx,
    )
    assert not res.is_error, res.content
    assert "Recorded fact in project memory" in res.content
    [item] = manager.project.list()
    assert item.layer == "project" and item.kind == "fact" and item.key == "port"
    assert item.source == "agent:task_7" and list(item.file_refs) == ["src/app.py"]

    found = await ex.execute(call("memory_search", query="port"), ctx)
    assert not found.is_error
    assert "1. [project/fact conf=0.7] The app listens on PORT 8000 (source: agent:task_7)" in found.content
    assert "repository state wins" in found.content

    again = await ex.execute(
        call("memory_record", kind="fact", content="The app listens on PORT 9000", key="port", files=["src/app.py"]),
        ctx,
    )
    assert "version 2" in again.content
    assert manager.project.count() == 1


async def test_layer_routing_by_kind(ws: Path, manager: MemoryManager) -> None:
    ctx, ex = setup(ws, manager)
    for kind in ("fact", "decision", "lesson", "convention", "known_issue"):
        res = await ex.execute(call("memory_record", kind=kind, content=f"a {kind} worth keeping", confidence=0.9), ctx)
        assert not res.is_error, res.content
    assert manager.global_store is not None
    assert [(i.layer, i.kind) for i in manager.global_store.list()] == [("engineering", "lesson")]
    assert {(i.layer, i.kind) for i in manager.project.list()} == {
        ("project", "fact"), ("decision", "decision"), ("project", "convention"), ("project", "known_issue"),
    }
    assert all(i.source == "agent:session" and i.confidence == 0.9 for i in manager.project.list())

    res = await ex.execute(call("memory_search", query="worth keeping", layers=["engineering"]), ctx)
    assert "[engineering/lesson" in res.content and "[project/" not in res.content
    res = await ex.execute(call("memory_search", query="worth keeping", limit=2), ctx)
    assert res.content.count("worth keeping") == 2
    res = await ex.execute(call("memory_search", query="nothing matches this"), ctx)
    assert not res.is_error and res.content == "No matching memories."


async def test_search_marks_stale(ws: Path, manager: MemoryManager) -> None:
    ctx, ex = setup(ws, manager)
    await ex.execute(call("memory_record", kind="fact", content="PORT constant is 8000", files=["src/app.py"]), ctx)
    (ws / "src" / "app.py").write_text("PORT = 8080  # changed\n")
    res = await ex.execute(call("memory_search", query="PORT constant"), ctx)
    assert "STALE: src/app.py" in res.content


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"kind": "rumor", "content": "something long enough"}, "invalid arguments"),
        ({"kind": "fact", "content": "tiny"}, "invalid arguments"),
        ({"kind": "fact", "content": "valid content", "confidence": 1.5}, "invalid arguments"),
        ({"kind": "fact", "content": "valid content", "extra": 1}, "invalid arguments"),
        ({"kind": "fact", "content": "valid content", "files": ["../outside.py"]}, "outside the workspace"),
        ({"kind": "fact", "content": "valid content", "files": ["src/missing.py"]}, "file not found"),
    ],
)
async def test_record_rejects_bad_input(ws: Path, manager: MemoryManager, args: dict, expected: str) -> None:
    ctx, ex = setup(ws, manager)
    res = await ex.execute(call("memory_record", **args), ctx)
    assert res.is_error and expected in res.content
    assert manager.project.count() == 0


@pytest.mark.parametrize(
    "args", [{"query": "x", "limit": 0}, {"query": "x", "limit": 31}, {"query": "x", "layers": ["bogus"]}, {}]
)
async def test_search_rejects_bad_input(ws: Path, manager: MemoryManager, args: dict) -> None:
    ctx, ex = setup(ws, manager)
    res = await ex.execute(call("memory_search", **args), ctx)
    assert res.is_error and "invalid arguments" in res.content


async def test_memory_unavailable(ws: Path) -> None:
    ctx, ex = setup(ws, None)
    for c in (call("memory_search", query="x"), call("memory_record", kind="fact", content="valid content")):
        res = await ex.execute(c, ctx)
        assert res.is_error and "memory is not available" in res.content


async def test_safe_mode_allows_search_but_not_record(ws: Path, manager: MemoryManager) -> None:
    ctx, ex = setup(ws, manager, mode=Mode.SAFE)
    res = await ex.execute(call("memory_record", kind="fact", content="should be denied"), ctx)
    assert res.is_error and "permission denied" in res.content
    assert manager.project.count() == 0
    res = await ex.execute(call("memory_search", query="anything"), ctx)
    assert not res.is_error


async def test_secrets_are_redacted(ws: Path, manager: MemoryManager) -> None:
    ctx, ex = setup(ws, manager)
    res = await ex.execute(call("memory_record", kind="known_issue", content=f"CI token {FAKE_SECRET} expires"), ctx)
    assert not res.is_error
    found = await ex.execute(call("memory_search", query="CI token"), ctx)
    assert FAKE_SECRET not in found.content and "[REDACTED]" in found.content
    db = ws / ".agent" / "state.db"
    data = b"".join(p.read_bytes() for p in db.parent.glob("state.db*"))
    assert FAKE_SECRET.encode() not in data


def test_tool_metadata() -> None:
    from ai_engineer.config.settings import PermissionLevel
    from ai_engineer.tools.base import SideEffect

    assert [t.name for t in MEMORY_TOOLS] == ["memory_search", "memory_record"]
    assert MemorySearchTool.level == PermissionLevel.READ_ONLY and MemorySearchTool().concurrency_safe
    assert MemoryRecordTool.level == PermissionLevel.SAFE_WRITE and MemoryRecordTool.side_effect == SideEffect.WRITE
    spec = MemoryRecordTool().spec()
    assert spec.input_schema["properties"]["kind"]["enum"] == ["fact", "decision", "lesson", "convention", "known_issue"]
    assert set(spec.input_schema["required"]) == {"kind", "content"}
