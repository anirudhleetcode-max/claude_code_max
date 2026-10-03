from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from ai_engineer.core.errors import StateError
from ai_engineer.memory import MemoryItem, MemoryLayer, MemoryManager, MemoryStore
from ai_engineer.memory import store as store_mod
from ai_engineer.memory.manager import CONTEXT_HEADER, command_confidence
from ai_engineer.memory.store import SCHEMA_VERSION, fts_query
from ai_engineer.security.secrets import Redactor

FAKE_SECRET = "ghp_" + "A" * 36


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n")
    (root / "b.py").write_text("b = 2\n")
    return root


@pytest.fixture
def store(tmp_path: Path, ws: Path):
    s = MemoryStore(tmp_path / "state" / "state.db", workspace=ws, redactor=Redactor(environ={}))
    yield s
    s.close()


@pytest.fixture
def no_fts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "_probe_fts5", lambda conn: False)


def _filler(store: MemoryStore, n: int = 8) -> None:
    topics = ["database", "frontend", "logging", "deployment", "caching", "routing", "styling", "metrics", "auth", "queue"]
    for topic in topics[:n]:
        store.add(layer="project", kind="fact", content=f"The {topic} layer is configured in its own module")


# ---- basics -------------------------------------------------------------------------------------


def test_add_get_and_render(store: MemoryStore) -> None:
    item = store.add(layer="project", kind="fact", content="Tests run with  pytest\n -q", source="agent:t1", confidence=0.8)
    assert item.id.startswith("mem_") and item.lineage.startswith("lin_")
    assert item.version == 1 and item.active and not item.stale
    got = store.get(item.id)
    assert got is not None and got.content == "Tests run with  pytest\n -q"
    assert got.render() == "[project/fact conf=0.8] Tests run with pytest -q (source: agent:t1)"
    assert store.get("mem_missing") is None
    assert store.count() == 1
    assert MemoryLayer.ENGINEERING == "engineering"


def test_render_truncates_and_omits_empty_source() -> None:
    item = MemoryItem(id="x", lineage="l", layer="project", kind="fact", content="word " * 200, created="t", updated="t")
    line = item.render(max_chars=50)
    assert line.startswith("[project/fact conf=0.7] ")
    assert line.endswith("…") and "(source:" not in line
    assert len(line) < 80


def test_validation_and_clamping(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="unknown memory layer"):
        store.add(layer="bogus", kind="fact", content="something")
    with pytest.raises(ValueError, match="content"):
        store.add(layer="project", kind="fact", content="   ")
    with pytest.raises(ValueError, match="kind"):
        store.add(layer="project", kind=" ", content="something")
    assert store.add(layer="project", kind="fact", content="too sure", confidence=7).confidence == 1.0
    assert store.add(layer="project", kind="fact", content="too unsure", confidence=-1).confidence == 0.0


# ---- versioning ---------------------------------------------------------------------------------


def test_keyed_identical_content_dedupes(store: MemoryStore) -> None:
    first = store.add(layer="project", kind="convention", key="style", content="Use black", confidence=0.6, tags=["fmt"])
    again = store.add(
        layer="project", kind="convention", key="style", content="Use black", confidence=0.9, tags=["style"],
        meta={"seen": 2},
    )
    assert again.id == first.id and again.version == 1
    assert again.confidence == 0.9
    assert again.tags == ["fmt", "style"] and again.meta == {"seen": 2}
    # Confidence never drops on a re-assertion.
    assert store.add(layer="project", kind="convention", key="style", content="Use black", confidence=0.1).confidence == 0.9
    assert store.count() == 1 and store.count(include_inactive=True) == 1
    # Same key in another kind or layer is a separate memory.
    other = store.add(layer="project", kind="fact", key="style", content="Use black")
    assert other.lineage != first.lineage


def test_keyed_different_content_creates_version(store: MemoryStore) -> None:
    v1 = store.add(layer="project", kind="fact", key="db", content="Uses SQLite")
    v2 = store.add(layer="project", kind="fact", key="db", content="Uses PostgreSQL 16")
    assert v2.id != v1.id and v2.lineage == v1.lineage and v2.version == 2
    old = store.get(v1.id)
    assert old is not None and not old.active
    hist = store.history(v2.id)
    assert [h.version for h in hist] == [1, 2]
    assert [h.content for h in hist] == ["Uses SQLite", "Uses PostgreSQL 16"]
    assert store.history(v1.id) == hist
    assert store.history("nope") == []
    assert store.count() == 1 and store.count(include_inactive=True) == 2
    assert [i.id for i in store.search("SQLite PostgreSQL")] == [v2.id]


def test_unkeyed_adds_are_independent(store: MemoryStore) -> None:
    a = store.add(layer="project", kind="fact", content="same text")
    b = store.add(layer="project", kind="fact", content="same text")
    assert a.id != b.id and a.lineage != b.lineage and store.count() == 2


def test_update_creates_new_version(store: MemoryStore) -> None:
    v1 = store.add(layer="project", kind="fact", content="Port is 8000", tags=["config"], meta={"a": 1})
    v2 = store.update(v1.id, content="Port is 8080", confidence=0.95)
    assert v2.version == 2 and v2.lineage == v1.lineage and v2.id != v1.id
    assert v2.tags == ["config"] and v2.meta == {"a": 1} and v2.confidence == 0.95
    v3 = store.update(v2.id, tags=["config", "network"])
    assert v3.version == 3 and v3.content == "Port is 8080" and v3.tags == ["config", "network"]
    with pytest.raises(KeyError):
        store.update(v1.id, content="stale version")  # inactive
    with pytest.raises(KeyError):
        store.update("mem_missing", content="x")
    assert [i.version for i in store.history(v1.id)] == [1, 2, 3]
    assert store.search("network")[0].id == v3.id


def test_update_in_place_keeps_identity(store: MemoryStore) -> None:
    v1 = store.add(layer="command", kind="test", key="test:pytest", content="pytest passed", meta={"successes": 1})
    same = store.update(v1.id, content="pytest failed", meta={"failures": 1}, new_version=False)
    assert same.id == v1.id and same.version == 1
    assert same.meta == {"successes": 1, "failures": 1}
    assert len(store.history(v1.id)) == 1
    assert [i.id for i in store.search("failed")] == [v1.id]
    assert store.search("passed") == []


def test_forget_deactivates_lineage(store: MemoryStore) -> None:
    v1 = store.add(layer="project", kind="fact", key="k", content="first statement")
    v2 = store.add(layer="project", kind="fact", key="k", content="second statement")
    keep = store.add(layer="project", kind="fact", content="unrelated statement")
    assert store.forget(v1.id) is True
    assert store.forget(v2.id) is False
    assert store.forget("mem_missing") is False
    assert all(not i.active for i in store.history(v2.id))
    assert [i.id for i in store.search("statement")] == [keep.id]
    assert store.count() == 1
    assert {i.id for i in store.list(include_inactive=True)} == {v1.id, v2.id, keep.id}
    # The key is free again: a new lineage starts.
    fresh = store.add(layer="project", kind="fact", key="k", content="third statement")
    assert fresh.lineage != v1.lineage and fresh.version == 1


# ---- redaction ----------------------------------------------------------------------------------


def _db_bytes(db: Path) -> bytes:
    return b"".join(p.read_bytes() for p in db.parent.glob(db.name + "*") if p.is_file())


def test_secrets_never_reach_disk(tmp_path: Path, ws: Path) -> None:
    db = tmp_path / "mem" / "state.db"
    store = MemoryStore(db, workspace=ws)  # default redactor
    item = store.add(
        layer="project", kind="fact", key=f"token-{FAKE_SECRET}", content=f"The CI token is {FAKE_SECRET} for deploys",
        source=f"https://x.test/?t={FAKE_SECRET}", tags=[FAKE_SECRET], meta={"nested": {"token": FAKE_SECRET}},
    )
    assert FAKE_SECRET not in item.model_dump_json()
    assert "[REDACTED]" in item.content and "[REDACTED]" in item.source
    updated = store.update(item.id, content=f"rotated to {FAKE_SECRET}")
    assert FAKE_SECRET not in updated.content
    assert FAKE_SECRET.encode() not in _db_bytes(db)  # including the WAL while open
    store.close()
    assert FAKE_SECRET.encode() not in _db_bytes(db)


def test_custom_redactor_values(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.db", redactor=Redactor(extra_values=["hunter2-local-pw"], environ={}))
    item = store.add(layer="session", kind="note", content="password hunter2-local-pw works")
    assert "hunter2-local-pw" not in item.content
    store.close()


# ---- staleness ----------------------------------------------------------------------------------


def test_stale_after_edit_and_delete(store: MemoryStore, ws: Path) -> None:
    item = store.add(layer="project", kind="fact", content="a and b define constants", file_refs=["a.py", str(ws / "b.py")])
    assert set(item.file_refs) == {"a.py", "b.py"} and all(item.file_refs.values())
    assert not item.stale
    (ws / "a.py").write_text("a = 1000\n")
    got = store.get(item.id)
    assert got is not None and got.stale and got.stale_files == ["a.py"]
    assert "STALE: a.py" in got.render()
    (ws / "b.py").unlink()
    got = store.get(item.id)
    assert got is not None and got.stale_files == ["a.py", "b.py"]
    assert "STALE: a.py, b.py" in got.render()


def test_missing_file_at_write_then_created(store: MemoryStore, ws: Path) -> None:
    item = store.add(layer="project", kind="known_issue", content="c.py will hold the cache", file_refs=["c.py"])
    assert item.file_refs == {"c.py": ""} and not item.stale
    (ws / "c.py").write_text("cache = {}\n")
    got = store.get(item.id)
    assert got is not None and got.stale and got.stale_files == ["c.py"]


def test_update_reasserts_file_hashes(store: MemoryStore, ws: Path) -> None:
    item = store.add(layer="project", kind="fact", content="a holds one constant", file_refs=["a.py"])
    (ws / "a.py").write_text("a = 1\nb = 2\n")
    assert store.get(item.id).stale  # type: ignore[union-attr]
    v2 = store.update(item.id, confidence=0.9)
    assert v2.stale  # a confidence-only change keeps the old hashes
    refreshed = store.update(v2.id, content="a holds two constants")
    assert not refreshed.stale and refreshed.version == 3


def test_no_workspace_never_stale(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "global.db")
    item = store.add(layer="engineering", kind="lesson", content="pin dependencies", file_refs=["x.py"])
    assert item.file_refs == {"x.py": ""} and not item.stale
    store.close()


def test_hash_cache_avoids_rehashing(store: MemoryStore, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    real = store_mod.file_sha1

    def counting(path: Path) -> str | None:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(store_mod, "file_sha1", counting)
    item = store.add(layer="project", kind="fact", content="a module constant", file_refs=["a.py"])
    for _ in range(5):
        store.get(item.id)
        store.search("constant")
    assert len(calls) == 1
    (ws / "a.py").write_text("a = 123456\n")
    assert store.get(item.id).stale  # type: ignore[union-attr]
    assert len(calls) == 2


# ---- search -------------------------------------------------------------------------------------


def test_fts_query_escaping() -> None:
    assert fts_query('parser "AND" (tokenizer) OR NOT x*') == '"parser" OR "and" OR "tokenizer" OR "or" OR "not" OR "x"'
    assert fts_query("  ") == ""
    assert fts_query("___ !!") == ""


@pytest.mark.parametrize("fts", [True, False], ids=["fts5", "like"])
def test_search_relevance_and_filters(tmp_path: Path, ws: Path, monkeypatch: pytest.MonkeyPatch, fts: bool) -> None:
    if not fts:
        monkeypatch.setattr(store_mod, "_probe_fts5", lambda conn: False)
    store = MemoryStore(tmp_path / "s.db", workspace=ws)
    assert store.fts_enabled is fts
    _filler(store)
    both = store.add(layer="project", kind="fact", content="The parser feeds the tokenizer output to the AST builder")
    one = store.add(layer="project", kind="fact", content="The parser lives in src/parse.py")
    lesson = store.add(layer="engineering", kind="lesson", content="Write a tokenizer before the parser")
    tagged = store.add(layer="project", kind="convention", content="Grammar files are generated", tags=["parser"])
    keyed = store.add(layer="decision", kind="adr", key="parser choice", content="Use a hand-written approach")

    results = store.search("parser tokenizer")
    ids = [i.id for i in results]
    assert ids[0] in {both.id, lesson.id}
    assert set(ids) == {both.id, one.id, lesson.id, tagged.id, keyed.id}  # content, tags and key are indexed
    assert ids.index(both.id) < ids.index(one.id)
    assert all(i.score >= 0 for i in results)

    assert {i.id for i in store.search("parser", layers=["engineering"])} == {lesson.id}
    assert {i.id for i in store.search("parser", kinds=["fact"])} == {both.id, one.id}
    assert {i.id for i in store.search("parser", layers=["project"], kinds=["convention", "fact"])} == {
        both.id, one.id, tagged.id,
    }
    assert store.search("parser", layers=[]) != []  # empty filter means no filter
    assert len(store.search("parser", limit=2)) == 2
    assert store.search("parser", limit=0) == []
    assert store.search("nonexistentword") == []
    # Hostile queries never raise.
    for q in ['"', "parser'", "AND OR NOT", "parse*", "a:b", "(", "NEAR(x y)", "100%", "under_score", "\\"]:
        store.search(q)
    store.close()


@pytest.mark.parametrize("fts", [True, False], ids=["fts5", "like"])
def test_search_ties_broken_by_confidence_then_recency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fts: bool
) -> None:
    if not fts:
        monkeypatch.setattr(store_mod, "_probe_fts5", lambda conn: False)
    store = MemoryStore(tmp_path / "s.db")
    _filler(store)
    low = store.add(layer="project", kind="fact", content="retry wrapper handles flaky network", confidence=0.3)
    high = store.add(layer="project", kind="fact", content="retry wrapper handles flaky network", confidence=0.9)
    older = store.add(layer="project", kind="fact", content="retry wrapper handles flaky network", confidence=0.6)
    newer = store.add(layer="project", kind="fact", content="retry wrapper handles flaky network", confidence=0.6)
    assert [i.id for i in store.search("flaky")] == [high.id, newer.id, older.id, low.id]
    store.close()


def test_search_pushes_stale_items_last(store: MemoryStore, ws: Path) -> None:
    _filler(store)
    stale = store.add(
        layer="project", kind="fact", content="config loader loader loader parses config", confidence=1.0,
        file_refs=["a.py"],
    )
    fresh = store.add(layer="project", kind="fact", content="the config loader is lazy", confidence=0.2)
    assert store.search("config loader")[0].id == stale.id
    (ws / "a.py").write_text("changed = True\n")
    results = store.search("config loader")
    assert [i.id for i in results] == [fresh.id, stale.id]
    assert results[1].stale
    assert [i.id for i in store.search("config loader", include_stale=False)] == [fresh.id]


def test_blank_query_returns_most_recent(store: MemoryStore) -> None:
    items = [store.add(layer="project", kind="fact", content=f"fact number {n}") for n in range(5)]
    assert [i.id for i in store.search("")] == [i.id for i in reversed(items)]
    assert [i.id for i in store.search("   ", limit=2)] == [items[4].id, items[3].id]
    bumped = store.add(layer="project", kind="fact", key="k", content="keyed fact")
    store.add(layer="project", kind="fact", content="later fact")
    store.add(layer="project", kind="fact", key="k", content="keyed fact")  # re-assert -> most recent
    assert store.search("")[0].id == bumped.id
    assert [i.id for i in store.search("", layers=["engineering"])] == []


def test_list_and_export(store: MemoryStore) -> None:
    a = store.add(layer="project", kind="fact", content="alpha fact")
    b = store.add(layer="command", kind="test", key="test:pytest", content="pytest works")
    c = store.add(layer="project", kind="convention", content="gamma convention")
    old = store.add(layer="project", kind="fact", key="k", content="old")
    new = store.add(layer="project", kind="fact", key="k", content="new")
    assert [i.id for i in store.list()] == [new.id, c.id, b.id, a.id]
    assert [i.id for i in store.list(layer="project", kind="fact")] == [new.id, a.id]
    assert len(store.list(limit=2)) == 2
    assert old.id in {i.id for i in store.list(include_inactive=True)}
    exported = store.export()
    assert {d["id"] for d in exported} == {a.id, b.id, c.id, new.id}
    assert all("score" not in d and d["active"] for d in exported)
    json.dumps(exported)


# ---- schema / FTS fallback ----------------------------------------------------------------------


def test_schema_version_and_idempotent_reopen(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    s1 = MemoryStore(db)
    item = s1.add(layer="project", kind="fact", content="persisted fact")
    s1.close()
    s1.close()  # idempotent
    s2 = MemoryStore(db)
    assert s2.get(item.id) is not None and s2.count() == 1
    assert [i.id for i in s2.search("persisted")] == [item.id]
    s2.close()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT value FROM memory_meta WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        conn.execute("UPDATE memory_meta SET value='999' WHERE key='schema_version'")
    conn.close()
    with pytest.raises(StateError, match="newer"):
        MemoryStore(db)


def test_closed_store_raises(tmp_path: Path) -> None:
    s = MemoryStore(tmp_path / "s.db")
    s.close()
    with pytest.raises(StateError):
        s.add(layer="project", kind="fact", content="after close")
    with pytest.raises(StateError):
        s.search("x")


def test_fts_index_rebuilt_after_fallback_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "s.db"
    MemoryStore(db).close()  # creates the FTS table
    monkeypatch.setattr(store_mod, "_probe_fts5", lambda conn: False)
    fallback = MemoryStore(db)
    assert not fallback.fts_enabled
    item = fallback.add(layer="project", kind="fact", content="written without the fulltext index")
    assert [i.id for i in fallback.search("fulltext")] == [item.id]
    fallback.close()
    monkeypatch.undo()
    restored = MemoryStore(db)
    assert restored.fts_enabled
    assert [i.id for i in restored.search("fulltext")] == [item.id]
    restored.close()


def test_like_fallback_escapes_wildcards(tmp_path: Path, no_fts: None) -> None:
    store = MemoryStore(tmp_path / "s.db")
    store.add(layer="project", kind="fact", content="coverage is 100 percent")
    pct = store.add(layer="project", kind="fact", content="use snake_case names")
    assert [i.id for i in store.search("snake_case")] == [pct.id]
    assert store.search("%") == store.search("")  # no searchable terms -> recent items
    store.close()


async def test_concurrent_use_from_threads(store: MemoryStore) -> None:
    await asyncio.gather(
        *(asyncio.to_thread(store.add, layer="project", kind="fact", content=f"parallel fact {n}") for n in range(40))
    )
    assert store.count() == 40
    await asyncio.gather(
        *(
            asyncio.to_thread(store.add, layer="project", kind="fact", key="shared", content=f"version {n}")
            for n in range(20)
        ),
        *(asyncio.to_thread(store.search, "parallel") for _ in range(10)),
    )
    active = store.list(layer="project", kind="fact", limit=1000)
    shared = [i for i in active if i.key == "shared"]
    assert len(shared) == 1
    assert [i.version for i in store.history(shared[0].id)] == list(range(1, 21))


# ---- manager ------------------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path, ws: Path):
    project = MemoryStore(tmp_path / "proj" / "state.db", workspace=ws)
    global_store = MemoryStore(tmp_path / "home" / "memory.db")
    mgr = MemoryManager(project, global_store)
    yield mgr
    mgr.close()


def test_manager_routes_engineering_to_global(manager: MemoryManager, tmp_path: Path) -> None:
    lesson = manager.add(layer="engineering", kind="lesson", content="Prefer small pure functions")
    fact = manager.add(layer="project", kind="fact", content="Project uses small modules")
    assert manager.global_store is not None
    assert manager.global_store.get(lesson.id) is not None and manager.project.get(lesson.id) is None
    assert manager.project.get(fact.id) is not None
    solo = MemoryManager(MemoryStore(tmp_path / "solo.db"))
    item = solo.add(layer="engineering", kind="lesson", content="Without a global store lessons stay local")
    assert solo.project.get(item.id) is not None
    solo.close()


def test_manager_search_merges_project_first_on_ties(manager: MemoryManager) -> None:
    g = manager.add(layer="engineering", kind="lesson", content="cache invalidation needs explicit keys")
    p = manager.add(layer="project", kind="fact", content="cache invalidation needs explicit keys")
    results = manager.search("cache invalidation")
    assert [i.id for i in results] == [p.id, g.id]
    assert [i.id for i in manager.search("cache", layers=["engineering"])] == [g.id]
    assert [i.id for i in manager.search("cache", kinds=["fact"])] == [p.id]
    assert len(manager.search("cache", limit=1)) == 1
    better = manager.add(layer="engineering", kind="lesson", content="cache cache cache invalidation", confidence=0.9)
    assert manager.search("cache")[0].id in {better.id, p.id}


def test_context_block(manager: MemoryManager, ws: Path) -> None:
    assert manager.context_block("anything") == ""
    manager.add(layer="project", kind="fact", content="The scheduler module retries jobs", file_refs=["a.py"])
    manager.add(layer="engineering", kind="lesson", content="Scheduler retries need jitter")
    block = manager.context_block("scheduler retries")
    lines = block.splitlines()
    assert lines[0] == CONTEXT_HEADER
    assert len([line for line in lines if line.startswith("- [")]) == 2
    assert "STALE" not in block
    (ws / "a.py").write_text("rewritten = True\n")
    block = manager.context_block("scheduler retries")
    assert "STALE: a.py" in block and "re-check" in block
    # Stale items come last.
    assert block.splitlines()[2].startswith("- [project/fact") and "STALE" in block.splitlines()[2]
    assert len(manager.context_block("scheduler", limit=1).splitlines()) == 2


def test_record_command_confidence_growth(manager: MemoryManager) -> None:
    first = manager.record_command("pytest -q", kind="test", ok=True, duration_s=1.25)
    assert first.layer == "command" and first.key == "test:pytest -q"
    assert first.meta["successes"] == 1 and first.meta["failures"] == 0 and first.meta["last_ok"] is True
    assert first.confidence == pytest.approx(0.6)
    confs = [first.confidence]
    for _ in range(6):
        confs.append(manager.record_command("pytest -q", kind="test", ok=True, duration_s=1.0).confidence)
    assert confs == sorted(confs) and confs[-1] == pytest.approx(0.95)
    failed = manager.record_command("pytest -q", kind="test", ok=False, duration_s=3.0, note="2 tests failed")
    assert failed.id == first.id  # statistics update in place
    assert failed.meta == {**failed.meta, "successes": 7, "failures": 1, "last_ok": False, "last_duration_s": 3.0}
    assert "failed" in failed.content and "2 tests failed" in failed.content
    assert len(manager.project.history(first.id)) == 1
    assert manager.project.count() == 1


def test_record_command_never_succeeding_loses_confidence(manager: MemoryManager) -> None:
    confs = [manager.record_command("make lint", kind="lint", ok=False, duration_s=0.1).confidence for _ in range(5)]
    assert confs == sorted(confs, reverse=True) and confs[-1] == pytest.approx(0.2)
    assert command_confidence(0, 100) == 0.2 and command_confidence(100, 0) == 0.95


def test_record_command_redacts(manager: MemoryManager) -> None:
    item = manager.record_command(f"curl -H 'Authorization: token {FAKE_SECRET}' x", kind="smoke", ok=True, duration_s=1)
    again = manager.record_command(f"curl -H 'Authorization: token {FAKE_SECRET}' x", kind="smoke", ok=True, duration_s=1)
    assert FAKE_SECRET not in (item.key or "") and FAKE_SECRET not in item.content
    assert again.id == item.id and again.meta["successes"] == 2


def test_known_commands_ordered_by_confidence(manager: MemoryManager) -> None:
    for _ in range(3):
        manager.record_command("pytest", kind="test", ok=True, duration_s=1)
    manager.record_command("tox", kind="test", ok=False, duration_s=1)
    manager.record_command("python -m unittest", kind="test", ok=True, duration_s=1)
    manager.record_command("ruff check .", kind="lint", ok=True, duration_s=1)
    known = manager.known_commands("test")
    assert [i.key for i in known] == ["test:pytest", "test:python -m unittest", "test:tox"]
    assert manager.known_commands("build") == []


def test_record_decision_and_supersede(manager: MemoryManager) -> None:
    d1 = manager.record_decision(
        "Storage engine", "Use SQLite", "Zero setup", alternatives=["PostgreSQL", "files"], task_id="task_1"
    )
    assert d1.layer == "decision" and d1.kind == "adr" and d1.key == "Storage engine"
    assert "Use SQLite" in d1.content and "Zero setup" in d1.content and "PostgreSQL" in d1.content
    assert d1.source == "task:task_1" and d1.meta["alternatives"] == ["PostgreSQL", "files"]
    d2 = manager.record_decision("Storage engine", "Use PostgreSQL", "Needs concurrency")
    assert d2.lineage == d1.lineage and d2.version == 2 and d2.source == "agent"
    assert [i.id for i in manager.search("storage", layers=["decision"])] == [d2.id]


def test_write_snapshot(manager: MemoryManager, tmp_path: Path) -> None:
    fact = manager.add(layer="project", kind="fact", content="Snapshot fact")
    cmd = manager.record_command("pytest", kind="test", ok=True, duration_s=1)
    adr = manager.record_decision("Logging", "Use stdlib logging", "No dependency")
    manager.add(layer="engineering", kind="lesson", content="global lessons are not in the project snapshot")
    out = tmp_path / ".agent" / "context.json"
    manager.write_snapshot(out, extra={"workspace": "demo"})
    data = json.loads(out.read_text())
    assert set(data) == {"generated", "project", "decisions", "workspace"}
    assert data["workspace"] == "demo"
    assert [d["id"] for d in data["project"]["project"]] == [fact.id]
    assert [d["id"] for d in data["project"]["command"]] == [cmd.id]
    assert [d["id"] for d in data["decisions"]] == [adr.id]
    assert "engineering" not in data["project"]
    assert not list(out.parent.glob("*.tmp"))
