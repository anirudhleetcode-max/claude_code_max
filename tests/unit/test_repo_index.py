from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

import ai_engineer.repo.index as index_mod
from ai_engineer.repo.index import FileRecord, IndexStats, RepoIndex
from ai_engineer.repo.symbols import Symbol
from tests.unit.test_repo_discovery import (
    PYTHON_PROJECT,
    make_go_project,
    make_js_project,
    make_python_project,
    write,
)


def open_index(root: Path, tmp_path: Path, name: str = "index.db", **kw: int) -> RepoIndex:
    return RepoIndex(root, tmp_path / "state" / "indexes" / name, **kw)


@pytest.fixture
def py_index(tmp_path: Path) -> RepoIndex:
    root = make_python_project(tmp_path / "proj")
    idx = open_index(root, tmp_path)
    idx.refresh()
    yield idx
    idx.close()


def bump(path: Path, text: str) -> None:
    """Rewrite a file and force a different mtime (coarse filesystem clocks)."""
    st = path.stat()
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def test_initial_refresh_and_noop_refresh(tmp_path: Path) -> None:
    root = make_python_project(tmp_path / "proj")
    db = tmp_path / "deep" / "dir" / "index.db"
    with RepoIndex(root, db) as idx:
        first = idx.refresh()
        assert isinstance(first, IndexStats)
        assert (first.added, first.updated, first.removed, first.unchanged) == (len(PYTHON_PROJECT), 0, 0, 0)
        assert first.duration_s >= 0
        second = idx.refresh()
        assert (second.added, second.updated, second.removed, second.unchanged) == (0, 0, 0, len(PYTHON_PROJECT))
    assert db.exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_incremental_refresh_add_modify_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_python_project(tmp_path / "proj")
    idx = open_index(root, tmp_path)
    idx.refresh()

    parsed: list[str] = []
    real_extract = index_mod.extract_symbols

    def counting(path: str, text: str, language: str | None) -> list[Symbol]:
        parsed.append(path)
        return real_extract(path, text, language)

    monkeypatch.setattr(index_mod, "extract_symbols", counting)

    bump(root / "src/pkg/util.py", "def helper(x: int) -> int:\n    return x * 3\n\n\ndef extra() -> None: ...\n")
    write(root, {"src/pkg/new_mod.py": "def fresh() -> None: ...\n"})
    (root / "src/pkg/api.py").unlink()
    stats = idx.refresh()
    assert (stats.added, stats.updated, stats.removed) == (1, 1, 1)
    assert stats.unchanged == len(PYTHON_PROJECT) - 2
    assert sorted(parsed) == ["src/pkg/new_mod.py", "src/pkg/util.py"]
    assert idx.file("src/pkg/api.py") is None
    assert [s.name for s in idx.file_symbols("src/pkg/util.py")] == ["helper", "extra"]
    assert idx.find_symbol("fresh")[0]["path"] == "src/pkg/new_mod.py"
    assert idx.find_symbol("handle_request") == []

    # touching a file (new mtime, same content) is not a re-parse
    parsed.clear()
    core = root / "src/pkg/core.py"
    bump(core, core.read_text(encoding="utf-8"))
    stats = idx.refresh()
    assert (stats.added, stats.updated, stats.removed) == (0, 0, 0)
    assert parsed == []
    record = idx.file("src/pkg/core.py")
    assert record is not None and record.mtime == core.stat().st_mtime
    idx.close()


def test_refresh_with_explicit_paths(tmp_path: Path) -> None:
    root = make_python_project(tmp_path / "proj")
    idx = open_index(root, tmp_path)
    idx.refresh()
    bump(root / "src/pkg/util.py", "def helper2() -> None: ...\n")
    (root / "src/pkg/cli.py").unlink()
    write(root, {"untracked_by_refresh.py": "x = 1\n"})
    stats = idx.refresh(["src/pkg/util.py", "src/pkg/cli.py", str(root / "src/pkg/core.py"), "../escape.py"])
    assert (stats.added, stats.updated, stats.removed, stats.unchanged) == (0, 1, 1, 1)
    # paths not mentioned are left alone
    assert idx.file("untracked_by_refresh.py") is None
    assert idx.file("src/pkg/api.py") is not None
    idx.close()


def test_index_persists_across_reopen(tmp_path: Path) -> None:
    root = make_python_project(tmp_path / "proj")
    open_index(root, tmp_path).refresh()
    idx = open_index(root, tmp_path)
    stats = idx.refresh()
    assert stats.unchanged == len(PYTHON_PROJECT)
    assert stats.added == 0
    idx.close()


def test_schema_version_mismatch_rebuilds(tmp_path: Path) -> None:
    root = make_python_project(tmp_path / "proj")
    idx = open_index(root, tmp_path)
    idx.refresh()
    idx.close()
    db = tmp_path / "state" / "indexes" / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value = 'old' WHERE key = 'schema'")
    idx = open_index(root, tmp_path)
    assert idx.files() == []
    assert idx.refresh().added == len(PYTHON_PROJECT)
    idx.close()


def test_corrupt_database_is_rebuilt(tmp_path: Path) -> None:
    root = make_python_project(tmp_path / "proj")
    db = tmp_path / "state" / "indexes" / "index.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not a sqlite database" * 100)
    idx = open_index(root, tmp_path)
    assert idx.refresh().added == len(PYTHON_PROJECT)
    idx.close()


def test_file_records(py_index: RepoIndex) -> None:
    record = py_index.file("src/pkg/core.py")
    assert isinstance(record, FileRecord)
    assert record.language == "python"
    assert record.lines == 21
    assert len(record.sha1) == 40
    assert record.generated is False
    assert [r.path for r in py_index.files()] == sorted(PYTHON_PROJECT)
    assert py_index.file("nope.py") is None


def test_large_and_binary_files_are_not_parsed(tmp_path: Path) -> None:
    root = write(tmp_path / "proj", {"big.py": "def big():\n    pass\n" * 20, "small.py": "def small(): pass\n"})
    (root / "blob.bin").write_bytes(b"\x00\x01\x02" * 50)
    idx = open_index(root, tmp_path, max_parse_bytes=100)
    idx.refresh()
    big = idx.file("big.py")
    assert big is not None and big.sha1 == "" and big.lines == 0
    assert idx.file_symbols("big.py") == []
    assert [s.name for s in idx.file_symbols("small.py")] == ["small"]
    blob = idx.file("blob.bin")
    assert blob is not None and blob.lines == 0 and blob.language is None
    idx.close()


def test_find_symbol_ranking(py_index: RepoIndex) -> None:
    names = [r["name"] for r in py_index.find_symbol("engine")]
    assert names == ["engine", "Engine", "EngineFactory", "start_engine"]
    names = [r["name"] for r in py_index.find_symbol("Engine")]
    assert names[:2] == ["Engine", "engine"]
    hit = py_index.find_symbol("Engine", kind="class")[0]
    assert hit == {
        "path": "src/pkg/core.py",
        "name": "Engine",
        "kind": "class",
        "line": 6,
        "end_line": 8,
        "parent": None,
        "signature": "class Engine",
    }
    assert [r["name"] for r in py_index.find_symbol("eng", kind="function")] == ["engine", "start_engine"]
    assert py_index.find_symbol("engine", limit=2) == py_index.find_symbol("engine")[:2]
    assert py_index.find_symbol("") == []
    # LIKE wildcards are literal
    assert py_index.find_symbol("%") == []


def test_find_symbol_qualified_member(py_index: RepoIndex) -> None:
    results = py_index.find_symbol("Engine.run")
    assert [(r["name"], r["parent"]) for r in results] == [("run", "Engine")]
    assert py_index.find_symbol("EngineFactory.run") == []


def test_python_import_edges_and_dependents(py_index: RepoIndex) -> None:
    assert py_index.imports_of("src/pkg/core.py") == ["src/pkg/util.py"]
    assert py_index.imports_of("src/pkg/cli.py") == ["src/pkg/__init__.py", "src/pkg/api.py"]
    assert py_index.dependents("src/pkg/util.py", transitive=False) == ["src/pkg/core.py"]
    assert py_index.dependents("src/pkg/util.py") == [
        "src/pkg/core.py",
        "src/pkg/api.py",
        "tests/test_core.py",
        "src/pkg/cli.py",
        "src/pkg/__main__.py",
    ]
    assert py_index.dependents("src/pkg/util.py", limit=2) == ["src/pkg/core.py", "src/pkg/api.py"]
    assert py_index.dependents("tests/test_core.py") == []


def test_related_tests_by_imports_and_naming(py_index: RepoIndex) -> None:
    # test_core imports core (which imports util); test_util matches by name
    assert py_index.related_tests("src/pkg/util.py") == ["tests/test_core.py", "tests/test_util.py"]
    assert py_index.related_tests("src/pkg/core.py") == ["tests/test_core.py"]
    assert py_index.related_tests("tests/test_core.py") == ["tests/test_core.py"]
    assert py_index.related_tests("src/pkg/util.py", limit=1) == ["tests/test_core.py"]


def test_edges_follow_file_changes(tmp_path: Path) -> None:
    root = make_python_project(tmp_path / "proj")
    idx = open_index(root, tmp_path)
    idx.refresh()
    assert "src/pkg/api.py" in idx.dependents("src/pkg/core.py")
    # a new module that is imported before it exists resolves once it appears
    write(root, {"src/pkg/late.py": "from pkg import later\n"})
    idx.refresh()
    assert idx.imports_of("src/pkg/late.py") == ["src/pkg/__init__.py"]
    write(root, {"src/pkg/later.py": "X = 1\n"})
    idx.refresh()
    assert idx.imports_of("src/pkg/late.py") == ["src/pkg/__init__.py", "src/pkg/later.py"]
    idx.close()


def test_content_update_rewires_edges(tmp_path: Path) -> None:
    root = make_python_project(tmp_path / "proj")
    idx = open_index(root, tmp_path)
    idx.refresh()
    assert idx.imports_of("src/pkg/api.py") == ["src/pkg/core.py"]
    bump(root / "src/pkg/api.py", "from pkg.util import helper\n")
    stats = idx.refresh()
    assert (stats.added, stats.updated, stats.removed) == (0, 1, 0)
    assert idx.imports_of("src/pkg/api.py") == ["src/pkg/util.py"]
    assert "src/pkg/api.py" not in idx.dependents("src/pkg/core.py", transitive=False)
    assert idx.dependents("src/pkg/util.py", transitive=False) == ["src/pkg/api.py", "src/pkg/core.py"]
    # an unrelated file's edges are untouched
    assert idx.imports_of("src/pkg/core.py") == ["src/pkg/util.py"]
    idx.close()


def test_js_relative_import_resolution(tmp_path: Path) -> None:
    root = make_js_project(tmp_path / "web")
    write(root, {"src/esm.ts": "import { add } from './math.js';\nimport x from 'lodash';\n"})
    idx = open_index(root, tmp_path)
    idx.refresh()
    assert idx.imports_of("src/index.ts") == ["src/components/Button/index.tsx", "src/math.ts"]
    assert idx.imports_of("src/esm.ts") == ["src/math.ts"]
    assert idx.dependents("src/math.ts") == ["src/esm.ts", "src/index.ts", "src/math.test.ts"]
    assert idx.related_tests("src/math.ts") == ["src/math.test.ts"]
    assert [r["name"] for r in idx.find_symbol("add")] == ["add"]
    idx.close()


def test_go_package_resolution(tmp_path: Path) -> None:
    root = make_go_project(tmp_path / "go")
    idx = open_index(root, tmp_path)
    idx.refresh()
    assert idx.imports_of("main.go") == ["internal/store/store.go"]
    assert idx.dependents("internal/store/store.go") == ["main.go"]
    assert idx.related_tests("internal/store/store.go") == ["internal/store/store_test.go"]
    get = idx.find_symbol("Get")[0]
    assert (get["kind"], get["parent"]) == ("method", "Store")
    idx.close()


def test_rust_and_java_resolution(tmp_path: Path) -> None:
    root = write(
        tmp_path / "proj",
        {
            "Cargo.toml": "[package]\nname = 'x'\n",
            "src/lib.rs": "mod config;\npub mod net;\nuse crate::config::Settings;\n",
            "src/config.rs": "pub struct Settings;\n",
            "src/net/mod.rs": "mod http;\nuse super::config::Settings;\n",
            "src/net/http.rs": "pub fn get() {}\n",
            "app/src/main/java/com/acme/App.java": "package com.acme;\nimport com.acme.util.Strings;\nclass App {}\n",
            "app/src/main/java/com/acme/util/Strings.java": "package com.acme.util;\npublic class Strings {}\n",
        },
    )
    idx = open_index(root, tmp_path)
    idx.refresh()
    assert idx.imports_of("src/lib.rs") == ["src/config.rs", "src/net/mod.rs"]
    assert idx.imports_of("src/net/mod.rs") == ["src/config.rs", "src/net/http.rs"]
    assert idx.imports_of("app/src/main/java/com/acme/App.java") == ["app/src/main/java/com/acme/util/Strings.java"]
    idx.close()


SEARCH_FILES = {
    "src/payments/refunds.py": (
        "def issue_refund(invoice_id: str) -> None:\n"
        '    """Refund an invoice: the refund is credited to the original card."""\n'
        "    refund = create_refund(invoice_id)\n"
        "    notify_refund(refund)\n"
    ),
    "src/payments/invoices.py": "def render_invoice(invoice_id: str) -> str:\n    return invoice_id\n",
    "src/users/profile.ts": "export function getUserName(id: string): string {\n  return id;\n}\n",
    "docs/guide.md": "# Guide\n\nHow to configure logging levels.\n",
    "package-lock.json": '{"name": "refund refund refund invoice"}\n',
}


def check_search(idx: RepoIndex) -> None:
    results = idx.search("how do refunds of an invoice work refund")
    assert results[0]["path"] == "src/payments/refunds.py"
    assert results[0]["start_line"] == 1
    assert results[0]["end_line"] == 4
    assert "refund" in results[0]["snippet"].lower()
    assert len(results[0]["snippet"]) <= 400
    assert results[0]["score"] > results[-1]["score"] or len(results) == 1
    assert {r["path"] for r in results} >= {"src/payments/refunds.py", "src/payments/invoices.py"}
    # generated files are never chunked
    assert all(r["path"] != "package-lock.json" for r in results)
    # camelCase identifiers are searchable by their words
    assert idx.search("user name")[0]["path"] == "src/users/profile.ts"
    assert idx.search("logging")[0]["path"] == "docs/guide.md"
    assert idx.search("zzzz_nothing_matches") == []
    assert idx.search("   ") == []
    # FTS syntax in user input is treated literally
    assert isinstance(idx.search('refund" OR NEAR( AND * -'), list)
    assert len(idx.search("refund invoice", limit=1)) == 1


def test_bm25_search_with_fts(tmp_path: Path) -> None:
    root = write(tmp_path / "proj", SEARCH_FILES)
    idx = open_index(root, tmp_path)
    idx.refresh()
    assert idx.fts_enabled is index_mod.fts5_available()
    check_search(idx)
    idx.close()


def test_search_fallback_without_fts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_mod, "fts5_available", lambda: False)
    root = write(tmp_path / "proj", SEARCH_FILES)
    idx = open_index(root, tmp_path, name="plain.db")
    assert idx.fts_enabled is False
    idx.refresh()
    check_search(idx)
    assert idx.stats()["fts"] is False
    idx.close()


def test_fts_creation_failure_degrades_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_mod, "fts5_available", lambda: True)
    monkeypatch.setattr(index_mod, "_FTS_SCHEMA", "CREATE VIRTUAL TABLE chunks_fts USING no_such_module(x);")
    root = write(tmp_path / "proj", SEARCH_FILES)
    idx = open_index(root, tmp_path, name="degraded.db")
    assert idx.fts_enabled is False
    idx.refresh()
    assert idx.search("refund")[0]["path"] == "src/payments/refunds.py"
    idx.close()


def test_sensitive_files_are_not_chunked(tmp_path: Path) -> None:
    root = write(tmp_path / "proj", {".env": "DB_PASSWORD=hunter2hunter2\n", "app.py": "password_policy = 1\n"})
    idx = open_index(root, tmp_path)
    idx.refresh()
    assert [r["path"] for r in idx.search("hunter2hunter2")] == []
    assert idx.file(".env") is not None
    idx.close()


def test_stats(py_index: RepoIndex) -> None:
    stats = py_index.stats()
    assert stats["files"] == len(PYTHON_PROJECT)
    assert stats["symbols"] >= 10
    assert stats["edges"] >= 6
    assert stats["chunks"] >= 8
    assert stats["languages"]["python"] == 10
    assert stats["languages"]["markdown"] == 1
    assert stats["last_refresh"]


def test_schema_mismatch_rebuilds_even_when_file_deletion_fails(tmp_path: Path, monkeypatch) -> None:
    """Regression (seen on Windows): an open handle blocks deleting the DB file."""
    root = make_python_project(tmp_path / "proj")
    idx = open_index(root, tmp_path)
    idx.refresh()
    idx.close()
    db = tmp_path / "state" / "indexes" / "index.db"
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = 'old' WHERE key = 'schema'")
    conn.commit()
    monkeypatch.setattr(RepoIndex, "_delete_db_files", lambda self: None)  # deletion "fails"
    try:
        idx = open_index(root, tmp_path)
        assert idx.files() == []  # stale-schema data was not kept
        assert idx.refresh().added == len(PYTHON_PROJECT)
        idx.close()
    finally:
        conn.close()
