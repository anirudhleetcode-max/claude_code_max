"""SQLite-backed, versioned memory store with FTS5 search and file-staleness tracking.

Every memory belongs to a *lineage*: changing a memory writes a new version row and
deactivates the previous one, so history is never lost. Items may reference
workspace files; the file's SHA-1 is captured at write time and compared on every
read, so memories about code that has since changed are flagged ``stale``.

The connection is shared across threads (callers use ``asyncio.to_thread``) and
guarded by a lock. When the SQLite build lacks FTS5, search degrades to ``LIKE``.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import re
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import StateError
from ..core.ids import new_id
from ..core.util import file_sha1, utcnow_iso
from ..security.secrets import Redactor

SCHEMA_VERSION = 1

_COLUMNS = (
    "seq, id, lineage, layer, kind, key, content, source, confidence, tags, file_refs, meta, "
    "version, created, updated, active, touch"
)
_TERM_RE = re.compile(r"\w+", re.UNICODE)
_MAX_TERMS = 32
_HASH_CACHE_MAX = 4096


class MemoryLayer(StrEnum):
    SESSION = "session"
    PROJECT = "project"
    ENGINEERING = "engineering"
    COMMAND = "command"
    DECISION = "decision"


LAYERS: frozenset[str] = frozenset(layer.value for layer in MemoryLayer)


class MemoryItem(BaseModel):
    """One version of one memory."""

    id: str
    lineage: str
    layer: str
    kind: str
    key: str | None = None
    content: str
    source: str = ""
    confidence: float = 0.7
    tags: list[str] = Field(default_factory=list)
    file_refs: dict[str, str] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    created: str
    updated: str
    active: bool = True
    stale: bool = False
    stale_files: list[str] = Field(default_factory=list)
    # Search relevance (higher is better); not persisted or exported.
    score: float = Field(default=0.0, exclude=True)

    def render(self, max_chars: int = 400) -> str:
        """One compact line for prompts."""
        conf = f"{self.confidence:.2f}".rstrip("0").rstrip(".") or "0"
        flags = f", STALE: {', '.join(self.stale_files)}" if self.stale else ""
        text = " ".join(self.content.split())
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        source = f" (source: {self.source})" if self.source else ""
        return f"[{self.layer}/{self.kind} conf={conf}{flags}] {text}{source}"


def _probe_fts5(conn: sqlite3.Connection) -> bool:
    """True when this SQLite build supports FTS5 (module-level so tests can patch it)."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._aie_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS temp._aie_fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression: quoted terms joined by OR."""
    terms = _terms(query)
    return " OR ".join(f'"{t}"' for t in terms)


def _terms(query: str) -> list[str]:
    seen: dict[str, None] = {}
    for term in _TERM_RE.findall(query.lower()):
        if term.strip("_"):
            seen.setdefault(term, None)
        if len(seen) >= _MAX_TERMS:
            break
    return list(seen)


def _like_escape(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class MemoryStore:
    """Persistent memory items in one SQLite database (tables prefixed ``memory_``)."""

    def __init__(self, db_path: Path, workspace: Path | None = None, redactor: Redactor | None = None) -> None:
        self.db_path = Path(db_path)
        self.workspace = workspace.resolve() if workspace is not None else None
        self.redactor = redactor or Redactor()
        self._lock = threading.RLock()
        self._hash_cache: dict[str, tuple[int, int, str | None]] = {}
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self.fts_enabled = False
        with self._lock:
            with contextlib.suppress(sqlite3.DatabaseError):
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._migrate()

    # ---- schema --------------------------------------------------------------------------

    @contextlib.contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise StateError("memory store is closed")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _migrate(self) -> None:
        with self._tx() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = conn.execute("SELECT value FROM memory_meta WHERE key = 'schema_version'").fetchone()
            current = int(row["value"]) if row else 0
            if current > SCHEMA_VERSION:
                raise StateError(
                    f"memory schema version {current} in {self.db_path} is newer than supported ({SCHEMA_VERSION})"
                )
            if current < 1:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_items (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT NOT NULL UNIQUE,
                        lineage TEXT NOT NULL,
                        layer TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        key TEXT,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT '',
                        confidence REAL NOT NULL DEFAULT 0.7,
                        tags TEXT NOT NULL DEFAULT '[]',
                        file_refs TEXT NOT NULL DEFAULT '{}',
                        meta TEXT NOT NULL DEFAULT '{}',
                        version INTEGER NOT NULL DEFAULT 1,
                        created TEXT NOT NULL,
                        updated TEXT NOT NULL,
                        active INTEGER NOT NULL DEFAULT 1,
                        touch INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS memory_items_lookup ON memory_items(layer, kind, key, active)")
                conn.execute("CREATE INDEX IF NOT EXISTS memory_items_lineage ON memory_items(lineage, version)")
                conn.execute("CREATE INDEX IF NOT EXISTS memory_items_touch ON memory_items(active, touch)")
            conn.execute(
                "INSERT OR REPLACE INTO memory_meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
            self._setup_fts(conn)

    def _setup_fts(self, conn: sqlite3.Connection) -> None:
        dirty = conn.execute("SELECT value FROM memory_meta WHERE key = 'fts_dirty'").fetchone()
        if not _probe_fts5(conn):
            self.fts_enabled = False
            # Writes made without FTS leave any existing index out of date; rebuild when FTS returns.
            conn.execute("INSERT OR REPLACE INTO memory_meta(key, value) VALUES ('fts_dirty', '1')")
            return
        existed = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
        ).fetchone()
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content, key, tags, tokenize = 'unicode61')"
            )
        except sqlite3.OperationalError:
            self.fts_enabled = False
            conn.execute("INSERT OR REPLACE INTO memory_meta(key, value) VALUES ('fts_dirty', '1')")
            return
        self.fts_enabled = True
        if not existed or (dirty and dirty["value"] == "1"):
            conn.execute("DELETE FROM memory_fts")
            rows = conn.execute("SELECT seq, content, key, tags FROM memory_items WHERE active = 1").fetchall()
            for row in rows:
                self._fts_insert(conn, row["seq"], row["content"], row["key"], json.loads(row["tags"]))
            conn.execute("DELETE FROM memory_meta WHERE key = 'fts_dirty'")

    def _fts_insert(self, conn: sqlite3.Connection, seq: int, content: str, key: str | None, tags: list[str]) -> None:
        if self.fts_enabled:
            conn.execute(
                "INSERT INTO memory_fts(rowid, content, key, tags) VALUES (?, ?, ?, ?)",
                (seq, content, key or "", " ".join(tags)),
            )

    def _fts_delete(self, conn: sqlite3.Connection, seqs: Iterable[int]) -> None:
        if self.fts_enabled:
            conn.executemany("DELETE FROM memory_fts WHERE rowid = ?", [(s,) for s in seqs])

    # ---- helpers -------------------------------------------------------------------------

    def _next_touch(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(touch), 0) + 1 AS t FROM memory_items").fetchone()
        return int(row["t"])

    def _redact(self, text: str) -> str:
        return self.redactor.redact_text(text) if text else text

    def _normalize_ref(self, ref: str) -> str:
        path = Path(ref)
        if self.workspace is not None and path.is_absolute():
            with contextlib.suppress(ValueError):
                path = Path(os.path.realpath(path)).relative_to(self.workspace)
        return Path(os.path.normpath(path)).as_posix()

    def _hash_ref(self, rel: str) -> str | None:
        if self.workspace is None:
            return None
        path = self.workspace / rel
        key = str(path)
        try:
            st = path.stat()
        except OSError:
            return None
        cached = self._hash_cache.get(key)
        if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]
        digest = file_sha1(path)
        if len(self._hash_cache) >= _HASH_CACHE_MAX:
            self._hash_cache.clear()
        self._hash_cache[key] = (st.st_mtime_ns, st.st_size, digest)
        return digest

    def _capture_refs(self, refs: Iterable[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for ref in refs:
            if not ref:
                continue
            rel = self._normalize_ref(ref)
            out[rel] = self._hash_ref(rel) or ""
        return out

    def _row_to_item(self, row: sqlite3.Row, score: float = 0.0) -> MemoryItem:
        item = MemoryItem(
            id=row["id"],
            lineage=row["lineage"],
            layer=row["layer"],
            kind=row["kind"],
            key=row["key"],
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            tags=json.loads(row["tags"]),
            file_refs=json.loads(row["file_refs"]),
            meta=json.loads(row["meta"]),
            version=row["version"],
            created=row["created"],
            updated=row["updated"],
            active=bool(row["active"]),
            score=score,
        )
        if self.workspace is not None and item.file_refs:
            stale = [rel for rel, digest in item.file_refs.items() if (self._hash_ref(rel) or "") != digest]
            item.stale = bool(stale)
            item.stale_files = sorted(stale)
        return item

    def _fetch(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            if self._closed:
                raise StateError("memory store is closed")
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _get_row(self, conn: sqlite3.Connection, item_id: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = conn.execute(
            f"SELECT {_COLUMNS} FROM memory_items WHERE id = ?", (item_id,)  # noqa: S608 - constant columns
        ).fetchone()
        return row

    def _insert(
        self,
        conn: sqlite3.Connection,
        *,
        lineage: str,
        layer: str,
        kind: str,
        key: str | None,
        content: str,
        source: str,
        confidence: float,
        tags: list[str],
        file_refs: dict[str, str],
        meta: dict[str, Any],
        version: int,
    ) -> str:
        item_id = new_id("mem")
        now = utcnow_iso()
        cur = conn.execute(
            "INSERT INTO memory_items(id, lineage, layer, kind, key, content, source, confidence, tags, file_refs, "
            "meta, version, created, updated, active, touch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                item_id, lineage, layer, kind, key, content, source, confidence,
                json.dumps(tags), json.dumps(file_refs, sort_keys=True), json.dumps(meta, default=str),
                version, now, now, self._next_touch(conn),
            ),
        )
        seq = cur.lastrowid
        assert seq is not None
        self._fts_insert(conn, seq, content, key, tags)
        return item_id

    def _deactivate(self, conn: sqlite3.Connection, where: str, params: tuple[Any, ...]) -> int:
        seqs = [r["seq"] for r in conn.execute(f"SELECT seq FROM memory_items WHERE active = 1 AND {where}", params)]  # noqa: S608
        if seqs:
            conn.executemany("UPDATE memory_items SET active = 0 WHERE seq = ?", [(s,) for s in seqs])
            self._fts_delete(conn, seqs)
        return len(seqs)

    def _clean_tags(self, tags: Iterable[str]) -> list[str]:
        out: dict[str, None] = {}
        for tag in tags:
            tag = self._redact(str(tag).strip())
            if tag:
                out.setdefault(tag, None)
        return list(out)

    # ---- public API ----------------------------------------------------------------------

    def add(
        self,
        *,
        layer: str,
        kind: str,
        content: str,
        key: str | None = None,
        source: str = "",
        confidence: float = 0.7,
        tags: Iterable[str] = (),
        file_refs: Iterable[str] = (),
        meta: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Store a memory (redacted). A keyed memory replaces the active one with the same key."""
        layer = str(layer)
        if layer not in LAYERS:
            raise ValueError(f"unknown memory layer {layer!r}; expected one of {sorted(LAYERS)}")
        if not kind or not kind.strip():
            raise ValueError("memory kind must not be empty")
        content = self._redact(content.strip())
        if not content:
            raise ValueError("memory content must not be empty")
        key = self._redact(key.strip()) if key and key.strip() else None
        source = self._redact(source)
        confidence = _clamp(confidence)
        tag_list = self._clean_tags(tags)
        refs = self._capture_refs(file_refs)
        clean_meta: dict[str, Any] = self.redactor.redact(dict(meta or {}))
        with self._tx() as conn:
            existing = None
            if key is not None:
                existing = conn.execute(
                    f"SELECT {_COLUMNS} FROM memory_items WHERE layer = ? AND kind = ? AND key = ? AND active = 1 "  # noqa: S608
                    "ORDER BY version DESC LIMIT 1",
                    (layer, kind, key),
                ).fetchone()
            if existing is not None and existing["content"] == content:
                merged_meta = {**json.loads(existing["meta"]), **clean_meta}
                merged_tags = self._clean_tags([*json.loads(existing["tags"]), *tag_list])
                merged_refs = {**json.loads(existing["file_refs"]), **refs}
                conn.execute(
                    "UPDATE memory_items SET updated = ?, confidence = ?, meta = ?, tags = ?, file_refs = ?, touch = ? "
                    "WHERE seq = ?",
                    (
                        utcnow_iso(), max(existing["confidence"], confidence), json.dumps(merged_meta, default=str),
                        json.dumps(merged_tags), json.dumps(merged_refs, sort_keys=True), self._next_touch(conn),
                        existing["seq"],
                    ),
                )
                if merged_tags != json.loads(existing["tags"]):
                    self._fts_delete(conn, [existing["seq"]])
                    self._fts_insert(conn, existing["seq"], content, key, merged_tags)
                item_id = existing["id"]
            else:
                lineage, version = new_id("lin"), 1
                if existing is not None:
                    lineage = existing["lineage"]
                    version = self._max_version(conn, lineage) + 1
                    self._deactivate(conn, "lineage = ?", (lineage,))
                item_id = self._insert(
                    conn, lineage=lineage, layer=layer, kind=kind, key=key, content=content, source=source,
                    confidence=confidence, tags=tag_list, file_refs=refs, meta=clean_meta, version=version,
                )
            row = self._get_row(conn, item_id)
        assert row is not None
        return self._row_to_item(row)

    def _max_version(self, conn: sqlite3.Connection, lineage: str) -> int:
        row = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM memory_items WHERE lineage = ?", (lineage,)).fetchone()
        return int(row["v"])

    def get(self, item_id: str) -> MemoryItem | None:
        rows = self._fetch(f"SELECT {_COLUMNS} FROM memory_items WHERE id = ?", (item_id,))  # noqa: S608
        return self._row_to_item(rows[0]) if rows else None

    def find(self, *, layer: str, kind: str, key: str) -> MemoryItem | None:
        """The active item with this (layer, kind, key), if any."""
        key = self._redact(key.strip())
        rows = self._fetch(
            f"SELECT {_COLUMNS} FROM memory_items WHERE layer = ? AND kind = ? AND key = ? AND active = 1 "  # noqa: S608
            "ORDER BY version DESC LIMIT 1",
            (layer, kind, key),
        )
        return self._row_to_item(rows[0]) if rows else None

    def update(
        self,
        item_id: str,
        *,
        content: str | None = None,
        confidence: float | None = None,
        tags: Iterable[str] | None = None,
        meta: dict[str, Any] | None = None,
        new_version: bool = True,
    ) -> MemoryItem:
        """Write a new version of an active item (KeyError if missing or inactive).

        ``meta`` is merged into the existing meta. ``new_version=False`` edits the row in
        place instead; that is meant for bookkeeping records such as command statistics.
        """
        with self._tx() as conn:
            row = self._get_row(conn, item_id)
            if row is None or not row["active"]:
                raise KeyError(f"no active memory item {item_id!r}")
            new_content = row["content"] if content is None else self._redact(content.strip())
            if not new_content:
                raise ValueError("memory content must not be empty")
            new_conf = row["confidence"] if confidence is None else _clamp(confidence)
            new_tags = json.loads(row["tags"]) if tags is None else self._clean_tags(tags)
            new_meta = {**json.loads(row["meta"]), **self.redactor.redact(dict(meta or {}))}
            refs: dict[str, str] = json.loads(row["file_refs"])
            if content is not None and new_content != row["content"]:
                refs = self._capture_refs(refs)  # re-asserted against the current files
            if new_version:
                self._deactivate(conn, "lineage = ?", (row["lineage"],))
                new_id_ = self._insert(
                    conn, lineage=row["lineage"], layer=row["layer"], kind=row["kind"], key=row["key"],
                    content=new_content, source=row["source"], confidence=new_conf, tags=new_tags, file_refs=refs,
                    meta=new_meta, version=self._max_version(conn, row["lineage"]) + 1,
                )
            else:
                new_id_ = item_id
                conn.execute(
                    "UPDATE memory_items SET content = ?, confidence = ?, tags = ?, meta = ?, file_refs = ?, "
                    "updated = ?, touch = ? WHERE seq = ?",
                    (
                        new_content, new_conf, json.dumps(new_tags), json.dumps(new_meta, default=str),
                        json.dumps(refs, sort_keys=True), utcnow_iso(), self._next_touch(conn), row["seq"],
                    ),
                )
                self._fts_delete(conn, [row["seq"]])
                self._fts_insert(conn, row["seq"], new_content, row["key"], new_tags)
            out = self._get_row(conn, new_id_)
        assert out is not None
        return self._row_to_item(out)

    def forget(self, item_id: str) -> bool:
        """Deactivate every version in the item's lineage. True if anything was active."""
        with self._tx() as conn:
            row = self._get_row(conn, item_id)
            if row is None:
                return False
            return self._deactivate(conn, "lineage = ?", (row["lineage"],)) > 0

    def search(
        self,
        query: str,
        *,
        layers: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
        limit: int = 10,
        include_stale: bool = True,
    ) -> list[MemoryItem]:
        """Active items ranked by relevance, confidence, then recency; stale items last."""
        if limit <= 0:
            return []
        filters, params = self._filters(layers, kinds, prefix="m.")
        terms = _terms(query)
        candidates = max(limit * 5, 50)
        scored: list[tuple[sqlite3.Row, float]]
        if not terms:
            rows = self._fetch(
                f"SELECT {_prefixed('m.')} FROM memory_items m WHERE m.active = 1{filters} "  # noqa: S608
                "ORDER BY m.touch DESC LIMIT ?",
                [*params, candidates],
            )
            scored = [(r, 0.0) for r in rows]
        elif self.fts_enabled:
            rows = self._fetch(
                f"SELECT {_prefixed('m.')}, bm25(memory_fts, 1.0, 2.0, 1.5) AS rank "  # noqa: S608
                f"FROM memory_fts JOIN memory_items m ON m.seq = memory_fts.rowid "
                f"WHERE memory_fts MATCH ? AND m.active = 1{filters} "
                "ORDER BY rank ASC, m.confidence DESC, m.touch DESC LIMIT ?",
                [fts_query(query), *params, candidates],
            )
            scored = [(r, round(-float(r["rank"]), 6)) for r in rows]
        else:
            scored = self._like_search(terms, filters, params)
        ordered = sorted(scored, key=lambda rs: (-rs[1], -rs[0]["confidence"], -rs[0]["touch"]))
        items = [self._row_to_item(row, score) for row, score in ordered]
        if not include_stale:
            items = [i for i in items if not i.stale]
        items.sort(key=lambda i: i.stale)  # stable: fresh first, ranking preserved
        return items[:limit]

    def _like_search(self, terms: list[str], filters: str, params: list[Any]) -> list[tuple[sqlite3.Row, float]]:
        clauses = []
        like_params: list[Any] = []
        for term in terms:
            pattern = f"%{_like_escape(term)}%"
            clauses.append("(m.content LIKE ? ESCAPE '\\' OR m.key LIKE ? ESCAPE '\\' OR m.tags LIKE ? ESCAPE '\\')")
            like_params += [pattern, pattern, pattern]
        rows = self._fetch(
            f"SELECT {_prefixed('m.')} FROM memory_items m WHERE m.active = 1{filters} "  # noqa: S608
            f"AND ({' OR '.join(clauses)}) ORDER BY m.confidence DESC, m.touch DESC LIMIT 2000",
            [*params, *like_params],
        )
        out = []
        for row in rows:
            haystack = " ".join((row["content"], row["key"] or "", row["tags"])).lower()
            hits = sum(haystack.count(t) for t in terms)
            matched = sum(1 for t in terms if t in haystack)
            out.append((row, round(matched + min(hits, 20) / 100, 4)))
        return out

    def _filters(
        self, layers: Iterable[str] | None, kinds: Iterable[str] | None, prefix: str = ""
    ) -> tuple[str, list[Any]]:
        sql = ""
        params: list[Any] = []
        for column, values in (("layer", layers), ("kind", kinds)):
            if values is None:
                continue
            vals = [str(v) for v in values]
            if not vals:
                continue
            sql += f" AND {prefix}{column} IN ({', '.join('?' for _ in vals)})"
            params += vals
        return sql, params

    def list(
        self, *, layer: str | None = None, kind: str | None = None, limit: int = 100, include_inactive: bool = False
    ) -> list[MemoryItem]:
        """Most recently touched items first."""
        filters, params = self._filters([layer] if layer else None, [kind] if kind else None)
        active = "1 = 1" if include_inactive else "active = 1"
        rows = self._fetch(
            f"SELECT {_COLUMNS} FROM memory_items WHERE {active}{filters} ORDER BY touch DESC, seq DESC LIMIT ?",  # noqa: S608
            [*params, max(0, limit)],
        )
        return [self._row_to_item(r) for r in rows]

    def history(self, item_id: str) -> builtins.list[MemoryItem]:
        """All versions in the item's lineage, oldest first ([] if unknown)."""
        rows = self._fetch(
            f"SELECT {_COLUMNS} FROM memory_items WHERE lineage = "  # noqa: S608
            "(SELECT lineage FROM memory_items WHERE id = ?) ORDER BY version ASC, seq ASC",
            (item_id,),
        )
        return [self._row_to_item(r) for r in rows]

    def export(self) -> builtins.list[dict[str, Any]]:
        """Active items as JSON-ready dicts, grouped by layer then oldest first."""
        rows = self._fetch(f"SELECT {_COLUMNS} FROM memory_items WHERE active = 1 ORDER BY layer, touch")  # noqa: S608
        return [self._row_to_item(r).model_dump(mode="json") for r in rows]

    def count(self, *, include_inactive: bool = False) -> int:
        where = "" if include_inactive else " WHERE active = 1"
        rows = self._fetch(f"SELECT COUNT(*) AS n FROM memory_items{where}")  # noqa: S608
        return int(rows[0]["n"])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _prefixed(prefix: str) -> str:
    return ", ".join(f"{prefix}{c.strip()}" for c in _COLUMNS.split(","))
