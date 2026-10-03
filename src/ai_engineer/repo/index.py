"""Incremental SQLite index: files, symbols, import graph and BM25 text chunks.

Files are re-parsed only when their size/mtime changed *and* their content hash
differs. Reading and parsing run in a thread pool; all SQLite writes happen on
the calling thread. Full-text search uses FTS5 when the local SQLite supports it
and falls back to a LIKE scan with Python-side BM25 scoring otherwise.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import posixpath
import re
import sqlite3
import stat
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ai_engineer.core.util import is_binary_bytes, sha1_bytes, utcnow_iso
from ai_engineer.repo.files import (
    is_generated_content,
    is_generated_path,
    is_test_path,
    language_of,
    list_files,
)
from ai_engineer.repo.symbols import Symbol, extract_imports, extract_symbols

SCHEMA_VERSION = "1"
MAX_PARSE_BYTES = 1_000_000
CHUNK_LINES = 60
_BATCH = 256
_SNIPPET_CHARS = 400
_MAX_QUERY_TERMS = 32


class IndexStats(BaseModel):
    """Outcome of :meth:`RepoIndex.refresh`."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    duration_s: float = 0.0


class FileRecord(BaseModel):
    """Indexed metadata for one file (``sha1`` is empty for files over the parse cap)."""

    path: str
    language: str | None
    size: int
    mtime: float
    sha1: str
    lines: int
    generated: bool


@functools.lru_cache(maxsize=1)
def fts5_available() -> bool:
    """Whether this interpreter's SQLite can create FTS5 tables."""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return True


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(
    path TEXT PRIMARY KEY,
    language TEXT,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha1 TEXT NOT NULL,
    lines INTEGER NOT NULL,
    generated INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS symbols(
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    name_lower TEXT NOT NULL,
    kind TEXT NOT NULL,
    line INTEGER NOT NULL,
    end_line INTEGER,
    parent TEXT,
    signature TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS symbols_path ON symbols(path);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name_lower);
CREATE TABLE IF NOT EXISTS imports(
    path TEXT NOT NULL,
    seq INTEGER NOT NULL,
    spec TEXT NOT NULL,
    PRIMARY KEY(path, seq)
);
CREATE TABLE IF NOT EXISTS edges(src TEXT NOT NULL, dst TEXT NOT NULL, PRIMARY KEY(src, dst));
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    terms TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    path, content, terms, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, path, content, terms) VALUES (new.id, new.path, new.content, new.terms);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, path, content, terms)
    VALUES ('delete', old.id, old.path, old.content, old.terms);
END;
"""

# File names whose content should never be copied into the search index.
_SENSITIVE_NAME = re.compile(
    r"(?i)^(\.env(\..+)?|.*\.(pem|key|p12|pfx|jks|keystore)|id_(rsa|dsa|ecdsa|ed25519)(\..*)?|\.netrc|\.pypirc|"
    r"\.npmrc|credentials(\..*)?|secrets?\.(ya?ml|json|toml))$"
)
_SAFE_ENV_TEMPLATES = (".env.example", ".env.sample", ".env.template", ".env.dist")


def _is_sensitive(path: str) -> bool:
    name = posixpath.basename(path).lower()
    return name not in _SAFE_ENV_TEMPLATES and bool(_SENSITIVE_NAME.match(name))


_CAMEL_IDENT = re.compile(r"\b[A-Za-z][A-Za-z0-9]*?[a-z0-9][A-Z][A-Za-z0-9]*\b")
_CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _camel_parts(word: str) -> list[str]:
    return [p.lower() for p in _CAMEL_PART.findall(word) if len(p) >= 2]


def _camel_terms(text: str, cap: int = 2000) -> str:
    seen: dict[str, None] = {}
    for m in _CAMEL_IDENT.finditer(text):
        for part in _camel_parts(m.group(0)):
            seen.setdefault(part, None)
        if len(seen) >= cap:
            break
    return " ".join(seen)


def _chunk_text(text: str) -> list[tuple[int, int, str, str]]:
    lines = text.splitlines()
    out: list[tuple[int, int, str, str]] = []
    for i in range(0, len(lines), CHUNK_LINES):
        block = lines[i : i + CHUNK_LINES]
        content = "\n".join(block)
        if not content.strip():
            continue
        out.append((i + 1, i + len(block), content, _camel_terms(content)))
    return out


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


@dataclass
class _Parsed:
    path: str
    size: int
    mtime: float
    sha1: str
    language: str | None
    lines: int = 0
    generated: bool = False
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    chunks: list[tuple[int, int, str, str]] = field(default_factory=list)
    content_unchanged: bool = False
    missing: bool = False


@dataclass
class _Work:
    path: str
    old_sha1: str | None
    size: int
    mtime: float


# --------------------------------------------------------------------------- import resolution

_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".d.ts", ".vue", ".svelte", ".json")
_JS_LANGS = frozenset({"javascript", "typescript", "vue", "svelte"})
_JVM_LANGS = frozenset({"java", "kotlin", "scala", "groovy"})
_JVM_EXTS = (".java", ".kt", ".kts", ".scala", ".groovy")
_C_LANGS = frozenset({"c", "cpp", "objective-c"})


def _py_module_names(path: str, init_dirs: set[str]) -> set[str]:
    stem = path.rsplit(".", 1)[0]
    parts = stem.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    names: set[str] = set()
    if not parts:
        return names
    names.add(".".join(parts))
    for i, comp in enumerate(parts[:-1]):
        if comp == "src":
            names.add(".".join(parts[i + 1 :]))
    directory = posixpath.dirname(path)
    top: str | None = None
    while directory and directory in init_dirs:
        top = directory
        directory = posixpath.dirname(directory)
    if top is not None:
        base = posixpath.dirname(top)
        rel = parts[len(base.split("/")) :] if base else parts
        if rel:
            names.add(".".join(rel))
    return names


class _Resolver:
    """Maps import specifiers to workspace files."""

    def __init__(self, root: Path, paths: Iterable[str]) -> None:
        self.paths: set[str] = set(paths)
        ordered = sorted(self.paths)
        init_dirs = {posixpath.dirname(p) for p in ordered if posixpath.basename(p) == "__init__.py"}
        self.py_modules: dict[str, list[str]] = {}
        self.go_dirs: dict[str, list[str]] = {}
        self.jvm_classes: dict[str, list[str]] = {}
        self.jvm_packages: dict[str, list[str]] = {}
        self.by_basename: dict[str, list[str]] = {}
        go_mods: list[tuple[str, str]] = []
        crates: list[str] = []
        for p in ordered:
            name = posixpath.basename(p)
            directory = posixpath.dirname(p)
            self.by_basename.setdefault(name, []).append(p)
            if p.endswith((".py", ".pyi")):
                for module in _py_module_names(p, init_dirs):
                    self.py_modules.setdefault(module, []).append(p)
            elif p.endswith(".go"):
                if not p.endswith("_test.go"):
                    self.go_dirs.setdefault(directory, []).append(p)
            elif p.endswith(_JVM_EXTS):
                parts = p.rsplit(".", 1)[0].split("/")
                for k in range(len(parts) - 1):
                    self.jvm_classes.setdefault(".".join(parts[k:]), []).append(p)
                dir_parts = parts[:-1]
                for k in range(len(dir_parts) - 1):
                    self.jvm_packages.setdefault(".".join(dir_parts[k:]), []).append(p)
            elif name == "go.mod":
                text = _read_small(root / p)
                m = re.search(r"^module\s+(\S+)", text or "", re.M)
                if m:
                    go_mods.append((m.group(1).strip("\"'"), directory))
            elif name == "Cargo.toml":
                crates.append(directory)
        self.go_mods = sorted(go_mods, key=lambda item: -len(item[0]))
        self.rust_modules: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        self.rust_file_module: dict[str, tuple[str, tuple[str, ...]]] = {}
        crates.sort(key=len, reverse=True)
        for p in ordered:
            if not p.endswith(".rs"):
                continue
            for crate in crates:
                src = f"{crate}/src/" if crate else "src/"
                if not p.startswith(src):
                    continue
                parts = p[len(src) : -3].split("/")
                if parts[0] == "bin" or parts in (["lib"], ["main"]):
                    self.rust_file_module[p] = (crate, ())
                    if parts[0] != "bin":
                        self.rust_modules.setdefault((crate, ()), []).append(p)
                    break
                if parts[-1] == "mod":
                    parts = parts[:-1]
                key = (crate, tuple(parts))
                self.rust_modules.setdefault(key, []).append(p)
                self.rust_file_module[p] = key
                break

    def resolve(self, importer: str, language: str | None, spec: str) -> list[str]:
        if language == "python":
            return self._python(importer, spec)
        if language in _JS_LANGS:
            return self._js(importer, spec)
        if language == "go":
            return self._go(spec)
        if language == "rust":
            return self._rust(importer, spec)
        if language in _JVM_LANGS:
            return self._jvm(spec)
        if language in _C_LANGS:
            return self._c(importer, spec)
        return []

    def _python(self, importer: str, spec: str) -> list[str]:
        parts = [p for p in spec.split(".") if p]
        for k in range(len(parts), 0, -1):
            hit = self.py_modules.get(".".join(parts[:k]))
            if hit:
                return hit
        directory = posixpath.dirname(importer)
        if directory:  # script-style sibling imports (directory on sys.path)
            for k in range(len(parts), 0, -1):
                base = posixpath.join(directory, *parts[:k])
                for candidate in (base + ".py", base + "/__init__.py"):
                    if candidate in self.paths:
                        return [candidate]
        return []

    def _js(self, importer: str, spec: str) -> list[str]:
        if not spec.startswith("."):
            return []
        spec = spec.split("?", 1)[0].split("#", 1)[0]
        base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
        if base == "..":
            return []
        if base.startswith("../"):
            return []
        if base == ".":
            base = ""
        if base in self.paths:
            return [base]
        for ext in _JS_EXTS:
            if base + ext in self.paths:
                return [base + ext]
        stem, ext = posixpath.splitext(base)
        if ext in (".js", ".jsx", ".mjs", ".cjs"):
            for alt in (".ts", ".tsx", ".mts", ".cts"):
                if stem + alt in self.paths:
                    return [stem + alt]
        for ext in _JS_EXTS:
            candidate = posixpath.join(base, "index" + ext) if base else "index" + ext
            if candidate in self.paths:
                return [candidate]
        return []

    def _go(self, spec: str) -> list[str]:
        for module, directory in self.go_mods:
            if spec == module or spec.startswith(module + "/"):
                sub = spec[len(module) :].strip("/")
                target = posixpath.join(directory, sub) if directory and sub else (directory or sub)
                return self.go_dirs.get(target, [])
        return []

    def _rust(self, importer: str, spec: str) -> list[str]:
        location = self.rust_file_module.get(importer)
        if location is None:
            return []
        crate, module = location
        segs = [s for s in spec.split("::") if s]
        if not segs:
            return []
        min_k = 1
        if segs[0] == "crate":
            target = segs[1:]
            min_k = 0
        elif segs[0] == "self":
            target = list(module) + segs[1:]
        elif segs[0] == "super":
            ups = 0
            while ups < len(segs) and segs[ups] == "super":
                ups += 1
            if ups > len(module):
                return []
            target = list(module[: len(module) - ups]) + segs[ups:]
        elif (crate, (segs[0],)) in self.rust_modules:
            target = segs
        else:
            return []
        for k in range(len(target), min_k - 1, -1):
            hit = self.rust_modules.get((crate, tuple(target[:k])))
            if hit:
                return hit
        return []

    def _jvm(self, spec: str) -> list[str]:
        if spec.endswith(".*"):
            return self.jvm_packages.get(spec[:-2], [])
        segs = spec.split(".")
        for k in range(len(segs), 1, -1):
            hit = self.jvm_classes.get(".".join(segs[:k]))
            if hit:
                return hit
        for k in range(len(segs) - 1, 1, -1):
            hit = self.jvm_packages.get(".".join(segs[:k]))
            if hit:
                return hit
        return []

    def _c(self, importer: str, spec: str) -> list[str]:
        local = posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
        if local in self.paths:
            return [local]
        rooted = posixpath.normpath(spec)
        if rooted in self.paths:
            return [rooted]
        matches = [p for p in self.by_basename.get(posixpath.basename(spec), []) if p.endswith("/" + rooted)]
        return matches[:3]


def _read_small(path: Path, limit: int = 256 * 1024) -> str | None:
    try:
        if path.stat().st_size > limit:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# --------------------------------------------------------------------------- test naming

_NEUTRAL_DIRS = frozenset({"src", "lib", "app", "tests", "test", "__tests__", "spec", "specs", "unit", "integration"})
_JS_SOURCE_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".vue", ".svelte")


def _test_name_candidates(path: str) -> set[str]:
    name = posixpath.basename(path)
    stem, ext = posixpath.splitext(name)
    ext = ext.lower()
    if stem in ("__init__", "index", "mod", "main") and posixpath.dirname(path):
        stem = posixpath.basename(posixpath.dirname(path))
    s = stem.lower()
    names: set[str] = set()
    if ext in (".py", ".pyi"):
        names |= {f"test_{s}.py", f"{s}_test.py", f"{s}_tests.py", f"tests_{s}.py"}
    elif ext in _JS_SOURCE_EXTS:
        for e in _JS_SOURCE_EXTS:
            names |= {f"{s}.test{e}", f"{s}.spec{e}"}
    elif ext == ".go":
        names.add(f"{s}_test.go")
    elif ext in _JVM_EXTS or ext in (".cs", ".php", ".swift"):
        for e in {ext, ".java", ".kt"} if ext in _JVM_EXTS else {ext}:
            names |= {f"{s}test{e}", f"{s}tests{e}", f"test{s}{e}", f"{s}it{e}", f"{s}spec{e}"}
    elif ext == ".rb":
        names |= {f"{s}_spec.rb", f"{s}_test.rb", f"test_{s}.rb"}
    elif ext == ".rs":
        names |= {f"{s}_test.rs", f"test_{s}.rs", f"{s}.rs"}
    elif ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"):
        for e in (".c", ".cc", ".cpp", ".cxx"):
            names |= {f"test_{s}{e}", f"{s}_test{e}", f"{s}_unittest{e}"}
    elif ext in (".ex", ".exs"):
        names.add(f"{s}_test.exs")
    elif ext == ".dart":
        names.add(f"{s}_test.dart")
    return names


def _dir_affinity(source: str, test: str) -> int:
    src_dir = posixpath.dirname(source)
    test_dir = posixpath.dirname(test)
    a = [c for c in src_dir.split("/") if c and c not in _NEUTRAL_DIRS]
    b = [c for c in test_dir.split("/") if c and c not in _NEUTRAL_DIRS]
    score = len(set(a) & set(b))
    if src_dir == test_dir:
        score += 5
    if b and (b == a or b == a[1:]):
        score += 2
    return score


# --------------------------------------------------------------------------- query helpers

_QUERY_STOP = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "on", "with", "how", "what", "where",
        "does", "do", "which", "that", "this", "it", "be", "by", "from", "as", "at", "are", "was", "we",
        "not", "no", "can", "when", "why", "who",
    }
)


def _query_terms(query: str) -> list[str]:
    raw: list[str] = []
    for word in re.findall(r"\w+", query):
        raw.append(word.lower())
        if re.search(r"[a-z0-9][A-Z]", word):
            raw.extend(_camel_parts(word))
        if "_" in word.strip("_"):
            raw.extend(p.lower() for p in word.split("_") if len(p) >= 2)
    filtered = [t for t in raw if len(t) >= 2 and t not in _QUERY_STOP]
    if not filtered:
        filtered = [t for t in raw if t]
    return list(dict.fromkeys(filtered))[:_MAX_QUERY_TERMS]


def _fts_query(terms: list[str]) -> str:
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def _snippet(content: str, terms: list[str]) -> str:
    lines = content.split("\n")
    best = 0
    for i, line in enumerate(lines):
        lowered = line.lower()
        if any(t in lowered for t in terms):
            best = i
            break
    out: list[str] = []
    total = 0
    for line in lines[max(0, best - 1) :]:
        if total + len(line) + 1 > _SNIPPET_CHARS:
            if not out:
                out.append(line[:_SNIPPET_CHARS])
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)[:_SNIPPET_CHARS]


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# --------------------------------------------------------------------------- the index


class RepoIndex:
    """Incremental repository index stored in SQLite (``.agent/indexes/index.db``)."""

    def __init__(self, root: Path, db_path: Path, max_parse_bytes: int = MAX_PARSE_BYTES) -> None:
        self.root = Path(root)
        self.db_path = Path(db_path)
        self.max_parse_bytes = max_parse_bytes
        self._lock = threading.RLock()
        self._workers = max(1, min(8, os.cpu_count() or 1))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts = False
        self._conn = self._open()

    # ----------------------------------------------------------------- lifecycle

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        with contextlib.suppress(sqlite3.DatabaseError):
            conn.execute("PRAGMA journal_mode=WAL")
        with contextlib.suppress(sqlite3.DatabaseError):
            conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _delete_db_files(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            with contextlib.suppress(OSError):
                Path(str(self.db_path) + suffix).unlink()

    def _open(self) -> sqlite3.Connection:
        want_fts = fts5_available()
        conn = self._connect()
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            stored = None
            if "meta" in tables:
                row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
                stored = row[0] if row else None
        except sqlite3.DatabaseError:
            tables, stored = {"<corrupt>"}, None
        expected = f"{SCHEMA_VERSION}:{'fts5' if want_fts else 'plain'}"
        if tables and stored != expected:
            conn.close()
            self._delete_db_files()
            conn = self._connect()
        conn.executescript(_SCHEMA)
        fts = False
        if want_fts:
            try:
                conn.executescript(_FTS_SCHEMA)
                fts = True
            except sqlite3.OperationalError:
                fts = False
        self._fts = fts
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema', ?)",
                (f"{SCHEMA_VERSION}:{'fts5' if fts else 'plain'}",),
            )
        return conn

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()

    def __enter__(self) -> RepoIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def fts_enabled(self) -> bool:
        """True when BM25 search uses SQLite FTS5."""
        return self._fts

    # ----------------------------------------------------------------- refresh

    def _normalize(self, path: str) -> str | None:
        p = Path(path)
        if p.is_absolute():
            try:
                p = p.resolve().relative_to(self.root.resolve())
            except (ValueError, OSError):
                return None
        rel = posixpath.normpath(p.as_posix().replace("\\", "/"))
        if rel in (".", "") or rel.startswith("../") or rel == "..":
            return None
        parts = rel.split("/")
        if ".git" in parts[:-1] or ".agent" in parts[:-1]:
            return None
        return rel

    def _process(self, work: _Work) -> _Parsed:
        path = work.path
        language = language_of(path)
        path_generated = is_generated_path(path)
        if work.size > self.max_parse_bytes:
            return _Parsed(path, work.size, work.mtime, "", language, generated=path_generated)
        try:
            with (self.root / path).open("rb") as fh:
                data = fh.read(self.max_parse_bytes + 1)  # never read past the cap
        except FileNotFoundError:
            return _Parsed(path, work.size, work.mtime, "", language, missing=True)
        except OSError:
            return _Parsed(path, work.size, work.mtime, "", language, generated=path_generated)
        if len(data) > self.max_parse_bytes:
            return _Parsed(path, work.size, work.mtime, "", language, generated=path_generated)
        digest = sha1_bytes(data)
        if work.old_sha1 is not None and work.old_sha1 == digest:
            return _Parsed(path, len(data), work.mtime, digest, language, content_unchanged=True)
        if is_binary_bytes(data):
            return _Parsed(path, len(data), work.mtime, digest, language, generated=path_generated)
        text = data.decode("utf-8", errors="replace")
        generated = path_generated or is_generated_content(text[:2048])
        minified = ".min." in posixpath.basename(path) or path.endswith(".map")
        symbols: list[Symbol] = []
        imports: list[str] = []
        try:
            if not minified:
                symbols = extract_symbols(path, text, language)
            imports = extract_imports(path, text, language)
        except Exception:  # a parser bug must not abort indexing of the whole repository
            symbols, imports = [], []
        chunks = [] if generated or _is_sensitive(path) else _chunk_text(text)
        return _Parsed(
            path=path,
            size=len(data),
            mtime=work.mtime,
            sha1=digest,
            language=language,
            lines=_count_lines(text),
            generated=generated,
            symbols=symbols,
            imports=imports,
            chunks=chunks,
        )

    def _delete_path(self, path: str) -> None:
        c = self._conn
        c.execute("DELETE FROM symbols WHERE path = ?", (path,))
        c.execute("DELETE FROM imports WHERE path = ?", (path,))
        c.execute("DELETE FROM chunks WHERE path = ?", (path,))
        c.execute("DELETE FROM files WHERE path = ?", (path,))

    def _write(self, res: _Parsed) -> None:
        c = self._conn
        self._delete_path(res.path)
        c.execute(
            "INSERT INTO files(path, language, size, mtime, sha1, lines, generated) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (res.path, res.language, res.size, res.mtime, res.sha1, res.lines, int(res.generated)),
        )
        c.executemany(
            "INSERT INTO symbols(path, name, name_lower, kind, line, end_line, parent, signature) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (res.path, s.name, s.name.lower(), s.kind, s.line, s.end_line, s.parent, s.signature)
                for s in res.symbols
            ],
        )
        c.executemany(
            "INSERT INTO imports(path, seq, spec) VALUES (?, ?, ?)",
            [(res.path, i, spec) for i, spec in enumerate(res.imports)],
        )
        c.executemany(
            "INSERT INTO chunks(path, start_line, end_line, content, terms) VALUES (?, ?, ?, ?, ?)",
            [(res.path, start, end, content, terms) for start, end, content, terms in res.chunks],
        )

    def refresh(self, files: list[str] | None = None) -> IndexStats:
        """Bring the index up to date.

        With ``files=None`` the whole workspace is listed: new files are added,
        changed ones re-parsed and vanished ones removed. With an explicit list,
        only those paths are refreshed (missing ones are removed); other indexed
        files are left untouched.
        """
        started = time.perf_counter()
        stats = IndexStats()
        with self._lock:
            existing: dict[str, tuple[int, float, str]] = {
                row[0]: (row[1], row[2], row[3])
                for row in self._conn.execute("SELECT path, size, mtime, sha1 FROM files")
            }
            if files is None:
                targets = list_files(self.root)
                to_remove = sorted(set(existing) - set(targets))
            else:
                targets = sorted({n for n in (self._normalize(f) for f in files) if n is not None})
                to_remove = []
            work: list[_Work] = []
            for path in targets:
                try:
                    st = os.stat(self.root / path)
                except OSError:
                    st = None
                if st is None or not stat.S_ISREG(st.st_mode):
                    if path in existing and path not in to_remove:
                        to_remove.append(path)
                    continue
                old = existing.get(path)
                if old is not None and old[0] == st.st_size and old[1] == st.st_mtime:
                    stats.unchanged += 1
                    continue
                work.append(_Work(path, old[2] if old and old[2] else None, st.st_size, st.st_mtime))

            updated_paths: set[str] = set()
            if work:
                with ThreadPoolExecutor(max_workers=self._workers) as pool:
                    for i in range(0, len(work), _BATCH):
                        batch = work[i : i + _BATCH]
                        results = list(pool.map(self._process, batch))
                        with self._conn:
                            for res in results:
                                if res.missing:
                                    if res.path in existing:
                                        to_remove.append(res.path)
                                    continue
                                if res.content_unchanged:
                                    self._conn.execute(
                                        "UPDATE files SET size = ?, mtime = ? WHERE path = ?",
                                        (res.size, res.mtime, res.path),
                                    )
                                    stats.unchanged += 1
                                    continue
                                self._write(res)
                                if res.path in existing:
                                    stats.updated += 1
                                    updated_paths.add(res.path)
                                else:
                                    stats.added += 1
            if to_remove:
                with self._conn:
                    for path in sorted(set(to_remove)):
                        self._delete_path(path)
                        stats.removed += 1
            edges_missing = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'edges_built'"
            ).fetchone() is None
            # Module resolution depends on the set of files (and go.mod/Cargo.toml contents),
            # so only a pure content update can re-resolve just the changed importers.
            layout_changed = bool(stats.added or stats.removed) or any(
                posixpath.basename(p) in ("go.mod", "Cargo.toml") for p in updated_paths
            )
            if edges_missing or layout_changed:
                self._rebuild_edges()
            elif updated_paths:
                self._rebuild_edges(only=updated_paths)
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_refresh', ?)", (utcnow_iso(),)
                )
        stats.duration_s = round(time.perf_counter() - started, 4)
        return stats

    def _rebuild_edges(self, only: set[str] | None = None) -> None:
        """Resolve stored imports into file edges (all files, or just ``only``)."""
        rows = self._conn.execute("SELECT path, language FROM files").fetchall()
        language = {path: lang for path, lang in rows}
        resolver = _Resolver(self.root, language)
        edges: set[tuple[str, str]] = set()

        def resolve(path: str, spec: str) -> None:
            for dst in resolver.resolve(path, language.get(path), spec):
                if dst != path:
                    edges.add((path, dst))

        if only is None:
            for path, spec in self._conn.execute("SELECT path, spec FROM imports"):
                resolve(path, spec)
        else:
            for src in sorted(only):
                for (spec,) in self._conn.execute("SELECT spec FROM imports WHERE path = ?", (src,)).fetchall():
                    resolve(src, spec)
        with self._conn:
            if only is None:
                self._conn.execute("DELETE FROM edges")
            else:
                self._conn.executemany("DELETE FROM edges WHERE src = ?", [(p,) for p in sorted(only)])
            self._conn.executemany("INSERT OR IGNORE INTO edges(src, dst) VALUES (?, ?)", sorted(edges))
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('edges_built', ?)", (utcnow_iso(),))

    # ----------------------------------------------------------------- queries

    @staticmethod
    def _record(row: tuple[Any, ...]) -> FileRecord:
        return FileRecord(
            path=row[0],
            language=row[1],
            size=row[2],
            mtime=row[3],
            sha1=row[4],
            lines=row[5],
            generated=bool(row[6]),
        )

    def file(self, path: str) -> FileRecord | None:
        """Indexed metadata for ``path`` (workspace-relative), if present."""
        rel = self._normalize(path) or path
        with self._lock:
            row = self._conn.execute(
                "SELECT path, language, size, mtime, sha1, lines, generated FROM files WHERE path = ?", (rel,)
            ).fetchone()
        return self._record(row) if row else None

    def files(self) -> list[FileRecord]:
        """All indexed files, sorted by path."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, language, size, mtime, sha1, lines, generated FROM files ORDER BY path"
            ).fetchall()
        return [self._record(r) for r in rows]

    def find_symbol(self, name: str, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Find symbols by name: exact, then prefix, then substring (case-insensitive).

        ``Parent.name`` restricts matches to members of ``Parent``.
        """
        query = name.strip()
        if not query:
            return []
        parent: str | None = None
        if "." in query.strip("."):
            parent, _, query = query.rpartition(".")
            parent = parent.rpartition(".")[2].lower() or None
        lowered = query.lower()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.path, s.name, s.kind, s.line, s.end_line, s.parent, s.signature,
                       CASE WHEN s.name = ? THEN 0
                            WHEN s.name_lower = ? THEN 1
                            WHEN s.name_lower LIKE ? ESCAPE '\\' THEN 2
                            ELSE 3 END AS rank,
                       COALESCE(f.generated, 0) AS gen
                FROM symbols s LEFT JOIN files f ON f.path = s.path
                WHERE s.name_lower LIKE ? ESCAPE '\\'
                  AND (? IS NULL OR s.kind = ?)
                  AND (? IS NULL OR lower(s.parent) = ?)
                ORDER BY rank, gen, length(s.name), s.path, s.line
                LIMIT ?
                """,
                (
                    query,
                    lowered,
                    _like_escape(lowered) + "%",
                    "%" + _like_escape(lowered) + "%",
                    kind,
                    kind,
                    parent,
                    parent,
                    max(0, limit),
                ),
            ).fetchall()
        return [
            {
                "path": r[0],
                "name": r[1],
                "kind": r[2],
                "line": r[3],
                "end_line": r[4],
                "parent": r[5],
                "signature": r[6],
            }
            for r in rows
        ]

    def file_symbols(self, path: str) -> list[Symbol]:
        """Symbols defined in ``path`` ordered by line."""
        rel = self._normalize(path) or path
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, kind, line, end_line, parent, signature FROM symbols WHERE path = ? ORDER BY line, id",
                (rel,),
            ).fetchall()
        return [
            Symbol(name=r[0], kind=r[1], line=r[2], end_line=r[3], parent=r[4], signature=r[5]) for r in rows
        ]

    def imports_of(self, path: str) -> list[str]:
        """Workspace files that ``path`` imports (resolved edges)."""
        rel = self._normalize(path) or path
        with self._lock:
            rows = self._conn.execute("SELECT dst FROM edges WHERE src = ? ORDER BY dst", (rel,)).fetchall()
        return [r[0] for r in rows]

    def _importers(self, path: str) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT src FROM edges WHERE dst = ? ORDER BY src", (path,))]

    def dependents(self, path: str, transitive: bool = True, limit: int = 500) -> list[str]:
        """Files importing ``path``; with ``transitive`` a BFS (nearest first)."""
        rel = self._normalize(path) or path
        out: list[str] = []
        seen = {rel}
        frontier = [rel]
        with self._lock:
            while frontier and len(out) < limit:
                nxt: list[str] = []
                for node in frontier:
                    for src in self._importers(node):
                        if src in seen:
                            continue
                        seen.add(src)
                        out.append(src)
                        nxt.append(src)
                if not transitive:
                    break
                frontier = nxt
        return out[: max(0, limit)]

    def related_tests(self, path: str, limit: int = 50) -> list[str]:
        """Tests likely covering ``path``: importers (depth <= 3) then naming conventions."""
        rel = self._normalize(path) or path
        found: list[str] = []
        if is_test_path(rel):
            found.append(rel)
        seen = {rel}
        frontier = [rel]
        with self._lock:
            for _ in range(3):
                nxt: list[str] = []
                for node in frontier:
                    for src in self._importers(node):
                        if src in seen:
                            continue
                        seen.add(src)
                        nxt.append(src)
                        if is_test_path(src):
                            found.append(src)
                frontier = nxt
                if not frontier:
                    break
            all_paths = [r[0] for r in self._conn.execute("SELECT path FROM files ORDER BY path")]
        names = _test_name_candidates(rel)
        is_go = rel.endswith(".go")
        scored: list[tuple[int, str]] = []
        if names:
            already = set(found)
            for candidate in all_paths:
                if candidate in already or candidate == rel:
                    continue
                if posixpath.basename(candidate).lower() not in names or not is_test_path(candidate):
                    continue
                if is_go and posixpath.dirname(candidate) != posixpath.dirname(rel):
                    continue
                scored.append((-_dir_affinity(rel, candidate), candidate))
        scored.sort()
        found.extend(p for _, p in scored)
        return list(dict.fromkeys(found))[: max(0, limit)]

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """BM25 search over ~60-line chunks of text files.

        Returns dicts with ``path``, ``start_line``, ``end_line``, ``score`` (higher
        is better) and ``snippet`` (<= 400 chars).
        """
        terms = _query_terms(query)
        if not terms or limit <= 0:
            return []
        with self._lock:
            if self._fts:
                try:
                    rows = self._conn.execute(
                        """
                        SELECT c.path, c.start_line, c.end_line, c.content,
                               bm25(chunks_fts, 2.0, 1.0, 0.5) AS rank
                        FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank, c.path, c.start_line
                        LIMIT ?
                        """,
                        (_fts_query(terms), limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = None
                if rows is not None:
                    return [
                        {
                            "path": r[0],
                            "start_line": r[1],
                            "end_line": r[2],
                            "score": round(-float(r[4]), 4),
                            "snippet": _snippet(r[3], terms),
                        }
                        for r in rows
                    ]
            return self._search_fallback(terms, limit)

    def _search_fallback(self, terms: list[str], limit: int) -> list[dict[str, Any]]:
        """LIKE-based candidate selection with Python-side BM25 scoring."""
        c = self._conn
        total = c.execute("SELECT count(*), avg(length(content)) FROM chunks").fetchone()
        n_chunks = int(total[0] or 0)
        avg_len = float(total[1] or 1.0)
        if n_chunks == 0:
            return []
        doc_terms: dict[int, int] = {}
        df: dict[str, int] = {}
        for term in terms:
            pattern = "%" + _like_escape(term) + "%"
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM chunks WHERE content LIKE ? ESCAPE '\\' OR path LIKE ? ESCAPE '\\'",
                    (pattern, pattern),
                )
            ]
            df[term] = len(ids)
            for chunk_id in ids:
                doc_terms[chunk_id] = doc_terms.get(chunk_id, 0) + 1
        candidates = sorted(doc_terms, key=lambda i: (-doc_terms[i], i))[:2000]
        k1, b = 1.2, 0.75
        scored: list[tuple[float, str, int, int, str]] = []
        for chunk_id in candidates:
            row = c.execute(
                "SELECT path, start_line, end_line, content FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            if row is None:
                continue
            path, start, end, content = row
            lowered = content.lower()
            lowered_path = path.lower()
            score = 0.0
            for term in terms:
                tf = lowered.count(term) + 2 * lowered_path.count(term)
                if not tf:
                    continue
                idf = math.log(1 + (n_chunks - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(content) / avg_len))
            scored.append((score, path, start, end, content))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "score": round(score, 4),
                "snippet": _snippet(content, terms),
            }
            for score, path, start, end, content in scored[:limit]
        ]

    def stats(self) -> dict[str, Any]:
        """Counts of files, symbols, edges and chunks plus files per language."""
        with self._lock:
            c = self._conn
            languages = {
                (lang or "unknown"): count
                for lang, count in c.execute(
                    "SELECT language, count(*) FROM files GROUP BY language ORDER BY language"
                )
            }
            last = c.execute("SELECT value FROM meta WHERE key = 'last_refresh'").fetchone()
            return {
                "files": c.execute("SELECT count(*) FROM files").fetchone()[0],
                "symbols": c.execute("SELECT count(*) FROM symbols").fetchone()[0],
                "edges": c.execute("SELECT count(*) FROM edges").fetchone()[0],
                "chunks": c.execute("SELECT count(*) FROM chunks").fetchone()[0],
                "languages": languages,
                "fts": self._fts,
                "last_refresh": last[0] if last else None,
            }
