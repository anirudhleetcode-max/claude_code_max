"""Text search (ripgrep with a pure-Python fallback) and filename globbing."""

from __future__ import annotations

import base64
import binascii
import contextlib
import fnmatch
import functools
import json
import os
import posixpath
import re
import shutil
import subprocess
import threading
from pathlib import Path

from pydantic import BaseModel

from ai_engineer.core.util import is_binary_bytes
from ai_engineer.repo.files import DEFAULT_IGNORE_DIRS, list_files

_MAX_LINE_CHARS = 300
_RG_TIMEOUT_S = 30.0
_FALLBACK_MAX_BYTES = 2 * 1024 * 1024


class TextMatch(BaseModel):
    """One matching line."""

    path: str
    line: int
    text: str


def _trim(text: str) -> str:
    return text.rstrip("\r\n")[:_MAX_LINE_CHARS]


def _compile(pattern: str, regex: bool, case_sensitive: bool) -> re.Pattern[str]:
    if not pattern:
        raise ValueError("search pattern must not be empty")
    flags = 0 if case_sensitive else re.IGNORECASE
    source = pattern if regex else re.escape(pattern)
    try:
        return re.compile(source, flags)
    except re.error as exc:
        raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from exc


@functools.lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a gitignore-style glob (``*``, ``?``, ``[...]``, ``**``) to a regex."""
    i = 0
    n = len(pattern)
    out: list[str] = []
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if pattern.startswith("**", i):
                i += 2
                if i < n and pattern[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(ch))
            else:
                body = pattern[i + 1 : j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body.replace(chr(92), chr(92) * 2)}]")
                i = j
        else:
            out.append(re.escape(ch))
        i += 1
    try:
        return re.compile("".join(out) + r"\Z")
    except re.error:
        return re.compile(re.escape(pattern) + r"\Z")


def _has_magic(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def glob_match(path: str, pattern: str) -> bool:
    """Match a workspace-relative path against a glob.

    Patterns without ``/`` also match the basename (case-insensitively); ``**``
    spans directories.
    """
    pattern = pattern.replace("\\", "/").removeprefix("./").lstrip("/")
    if not pattern:
        return False
    if _glob_regex(pattern).match(path):
        return True
    if "/" not in pattern:
        name = posixpath.basename(path)
        return fnmatch.fnmatchcase(name.lower(), pattern.lower())
    return False


def find_files(root: Path, pattern: str, max_results: int = 500, files: list[str] | None = None) -> list[str]:
    """Find files by glob on the relative path or the basename.

    A pattern without wildcards matches basenames containing it (case-insensitive)
    or paths ending with it; exact basename matches are listed first.
    """
    pattern = pattern.strip().replace("\\", "/")
    if not pattern:
        return []
    candidates = files if files is not None else list_files(Path(root))
    if _has_magic(pattern):
        return [p for p in candidates if glob_match(p, pattern)][:max_results]
    needle = pattern.removeprefix("./").lstrip("/").lower()
    exact: list[str] = []
    partial: list[str] = []
    for path in candidates:
        lower = path.lower()
        name = posixpath.basename(lower)
        if name == needle or lower == needle or lower.endswith("/" + needle):
            exact.append(path)
        elif needle in name or ("/" in needle and needle in lower):
            partial.append(path)
    return (exact + partial)[:max_results]


def _rg_text(field: object) -> str | None:
    """Decode an rg JSON ``{"text": ...}`` / ``{"bytes": base64}`` value."""
    if not isinstance(field, dict):
        return None
    text = field.get("text")
    if isinstance(text, str):
        return text
    raw = field.get("bytes")
    if isinstance(raw, str):
        try:
            return base64.b64decode(raw).decode("utf-8", errors="replace")
        except (binascii.Error, ValueError):
            return None
    return None


def _rg_command(rg: str, pattern: str, regex: bool, case_sensitive: bool, glob: str | None) -> list[str]:
    cmd = [
        rg,
        "--json",
        "--max-columns",
        "400",
        "--max-filesize",
        "2M",
        "--sort",
        "path",
        "--hidden",
        "--path-separator",
        "/",
        "--glob",
        "!.git",
    ]
    for name in sorted(DEFAULT_IGNORE_DIRS):
        cmd += ["--glob", f"!{name}/"]
    if glob:
        cmd += ["--glob", glob]
    if not regex:
        cmd.append("--fixed-strings")
    cmd.append("--case-sensitive" if case_sensitive else "--ignore-case")
    cmd += ["--regexp", pattern, "--", "."]
    return cmd


def _search_rg(
    rg: str, root: Path, pattern: str, regex: bool, case_sensitive: bool, glob: str | None, max_results: int
) -> list[TextMatch] | None:
    """Run ripgrep; ``None`` means "fall back to Python"."""
    cmd = _rg_command(rg, pattern, regex, case_sensitive, glob)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # an unread stderr pipe could fill up and stall rg
        )
    except OSError:
        return None
    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        with contextlib.suppress(OSError):
            proc.kill()

    timer = threading.Timer(_RG_TIMEOUT_S, _kill)
    timer.daemon = True
    timer.start()
    results: list[TextMatch] = []
    truncated = False
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data", {})
            path = _rg_text(data.get("path"))
            text = _rg_text(data.get("lines"))
            line = data.get("line_number")
            if path is None or text is None or line is None:
                continue
            path = path.removeprefix("./")
            results.append(TextMatch(path=path, line=int(line), text=_trim(text)))
            if len(results) >= max_results:
                truncated = True
                break
    finally:
        timer.cancel()
        if truncated or timed_out.is_set():
            with contextlib.suppress(OSError):
                proc.kill()
        with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
            proc.communicate(timeout=5)
    if truncated or timed_out.is_set():
        return results  # partial results; a Python scan would only be slower
    if proc.returncode not in (0, 1) and not results:
        return None  # e.g. a regex rg cannot handle (look-around); let Python try
    return results


def _search_python(
    root: Path,
    compiled: re.Pattern[str],
    glob: str | None,
    max_results: int,
    files: list[str] | None,
) -> list[TextMatch]:
    candidates = files if files is not None else list_files(root)
    results: list[TextMatch] = []
    for rel in candidates:
        if glob and not glob_match(rel, glob):
            continue
        full = root / rel
        try:
            if not full.is_file() or full.stat().st_size > _FALLBACK_MAX_BYTES:
                continue
            with full.open("rb") as fh:
                data = fh.read(_FALLBACK_MAX_BYTES + 1)
        except OSError:
            continue
        if len(data) > _FALLBACK_MAX_BYTES:
            continue
        if is_binary_bytes(data):
            continue
        text = data.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                results.append(TextMatch(path=rel, line=lineno, text=_trim(line)))
                if len(results) >= max_results:
                    return results
    return results


def search_text(
    root: Path,
    pattern: str,
    *,
    regex: bool = True,
    case_sensitive: bool = False,
    glob: str | None = None,
    max_results: int = 200,
    files: list[str] | None = None,
) -> list[TextMatch]:
    """Search file contents line by line.

    Uses ``rg`` when installed (respects ``.gitignore``), otherwise a Python scan
    over :func:`list_files`. When ``files`` is given, only those paths are searched
    (always with the Python scanner). Raises ``ValueError`` for an invalid regex.
    """
    root = Path(root)
    compiled = _compile(pattern, regex, case_sensitive)
    if max_results <= 0:
        return []
    if files is None:
        rg = shutil.which("rg")
        if rg is not None and os.path.isdir(root):
            found = _search_rg(rg, root, pattern, regex, case_sensitive, glob, max_results)
            if found is not None:
                return found
    return _search_python(root, compiled, glob, max_results, files)
