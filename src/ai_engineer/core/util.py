"""Small, dependency-free helpers."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utcnow() -> datetime:
    return datetime.now(UTC)


def utcnow_iso() -> str:
    return utcnow().isoformat(timespec="seconds")


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()


def sha1_text(text: str) -> str:
    return sha1_bytes(text.encode("utf-8", errors="surrogateescape"))


def file_sha1(path: Path) -> str | None:
    try:
        h = hashlib.sha1(usedforsecurity=False)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8", newline: str | None = None) -> None:
    """Write ``text`` to ``path`` atomically (temp file + rename), preserving mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = None
    with contextlib.suppress(OSError):
        mode = path.stat().st_mode & 0o7777
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def truncate_middle(text: str, max_chars: int, marker: str = "\n... [{omitted} characters omitted] ...\n") -> str:
    """Keep the head and tail of ``text``; errors usually live at the end."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.4)
    tail = max_chars - head
    omitted = len(text) - head - tail
    return text[:head] + marker.format(omitted=omitted) + text[-tail:]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token for code and English)."""
    return max(1, (len(text) + 3) // 4)


def is_binary_bytes(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data[:8192]:
        return True
    sample = data[:8192]
    text_chars = bytes(range(32, 127)) + b"\n\r\t\f\b"
    nontext = sum(1 for b in sample if b not in text_chars and b < 128)
    return nontext / max(1, len(sample)) > 0.30


def read_text_file(path: Path, max_bytes: int | None = None) -> tuple[str, str]:
    """Read a text file; returns (text, newline_style). Raises ValueError for binary files."""
    data = path.read_bytes() if max_bytes is None else path.open("rb").read(max_bytes)
    if is_binary_bytes(data):
        raise ValueError(f"{path} appears to be a binary file")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    newline = "\r\n" if "\r\n" in text else "\n"
    return text, newline


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def keywords(text: str, limit: int = 20) -> list[str]:
    """Extract distinctive identifiers/words from free text (deterministic)."""
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "when", "then", "should", "would",
        "could", "make", "please", "need", "want", "add", "fix", "use", "using", "are", "was", "will",
        "can", "not", "all", "any", "has", "have", "its", "but", "you", "your", "our", "new", "file",
        "files", "code", "implement", "create", "update", "change", "also", "only", "there", "which",
    }
    seen: dict[str, None] = {}
    for word in _WORD_RE.findall(text):
        lw = word.lower()
        if lw in stop or lw in seen:
            continue
        seen[lw] = None
        if len(seen) >= limit:
            break
    return list(seen)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from model text.

    Accepts fenced code blocks or bare JSON embedded in prose. Raises ValueError.
    """
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidates: list[str] = []
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        found = _scan_balanced(candidate)
        if found is not None:
            return found
    raise ValueError("no JSON object found in model output")


def _scan_balanced(text: str) -> Any:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
