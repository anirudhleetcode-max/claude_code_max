"""Tracks files the agent read and changed.

- Stale-write protection: an existing file may only be overwritten if the agent
  has read its current version.
- Change tracking: which files were created/modified (for review, reports and
  non-git checkpoints). ``on_first_write`` lets the checkpoint manager back up
  originals before the first modification.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import ToolError
from ..core.util import file_sha1, sha1_bytes


@dataclass
class ChangeRecord:
    path: str  # workspace-relative
    created: bool
    original_sha1: str | None
    current_sha1: str | None
    deleted: bool = False


FirstWriteHook = Callable[[str, bytes | None], None]


class FileStateTracker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._read_hashes: dict[str, str] = {}
        self._changes: dict[str, ChangeRecord] = {}
        self._lock = threading.Lock()
        self.on_first_write: FirstWriteHook | None = None

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def record_read(self, path: Path, content: bytes | None = None) -> None:
        digest = sha1_bytes(content) if content is not None else file_sha1(path)
        if digest:
            with self._lock:
                self._read_hashes[self._rel(path)] = digest

    def has_read(self, path: Path) -> bool:
        return self._rel(path) in self._read_hashes

    def check_write(self, path: Path, require_read: bool = True) -> None:
        """Raise if overwriting ``path`` would clobber content the agent has not seen."""
        if not path.exists():
            return
        rel = self._rel(path)
        known = self._read_hashes.get(rel)
        current = file_sha1(path)
        if known is None:
            if require_read:
                raise ToolError(f"{rel} exists and has not been read in this session; read it before overwriting")
            return
        if current != known:
            raise ToolError(
                f"{rel} changed on disk since it was last read (possibly edited by someone else); re-read it first"
            )

    def before_write(self, path: Path) -> None:
        """Record the original state the first time a file is modified."""
        rel = self._rel(path)
        with self._lock:
            if rel in self._changes:
                return
            exists = path.exists()
            original = path.read_bytes() if exists and path.is_file() else None
            self._changes[rel] = ChangeRecord(
                path=rel,
                created=not exists,
                original_sha1=sha1_bytes(original) if original is not None else None,
                current_sha1=None,
            )
        if self.on_first_write is not None:
            self.on_first_write(rel, original)

    def after_write(self, path: Path, deleted: bool = False) -> None:
        rel = self._rel(path)
        digest = None if deleted else file_sha1(path)
        with self._lock:
            record = self._changes.get(rel)
            if record is not None:
                record.current_sha1 = digest
                record.deleted = deleted
            if digest:
                self._read_hashes[rel] = digest  # the agent knows what it just wrote
            else:
                self._read_hashes.pop(rel, None)

    def changes(self) -> list[ChangeRecord]:
        with self._lock:
            return [c for c in self._changes.values() if c.deleted or c.created or c.current_sha1 != c.original_sha1]

    def changed_paths(self) -> list[str]:
        return sorted(c.path for c in self.changes())

    def reset_changes(self) -> None:
        with self._lock:
            self._changes.clear()

    def forget_reads(self) -> None:
        with self._lock:
            self._read_hashes.clear()
