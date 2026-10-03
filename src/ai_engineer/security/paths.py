"""Workspace path jail and protected-path rules."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from ..core.errors import PathViolation


@lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_match(rel_path: str, pattern: str, *, ignore_case: bool = False) -> bool:
    """gitignore-like matching: patterns without '/' match the basename at any depth."""
    rel = rel_path.replace("\\", "/")
    if ignore_case:
        rel, pattern = rel.casefold(), pattern.casefold()
    while rel.startswith("./"):
        rel = rel[2:]
    rel = rel.lstrip("/")
    if rel == ".":
        rel = ""
    if "/" not in pattern.rstrip("/"):
        name = rel.rsplit("/", 1)[-1]
        return bool(_glob_regex(pattern).match(name))
    return bool(_glob_regex(pattern.lstrip("/")).match(rel))


def _guarded_match(rel_path: str, pattern: str) -> bool:
    # Case-insensitive on every platform: on macOS/Windows ".GIT/hooks/x" or ".ENV" name the same file
    # as the protected one, and treating them alike everywhere keeps the policy platform-independent.
    return glob_match(rel_path, pattern, ignore_case=True)


class PathGuard:
    """Resolves user/model supplied paths and enforces the workspace boundary."""

    def __init__(
        self,
        root: Path,
        protected: list[str] | None = None,
        secret_files: list[str] | None = None,
        extra_read_roots: list[Path] | None = None,
    ) -> None:
        self.root = Path(os.path.realpath(root))
        self.protected = list(protected or [])
        self.secret_files = list(secret_files or [])
        self.extra_read_roots = [Path(os.path.realpath(p)) for p in (extra_read_roots or [])]

    def _within(self, path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    def resolve(self, path: str | os.PathLike[str], *, for_write: bool = False) -> Path:
        raw = os.fspath(path)
        if not raw or "\x00" in raw:
            raise PathViolation("empty or invalid path")
        if raw.startswith("~"):
            raise PathViolation(f"home-relative paths are not allowed: {raw}")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # realpath resolves symlinks of existing components, preventing symlink escapes
        resolved = Path(os.path.realpath(candidate))
        allowed = [self.root] if for_write else [self.root, *self.extra_read_roots]
        if not any(self._within(resolved, base) for base in allowed):
            raise PathViolation(f"path is outside the workspace: {raw}")
        if for_write:
            rel = self.relative(resolved)
            for pattern in self.protected:
                if _guarded_match(rel, pattern):
                    raise PathViolation(f"path is protected ({pattern}): {rel}")
        return resolved

    def relative(self, path: Path) -> str:
        try:
            rel = Path(os.path.realpath(path)).relative_to(self.root).as_posix()
        except ValueError:
            return Path(path).as_posix()
        return rel or "."

    def is_secret_file(self, path: Path | str) -> bool:
        rel = self.relative(Path(path)) if isinstance(path, Path) else path
        return any(_guarded_match(rel, p) for p in self.secret_files)

    def is_protected(self, rel: str) -> bool:
        return any(_guarded_match(rel, p) for p in self.protected)
