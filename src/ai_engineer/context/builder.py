"""Selects what goes into a model request: never the whole repository.

Sources: the repository profile, project instruction files, memory hints (marked as
hints), and the most relevant files found via path hints, symbol lookup and BM25
search — included as numbered excerpts within a character budget.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.util import is_binary_bytes, keywords

INSTRUCTION_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    ".github/copilot-instructions.md",
    "CONTRIBUTING.md",
    ".cursorrules",
    "docs/CONTRIBUTING.md",
)

_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\b")
_PATHLIKE = re.compile(r"(?<![\w/])([\w.\-]+(?:/[\w.\-]+)+|[\w\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|cs|kt|md|toml|json|ya?ml|cfg|ini|sql|html|css))\b")


class ContextBuilder:
    def __init__(self, workspace: Path, profile: Any = None, index: Any = None, memory: Any = None, redactor: Any = None) -> None:
        self.workspace = workspace
        self.profile = profile
        self.index = index
        self.memory = memory
        self.redactor = redactor

    def _redact(self, text: str) -> str:
        return self.redactor.redact_text(text) if self.redactor is not None else text

    # ---- project brief ----------------------------------------------------------------

    def project_brief(self, max_chars: int = 6000) -> str:
        p = self.profile
        lines: list[str] = []
        if p is not None:
            lines.append(p.summary)
            if p.frameworks:
                lines.append("Frameworks/libraries: " + ", ".join(p.frameworks[:12]))
            if p.package_managers:
                lines.append("Package managers: " + ", ".join(p.package_managers))
            for kind in ("test", "lint", "typecheck", "build"):
                suggestions = p.commands.get(kind) or []
                if suggestions:
                    lines.append(f"{kind} command: {suggestions[0].command}")
            if p.entry_points:
                lines.append("Entry points: " + ", ".join(p.entry_points[:8]))
            if p.test_dirs:
                lines.append("Test directories: " + ", ".join(p.test_dirs[:6]))
        instructions = self.instructions(max_chars=max_chars // 2)
        if instructions:
            lines.append("")
            lines.append("Project instructions (from the repository; follow them):")
            lines.append(instructions)
        return self._redact("\n".join(lines))[:max_chars]

    def instructions(self, max_chars: int = 3000) -> str:
        parts: list[str] = []
        budget = max_chars
        for name in INSTRUCTION_FILES:
            path = self.workspace / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if not text:
                continue
            chunk = text[: max(0, budget - len(name) - 10)]
            if not chunk:
                break
            parts.append(f"--- {name} ---\n{chunk}")
            budget -= len(chunk) + len(name) + 10
            if budget <= 200:
                break
        return "\n".join(parts)

    # ---- relevance -------------------------------------------------------------------------

    def relevant_files(self, query: str, hints: list[str] | None = None, limit: int = 8) -> list[str]:
        chosen: dict[str, None] = {}

        def add(path: str) -> None:
            path = path.strip().lstrip("./")
            if path and (self.workspace / path).is_file() and len(chosen) < limit:
                chosen.setdefault(path, None)

        for hint in hints or []:
            add(hint)
        for match in _PATHLIKE.findall(query):
            add(match)
        if self.index is not None:
            for ident in dict.fromkeys(_IDENT.findall(query)):
                if len(chosen) >= limit:
                    break
                if not re.search(r"[A-Z_]", ident[1:]) and "." not in ident:
                    continue  # only CamelCase / snake_case / dotted identifiers look like symbols
                try:
                    for hit in self.index.find_symbol(ident.rsplit(".", 1)[-1], limit=3):
                        add(hit["path"])
                except Exception:  # noqa: S112 - index is best-effort
                    continue
            try:
                for hit in self.index.search(" ".join(keywords(query, 12)), limit=limit):
                    add(hit["path"])
            except Exception:  # noqa: S110 - index is best-effort
                pass
        return list(chosen)

    def excerpts(self, paths: list[str], budget_chars: int = 24000, per_file_max: int = 8000) -> str:
        out: list[str] = []
        remaining = budget_chars
        for rel in paths:
            if remaining <= 500:
                break
            path = self.workspace / rel
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if is_binary_bytes(data):
                continue
            lines = data.decode("utf-8", errors="replace").splitlines()
            cap = min(per_file_max, remaining)
            body: list[str] = []
            used = 0
            for i, line in enumerate(lines, 1):
                entry = f"{i:>5}\t{line}"
                if used + len(entry) + 1 > cap:
                    body.append(f"... ({len(lines) - i + 1} more lines; use read_file with offset={i})")
                    break
                body.append(entry)
                used += len(entry) + 1
            block = f"### {rel}\n" + "\n".join(body)
            out.append(block)
            remaining -= len(block)
        return self._redact("\n\n".join(out))

    def memory_block(self, query: str, limit: int = 8) -> str:
        if self.memory is None:
            return ""
        try:
            return str(self.memory.context_block(query, limit=limit))
        except Exception:
            return ""

    def task_context(self, query: str, hints: list[str] | None = None, budget_chars: int = 30000) -> str:
        files = self.relevant_files(query, hints)
        parts = []
        mem = self.memory_block(query)
        if mem:
            parts.append(mem)
        if files:
            parts.append("Possibly relevant files (verify before relying on them):\n" + self.excerpts(files, budget_chars=budget_chars))
        return "\n\n".join(parts)
