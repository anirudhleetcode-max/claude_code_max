"""Memory facade: a project store plus an optional global (cross-project) engineering store."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..core.util import atomic_write_json, utcnow_iso
from .store import MemoryItem, MemoryLayer, MemoryStore

CONTEXT_HEADER = "Memory (hints from earlier work — verify against the repository; repository state wins):"
STALE_NOTE = (
    "Items marked STALE reference files that changed after they were recorded, or were invalidated; "
    "the repository is the source of truth: re-check them before relying on them."
)


def command_confidence(successes: int, failures: int) -> float:
    """Confidence that a command works: grows with successes, decays only while it has never succeeded."""
    if successes > 0:
        return round(min(0.95, 0.5 + 0.1 * successes), 4)
    return round(max(0.2, 0.5 - 0.1 * failures), 4)


class MemoryManager:
    """Facade over a project store (``<project>/.agent`` state database) and a global engineering store."""

    def __init__(self, project: MemoryStore, global_store: MemoryStore | None = None) -> None:
        self.project = project
        self.global_store = global_store

    def _store_for(self, layer: str) -> MemoryStore:
        if layer == MemoryLayer.ENGINEERING and self.global_store is not None:
            return self.global_store
        return self.project

    def _stores(self) -> list[MemoryStore]:
        return [self.project] if self.global_store is None else [self.project, self.global_store]

    def add(self, **kwargs: Any) -> MemoryItem:
        """``MemoryStore.add``; the ``engineering`` layer goes to the global store when there is one."""
        return self._store_for(str(kwargs.get("layer", ""))).add(**kwargs)

    def search(
        self,
        query: str,
        *,
        layers: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Merged results from both stores; fresh before stale, then relevance; project wins ties."""
        layer_list = list(layers) if layers is not None else None
        kind_list = list(kinds) if kinds is not None else None
        merged: list[MemoryItem] = []
        for store in self._stores():
            merged += store.search(query, layers=layer_list, kinds=kind_list, limit=limit)
        # Stable sort keeps project items ahead of global ones on equal keys.
        merged.sort(key=lambda i: (i.stale, -i.score, -i.confidence))
        return merged[:limit]

    def invalidate(self, *, layer: str | None = None, kind: str | None = None, item_id: str | None = None) -> int:
        """Invalidate matching items in every store (see ``MemoryStore.invalidate``)."""
        return sum(store.invalidate(layer=layer, kind=kind, item_id=item_id) for store in self._stores())

    def context_block(self, query: str, limit: int = 8) -> str:
        """Prompt-ready memory hints for ``query`` ("" when nothing is relevant)."""
        items = self.search(query, limit=limit)
        if not items:
            return ""
        lines = [CONTEXT_HEADER, *(f"- {item.render()}" for item in items)]
        if any(item.stale for item in items):
            lines.append(STALE_NOTE)
        return "\n".join(lines)

    def record_command(self, command: str, *, kind: str, ok: bool, duration_s: float, note: str = "") -> MemoryItem:
        """Remember how a command (test/build/lint/...) behaved; statistics accumulate in ``meta``."""
        key = f"{kind}:{command}"
        existing = self.project.find(layer=MemoryLayer.COMMAND, kind=kind, key=key)
        meta = dict(existing.meta) if existing is not None else {}
        successes = int(meta.get("successes", 0)) + (1 if ok else 0)
        failures = int(meta.get("failures", 0)) + (0 if ok else 1)
        meta.update(
            successes=successes, failures=failures, last_ok=ok, last_duration_s=round(float(duration_s), 3),
            last_run=utcnow_iso(),
        )
        outcome = "succeeded" if ok else "failed"
        content = (
            f"{kind} command `{command}`: last run {outcome} in {duration_s:.1f}s "
            f"({successes} succeeded, {failures} failed so far)"
        )
        if note:
            content += f". {note}"
        confidence = command_confidence(successes, failures)
        if existing is not None:
            # Statistics are bookkeeping, not knowledge: edit in place rather than growing history.
            return self.project.update(existing.id, content=content, confidence=confidence, meta=meta, new_version=False)
        return self.project.add(
            layer=MemoryLayer.COMMAND, kind=kind, key=key, content=content, source="command-runner",
            confidence=confidence, tags=[kind], meta=meta,
        )

    def record_decision(
        self,
        title: str,
        decision: str,
        rationale: str,
        *,
        alternatives: list[str] | None = None,
        task_id: str = "",
    ) -> MemoryItem:
        """Record an architecture decision (ADR); recording the same title again supersedes it."""
        alts = [a for a in (alternatives or []) if a.strip()]
        content = f"{title}: {decision}. Rationale: {rationale}"
        if alts:
            content += f". Alternatives considered: {'; '.join(alts)}"
        return self.project.add(
            layer=MemoryLayer.DECISION, kind="adr", key=title, content=content,
            source=f"task:{task_id}" if task_id else "agent", confidence=0.8, tags=["adr"],
            meta={"title": title, "decision": decision, "rationale": rationale, "alternatives": alts, "task_id": task_id},
        )

    def known_commands(self, kind: str) -> list[MemoryItem]:
        """Command memories of ``kind``, most trusted first."""
        items = self.project.list(layer=MemoryLayer.COMMAND, kind=kind, limit=1000)
        return sorted(items, key=lambda i: -i.confidence)  # stable: most recent first on ties

    def write_snapshot(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        """Atomically write a JSON snapshot of the project's active memory."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        decisions: list[dict[str, Any]] = []
        for item in self.project.export():
            if item["layer"] == MemoryLayer.DECISION:
                decisions.append(item)
            else:
                grouped.setdefault(item["layer"], []).append(item)
        data: dict[str, Any] = {"generated": utcnow_iso(), "project": grouped, "decisions": decisions}
        data.update(extra or {})
        atomic_write_json(path, data)

    def close(self) -> None:
        for store in self._stores():
            store.close()
