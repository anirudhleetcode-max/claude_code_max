"""Recoverable checkpoints.

git workspaces: snapshot commits of the working tree under refs/ai-engineer/ (user index,
HEAD and branch untouched).

other workspaces: copy-on-first-write file backups. Each checkpoint opens an epoch; the
first time a file is modified during an epoch its previous content is saved into that
epoch's directory. Restoring checkpoint K replays the backups of every epoch >= K from
newest to oldest, so the oldest saved version (the state at K) wins.

Every restore first creates a safety checkpoint, so restores are themselves reversible.
"""

from __future__ import annotations

import difflib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from ..core.errors import StateError
from ..core.events import EventBus, EventType
from ..core.ids import new_id
from ..core.util import atomic_write_json
from ..git.repo import GitRepo
from ..security.secrets import redact_secret_files_in_diff
from ..tools.file_state import FileStateTracker
from .store import CheckpointRecord, StateStore

_NOISE_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", "node_modules", ".next",
    ".nuxt", ".cache", ".parcel-cache", ".turbo", ".gradle", ".hypothesis", "htmlcov", ".agent",
}
_NOISE_SUFFIXES = (".pyc", ".pyo", ".class", ".o", ".obj")
_NOISE_NAMES = {".coverage", ".DS_Store", "Thumbs.db"}


def is_noise(rel: str) -> bool:
    """Caches and build artifacts produced by running tools, not by the agent's intent."""
    parts = rel.replace("\\", "/").split("/")
    return any(p in _NOISE_DIRS for p in parts[:-1]) or parts[-1] in _NOISE_NAMES or parts[-1].endswith(_NOISE_SUFFIXES)


class CheckpointManager:
    def __init__(
        self,
        workspace: Path,
        store: StateStore,
        files: FileStateTracker,
        state_dir: Path,
        git: GitRepo | None = None,
        bus: EventBus | None = None,
        is_secret: Callable[[str], bool] | None = None,
    ) -> None:
        self.workspace = workspace
        self.is_secret = is_secret
        self.store = store
        self.files = files
        self.dir = state_dir / "checkpoints"
        self.git = git
        self.bus = bus
        self._epoch: str | None = None
        files.on_first_write = self._backup_hook

    @property
    def kind(self) -> str:
        return "git" if self.git is not None else "files"

    # ---- file-backup mode ---------------------------------------------------------------

    def _manifest_path(self, cp_id: str) -> Path:
        return self.dir / cp_id / "manifest.json"

    def _load_manifest(self, cp_id: str) -> dict[str, dict[str, str | None]]:
        path = self._manifest_path(cp_id)
        if not path.exists():
            return {}
        data: dict[str, dict[str, str | None]] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def _backup_hook(self, rel: str, original: bytes | None) -> None:
        if self.git is not None:
            return
        if self._epoch is None:
            # a new process (e.g. a resumed task) keeps backing up into the latest epoch, so its
            # writes stay visible to diffs and reversible by restores
            latest = self.store.latest_checkpoint("files")
            if latest is None or not (self.dir / latest.id).is_dir():
                return
            self._epoch = latest.id
        manifest = self._load_manifest(self._epoch)
        if rel in manifest:
            return
        entry: dict[str, str | None] = {"existed": "1" if original is not None else None, "blob": None}
        if original is not None:
            blob_name = new_id("blob")
            blob = self.dir / self._epoch / "blobs" / blob_name
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(original)
            entry["blob"] = blob_name
        manifest[rel] = entry
        atomic_write_json(self._manifest_path(self._epoch), manifest)

    # ---- public API ------------------------------------------------------------------------

    async def create(self, label: str, task_id: str | None = None, subtask_id: str | None = None, verified: bool = False) -> CheckpointRecord:
        cp_id = new_id("ckpt")
        record = CheckpointRecord(id=cp_id, task_id=task_id, subtask_id=subtask_id, label=label, kind=self.kind, verified=verified)
        if self.git is not None:
            commit, tree = await self.git.snapshot(cp_id, f"ai-engineer checkpoint: {label}")
            record.ref, record.tree = commit, tree
        else:
            (self.dir / cp_id).mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._manifest_path(cp_id), {})
            self._epoch = cp_id
            # files written from now on are backed up into this epoch
            self.files.reset_changes()
        self.store.add_checkpoint(record)
        if self.bus is not None:
            self.bus.emit(EventType.CHECKPOINT_CREATED, f"checkpoint {cp_id[-8:]}: {label}", data={"checkpoint_id": cp_id, "verified": verified, "kind": record.kind})
        return record

    async def changed_files_since(self, cp_id: str) -> list[str]:
        record = self._require(cp_id)
        if record.kind == "git" and self.git is not None and record.tree:
            return [p for p in await self.git.changed_files_since(record.tree) if not is_noise(p)]
        changed: set[str] = set()
        for epoch in self._epochs_from(record):
            changed |= set(self._load_manifest(epoch.id))
        return sorted(p for p in changed if not is_noise(p) and self._differs_from_backup(record, p))

    async def diff_since(self, cp_id: str, paths: list[str] | None = None, stat: bool = False) -> str:
        record = self._require(cp_id)
        if record.kind == "git" and self.git is not None and record.tree:
            relevant = await self.changed_files_since(cp_id)
            if paths is not None:
                relevant = [p for p in relevant if p in paths]
            if not relevant:
                return ""
            return self._redact(await self.git.diff_since(record.tree, stat=stat, paths=relevant), stat)
        chunks = []
        for rel in await self.changed_files_since(cp_id):
            if paths and rel not in paths:
                continue
            before = self._original_content(record, rel)
            current = self.workspace / rel
            after = current.read_bytes() if current.exists() else None
            a = (before or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            b = (after or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            if stat:
                added = sum(1 for line in difflib.ndiff(a, b) if line.startswith("+ "))
                removed = sum(1 for line in difflib.ndiff(a, b) if line.startswith("- "))
                chunks.append(f" {rel} | +{added} -{removed}\n")
            else:
                chunks.append("".join(difflib.unified_diff(a, b, f"a/{rel}" if before is not None else "/dev/null", f"b/{rel}" if after is not None else "/dev/null")))
        return self._redact("".join(chunks), stat)

    def _redact(self, diff: str, stat: bool) -> str:
        """Diffs go to the reviewer model, reports, the CLI and the dashboard: hide secret-file values."""
        if stat or self.is_secret is None:
            return diff
        return redact_secret_files_in_diff(diff, self.is_secret)

    async def restore(self, cp_id: str) -> tuple[CheckpointRecord, list[str]]:
        """Restore the workspace to ``cp_id``. Returns (safety checkpoint, restored/removed paths)."""
        record = self._require(cp_id)
        safety = await self.create(f"before restoring {cp_id}", task_id=record.task_id, subtask_id=record.subtask_id)
        try:
            touched = await self._restore_into(record, safety)
        except Exception as exc:
            # a restore that dies half-way (disk full, killed git, locked file) leaves a mixed tree:
            # the safety checkpoint taken above is the way back, so name it
            if self.bus is not None:
                self.bus.emit(EventType.WARNING, f"restore of {cp_id} failed: {exc}; undo with the safety checkpoint {safety.id}", data={"safety_checkpoint": safety.id})
            raise StateError(f"restore of {cp_id} failed part-way ({exc}); return to the state before the restore with: aie restore {safety.id}") from exc
        self.files.forget_reads()
        if self.bus is not None:
            self.bus.emit(EventType.CHECKPOINT_RESTORED, f"restored checkpoint {cp_id[-8:]} ({len(touched)} file(s)); safety checkpoint {safety.id[-8:]}", data={"checkpoint_id": cp_id, "safety_checkpoint": safety.id, "paths": touched[:100]})
        return safety, sorted(touched)

    async def _restore_into(self, record: CheckpointRecord, safety: CheckpointRecord) -> list[str]:
        touched: list[str] = []
        if record.kind == "git" and self.git is not None and record.tree:
            removed = await self.git.restore_tree(record.tree)
            touched = sorted(set(await self.git.changed_files_since(safety.tree or record.tree)) | set(removed))
        else:
            originals: dict[str, bytes | None] = {}
            for epoch in reversed(self._epochs_from(record)):
                for rel, entry in self._load_manifest(epoch.id).items():
                    originals[rel] = self._read_blob(epoch.id, entry)
            for rel, content in originals.items():
                target = self.workspace / rel
                current = target.read_bytes() if target.is_file() else None
                if current == content:
                    continue
                # back up the pre-restore state into the safety epoch so this restore can be undone
                self.files.before_write(target)
                if content is None:
                    target.unlink()
                    self.files.after_write(target, deleted=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    self.files.after_write(target)
                touched.append(rel)
        return touched

    def list_checkpoints(self, task_id: str | None = None) -> list[CheckpointRecord]:
        return self.store.list_checkpoints(task_id)

    def purge_files(self, cp_id: str) -> None:
        shutil.rmtree(self.dir / cp_id, ignore_errors=True)

    # ---- internals --------------------------------------------------------------------------

    def _require(self, cp_id: str) -> CheckpointRecord:
        record = self.store.get_checkpoint(cp_id)
        if record is None:
            raise KeyError(f"no such checkpoint: {cp_id}")
        return record

    def _epochs_from(self, record: CheckpointRecord) -> list[CheckpointRecord]:
        all_files = [c for c in self.store.list_checkpoints(limit=100000) if c.kind == "files"]
        all_files.sort(key=lambda c: (c.created, c.id))
        return [c for c in all_files if (c.created, c.id) >= (record.created, record.id)]

    def _read_blob(self, epoch_id: str, entry: dict[str, str | None]) -> bytes | None:
        if not entry.get("existed"):
            return None
        blob = entry.get("blob")
        return (self.dir / epoch_id / "blobs" / str(blob)).read_bytes() if blob else b""

    def _original_content(self, record: CheckpointRecord, rel: str) -> bytes | None:
        for epoch in self._epochs_from(record):
            manifest = self._load_manifest(epoch.id)
            if rel in manifest:
                return self._read_blob(epoch.id, manifest[rel])
        current = self.workspace / rel
        return current.read_bytes() if current.exists() else None

    def _differs_from_backup(self, record: CheckpointRecord, rel: str) -> bool:
        before = self._original_content(record, rel)
        current = self.workspace / rel
        after = current.read_bytes() if current.exists() else None
        return before != after
