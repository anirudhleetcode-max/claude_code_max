"""`aie doctor`: a read-only health report of the installation, workspace and model providers.

Never prints secret values: credentials are reported only as "set" / "not set" by variable name,
and every detail passes through the secret redactor.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import global_config_dir, global_data_dir, project_state_dir
from ..core.errors import ConfigError
from ..security.secrets import Redactor

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
MARKS = {OK: "✔", WARN: "!", FAIL: "✘", INFO: "·"}


@dataclass
class Check:
    section: str
    name: str
    status: str
    detail: str = ""


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 10.0) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return out.returncode, (out.stdout or out.stderr).strip()


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".aie-doctor-", delete=True):
            pass
        return True
    except OSError:
        return False


# What makes a provider name available without a [models.providers.<name>] section.
_PROVIDER_HINTS = {
    "anthropic": "set ANTHROPIC_API_KEY (auto-detected) or add [models.providers.anthropic]",
    "openai": "set OPENAI_API_KEY (auto-detected) or add [models.providers.openai]",
    "google": "set GEMINI_API_KEY or GOOGLE_API_KEY (auto-detected) or add [models.providers.google]",
    "openai_compatible": "set OPENAI_COMPATIBLE_BASE_URL (auto-detected) or add [models.providers.openai_compatible]",
    "scripted": "add [models.providers.scripted] with type = \"scripted\" and options.script_file",
}


class Doctor:
    def __init__(self, workspace: Path, overrides: dict[str, Any] | None = None, *, connect: bool = True, timeout_s: float = 15.0) -> None:
        self.workspace = workspace
        self.overrides = overrides or {}
        self.connect = connect
        self.timeout_s = timeout_s
        self.checks: list[Check] = []
        self.redactor = Redactor()
        self.settings: Any = None

    def add(self, section: str, name: str, status: str, detail: str = "", *, redact: bool = True) -> None:
        # paths and versions are not secrets; the redactor's entropy heuristic would mangle
        # random-looking directory names, so callers opt out for those
        self.checks.append(Check(section, name, status, self.redactor.redact_text(detail) if redact else detail))

    # ---- sections --------------------------------------------------------------------------

    def runtime(self) -> None:
        v = sys.version_info
        ok = v >= (3, 11)
        self.add("runtime", "python", OK if ok else FAIL, f"{platform.python_version()} ({sys.executable})" + ("" if ok else "; 3.11 or newer is required"), redact=False)
        self.add("runtime", "platform", INFO, f"{platform.system()} {platform.release()} ({platform.machine()})")
        self.add("runtime", "ai-engineer", INFO, __version__)

    def dependencies(self) -> None:
        for dist in ("pydantic", "httpx"):
            ver = _version(dist)
            self.add("dependencies", dist, OK if ver else FAIL, ver or "missing: reinstall ai-engineer")
        optional = {
            "anthropic": "needed for providers of type 'anthropic' (pip install 'ai-engineer[anthropic]')",
            "starlette": "needed for the web dashboard (pip install 'ai-engineer[web]')",
            "uvicorn": "needed for the web dashboard (pip install 'ai-engineer[web]')",
            "playwright": "needed for the browser tool (pip install 'ai-engineer[browser]')",
        }
        used_types = self._provider_types_in_use()
        for dist, why in optional.items():
            ver = _version(dist)
            if ver:
                self.add("dependencies", dist, OK, ver)
            else:
                needed = dist == "anthropic" and "anthropic" in used_types
                self.add("dependencies", dist, FAIL if needed else INFO, f"not installed; {why}")

    def tools(self) -> None:
        git = shutil.which("git")
        if git:
            code, out = _run([git, "--version"])
            self.add("tools", "git", OK if code == 0 else WARN, out)
        else:
            self.add("tools", "git", WARN, "not found: checkpoints fall back to file backups, no branches or commits")
        rg = shutil.which("rg")
        self.add("tools", "ripgrep", OK if rg else INFO, rg or "not found: text search uses the slower built-in scanner")
        sandbox = getattr(getattr(self.settings, "terminal", None), "sandbox", None)
        docker = shutil.which("docker")
        if sandbox == "docker":
            code, out = _run([docker, "info", "--format", "{{.ServerVersion}}"], timeout=15) if docker else (1, "docker not found")
            self.add("tools", "docker sandbox", OK if code == 0 else FAIL, f"server {out}" if code == 0 else f"configured but unusable: {out[:200]}")
        else:
            self.add("tools", "docker sandbox", INFO, "not enabled ([terminal] sandbox = \"docker\" isolates commands)")
        if _version("playwright"):
            exe = (getattr(getattr(self.settings, "web", None), "browser_executable", None) or os.environ.get("AIE_BROWSER_EXECUTABLE") or "")
            if exe:
                self.add("tools", "browser", OK if Path(exe).exists() else FAIL, exe if Path(exe).exists() else f"configured executable not found: {exe}")
            else:
                self.add("tools", "browser", INFO, "using Playwright's own Chromium (run `playwright install chromium` if the browser tool fails)")

    def database(self) -> None:
        self.add("database", "sqlite", OK, sqlite3.sqlite_version)
        try:
            con = sqlite3.connect(":memory:")
            con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            con.close()
            self.add("database", "fts5", OK, "available (memory search)")
        except sqlite3.OperationalError:
            self.add("database", "fts5", WARN, "unavailable: memory search falls back to slower matching")
        state = project_state_dir(self.workspace)
        for name in ("state.db", "memory.db"):
            path = state / name
            if not path.exists():
                self.add("database", name, INFO, "not created yet")
                continue
            try:
                con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
                try:
                    result = con.execute("PRAGMA quick_check").fetchone()[0]
                finally:
                    con.close()
                self.add("database", name, OK if result == "ok" else FAIL, "integrity ok" if result == "ok" else f"integrity check: {result}")
            except sqlite3.Error as exc:
                self.add("database", name, FAIL, f"cannot read: {exc}")

    def workspace_checks(self) -> None:
        ws = self.workspace
        self.add("workspace", "path", INFO, str(ws), redact=False)
        state = project_state_dir(ws)
        if state.exists():
            self.add("workspace", ".agent", OK if _writable(state) else FAIL, "writable" if _writable(state) else f"{state} is not writable")
        else:
            self.add("workspace", ".agent", WARN, "not initialised: run `aie init` (created automatically on first run)")
        for label, directory in (("global config dir", global_config_dir()), ("global data dir", global_data_dir())):
            if directory.exists():
                self.add("workspace", label, OK if os.access(directory, os.W_OK) else FAIL, str(directory), redact=False)
            else:
                self.add("workspace", label, INFO, f"{directory} (created on first use)", redact=False)
        if self.settings is not None:
            self.add("workspace", "configuration", OK, "loaded (project, global, environment and .env layers)")
        git = shutil.which("git")
        in_repo = bool(git) and ((ws / ".git").exists() or _run([str(git), "rev-parse", "--is-inside-work-tree"], cwd=ws)[1] == "true")
        if git and in_repo:
            has_commits = _run([str(git), "rev-parse", "--verify", "-q", "HEAD"], cwd=ws)[0] == 0
            branch = _run([str(git), "rev-parse", "--abbrev-ref", "HEAD"], cwd=ws)[1] if has_commits else "(no commits yet)"
            _, status = _run([str(git), "status", "--porcelain"], cwd=ws)
            changes = len([line for line in status.splitlines() if line.strip()])
            self.add("workspace", "git repository", OK, f"branch {branch or '?'}; " + (f"{changes} uncommitted change(s) (agent changes will be left uncommitted)" if changes else "clean"))
            if (ws / ".env").exists():
                code, _ = _run([str(git), "check-ignore", "-q", ".env"], cwd=ws)
                self.add("workspace", ".env", OK if code == 0 else WARN, "ignored by git" if code == 0 else ".env exists and is NOT ignored by git: it could be committed")
        else:
            self.add("workspace", "git repository", WARN, "not a git repository: file-backup checkpoints; no branches or commits")

    def security(self) -> None:
        if self.settings is None:
            return
        perms = self.settings.permissions
        self.add("security", "mode", INFO, f"{perms.mode} (max level {perms.max_level.name}; high-risk actions always need approval)")
        self.add("security", "secret env stripping", OK if self.settings.terminal.strip_secret_env else WARN,
                 "provider keys and secret-like variables are removed from commands the agent runs" if self.settings.terminal.strip_secret_env else "disabled: commands the agent runs can read your credentials")

    def providers(self) -> None:
        if self.settings is None:
            return
        from ..models.router import ModelRouter

        router = ModelRouter(self.settings.models)
        roles = router.configured_roles()
        if not roles:
            self.add("providers", "roles", FAIL, "no model configured: set AIE_MODEL=provider:model-id or [models.roles] in .agent/config.toml")
        for role, chain in roles.items():
            self.add("providers", f"role {role}", INFO, " → ".join(chain))
        used = {ref.split(":", 1)[0] for chain in roles.values() for ref in chain}
        known = set(router.registry.names())
        for name in sorted(used - known):  # built-in types (e.g. scripted) are created on first use
            try:
                router.registry.get(name)
                known.add(name)
            except Exception as exc:
                hint = _PROVIDER_HINTS.get(name)
                self.add("providers", name, FAIL, f"referenced by a role but not configured: {hint}" if hint else f"referenced by a role but not configured: {exc}")
        for name in sorted(known):
            cfg = self.settings.models.providers.get(name)
            kind = getattr(cfg, "type", "?") if cfg else "?"
            where = f" {cfg.base_url}" if cfg is not None and getattr(cfg, "base_url", None) else ""
            if name not in used:
                self.add("providers", name, INFO, f"type {kind}{where}; registered, not used by any role")
                continue
            key_env = getattr(cfg, "api_key_env", None) if cfg else None
            if key_env:
                present = bool(os.environ.get(key_env))
                self.add("providers", f"{name} credentials", OK if present else FAIL, f"{key_env} is {'set' if present else 'not set'}")
            self.add("providers", name, INFO, f"type {kind}{where}")
        if self.connect and used:
            asyncio.run(self._health(router, sorted(used & known)))
        elif used:
            self.add("providers", "connectivity", INFO, "not checked (--offline)")

    async def _health(self, router: Any, names: list[str]) -> None:
        try:
            for name in names:
                try:
                    health = await asyncio.wait_for(router.registry.get(name).health_check(), self.timeout_s)
                except Exception as exc:  # timeouts, configuration errors
                    self.add("providers", f"{name} connectivity", FAIL, f"{type(exc).__name__}: {exc}")
                    continue
                if health.ok and health.detail.startswith("no health endpoint"):
                    self.add("providers", f"{name} connectivity", WARN, "UNVERIFIED: the provider has no health endpoint; run `aie providers test`")
                elif health.ok:
                    self.add("providers", f"{name} connectivity", OK, f"reachable in {health.latency_s:.1f}s" + (f"; {len(health.models)} model(s) listed" if health.models else ""))
                else:
                    self.add("providers", f"{name} connectivity", FAIL, health.detail[:300])
        finally:
            with contextlib.suppress(Exception):
                await router.aclose()

    # ---- driver ------------------------------------------------------------------------------

    def _provider_types_in_use(self) -> set[str]:
        if self.settings is None:
            return set()
        roles = getattr(self.settings.models, "roles", {}) or {}
        names = {ref.split(":", 1)[0] for chain in roles.values() for ref in (chain or [])}
        return {getattr(cfg, "type", "") for name, cfg in self.settings.models.providers.items() if name in names}

    def run(self) -> list[Check]:
        from ..config.loader import load_dotenv, load_settings

        load_dotenv(self.workspace / ".env")  # as `aie run` does, so credential checks match reality
        try:
            self.settings = load_settings(self.workspace, self.overrides)
        except ConfigError as exc:
            self.add("workspace", "configuration", FAIL, str(exc))
        steps: list[Callable[[], None]] = [self.runtime, self.dependencies, self.tools, self.database, self.workspace_checks, self.security, self.providers]
        for step in steps:
            try:
                step()
            except Exception as exc:  # a broken check must not hide the others
                self.add(step.__name__, "check crashed", FAIL, f"{type(exc).__name__}: {exc}")
        return self.checks


def render(checks: list[Check]) -> str:
    lines: list[str] = []
    section = None
    for c in checks:
        if c.section != section:
            section = c.section
            lines.append(f"\n{section.upper()}")
        lines.append(f"  {MARKS.get(c.status, '?')} {c.name}: {c.detail}" if c.detail else f"  {MARKS.get(c.status, '?')} {c.name}")
    fails = sum(1 for c in checks if c.status == FAIL)
    warns = sum(1 for c in checks if c.status == WARN)
    lines.append("")
    lines.append("All required checks passed." if not fails else f"{fails} problem(s) found.")
    if warns:
        lines.append(f"{warns} warning(s).")
    return "\n".join(lines).lstrip("\n")


def as_json(checks: list[Check]) -> list[dict[str, str]]:
    return [asdict(c) for c in checks]
