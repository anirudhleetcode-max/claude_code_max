"""Platform-appropriate locations for global configuration and data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "ai-engineer"
PROJECT_DIR_NAME = ".agent"


def global_config_dir() -> Path:
    override = os.environ.get("AIE_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / APP_NAME


def global_data_dir() -> Path:
    override = os.environ.get("AIE_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform in ("win32", "darwin"):
        return global_config_dir()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def project_state_dir(workspace: Path) -> Path:
    return workspace / PROJECT_DIR_NAME


def find_workspace_root(start: Path) -> Path:
    """Nearest ancestor containing ``.agent`` or ``.git``; otherwise ``start``."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / PROJECT_DIR_NAME).is_dir() or (candidate / ".git").exists():
            return candidate
    return start
