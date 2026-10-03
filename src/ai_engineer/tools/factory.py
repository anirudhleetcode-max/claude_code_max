"""Assembly helpers for the tool layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.settings import Settings
from ..core.cancel import CancellationToken
from ..core.events import EventBus
from ..security.paths import PathGuard
from ..security.secrets import Redactor
from .base import Tool, ToolContext
from .file_state import FileStateTracker
from .process import ProcessManager


def builtin_tool_classes() -> list[type[Tool]]:
    from .builtin.browser import BROWSER_TOOLS
    from .builtin.database import DB_TOOLS
    from .builtin.env import ENV_TOOLS
    from .builtin.fs import FS_TOOLS
    from .builtin.git_tools import GIT_TOOLS
    from .builtin.search import SEARCH_TOOLS
    from .builtin.terminal import TERMINAL_TOOLS
    from .builtin.web import WEB_TOOLS

    classes: list[type[Tool]] = [
        *FS_TOOLS, *SEARCH_TOOLS, *TERMINAL_TOOLS, *GIT_TOOLS, *ENV_TOOLS, *DB_TOOLS, *WEB_TOOLS, *BROWSER_TOOLS,
    ]
    try:
        from .builtin.testing import TESTING_TOOLS

        classes += TESTING_TOOLS
    except ImportError:  # pragma: no cover - testing tools ship with the package
        pass
    try:
        from .builtin.memory_tools import MEMORY_TOOLS

        classes += MEMORY_TOOLS
    except ImportError:  # pragma: no cover
        pass
    return classes


def default_tools() -> list[Tool]:
    return [cls() for cls in builtin_tool_classes()]


def make_context(
    workspace: Path,
    settings: Settings | None = None,
    *,
    bus: EventBus | None = None,
    cancel: CancellationToken | None = None,
    redactor: Redactor | None = None,
    **services: Any,
) -> ToolContext:
    settings = settings or Settings()
    workspace = workspace.resolve()
    guard = PathGuard(workspace, settings.permissions.protected_paths, settings.permissions.secret_files)
    redactor = redactor or Redactor()
    bus = bus or EventBus(redactor)
    return ToolContext(
        workspace=workspace,
        settings=settings,
        guard=guard,
        redactor=redactor,
        bus=bus,
        cancel=cancel or CancellationToken(),
        files=FileStateTracker(workspace),
        processes=ProcessManager(),
        **services,
    )
