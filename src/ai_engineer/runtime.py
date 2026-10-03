"""Runtime: assembles every service for one workspace from configuration.

The CLI, web UI, benchmarks and tests all go through this class, so the agent
behaves identically regardless of the front end.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_PROJECT_CONFIG,
    Settings,
    global_data_dir,
    load_dotenv,
    load_settings,
    project_state_dir,
)
from .context.builder import ContextBuilder
from .core.events import EventBus, EventType
from .core.ids import new_id
from .executor.control import CONTROL_TOOLS
from .models.registry import ProviderRegistry
from .models.router import ModelRouter
from .observability.logging_setup import configure_logging
from .observability.metrics import MetricsCollector
from .observability.trace import TraceWriter
from .orchestrator.questions import NoQuestions, QuestionBroker
from .security.audit import AuditLog
from .security.secrets import Redactor
from .tasks.checkpoints import CheckpointManager
from .tasks.store import StateStore, Task, TaskStatus, lease_owner_id
from .tools.approval import ApprovalBroker, DenyAllBroker
from .tools.executor import ToolExecutor, ToolRegistry
from .tools.factory import default_tools, make_context
from .tools.permissions import PermissionPolicy

log = logging.getLogger(__name__)

# Everything under .agent/ (including this file) stays out of version control so the
# user's working tree is never dirtied. To version the configuration deliberately:
#   git add -f .agent/config.toml
AGENT_GITIGNORE = "# AI Engineer state. To version the config: git add -f .agent/config.toml\n*\n"


def init_project_dir(workspace: Path) -> Path:
    """Create ``.agent/`` with a config template and a .gitignore. Idempotent."""
    state = project_state_dir(workspace)
    for sub in ("logs", "reports", "checkpoints", "indexes"):
        (state / sub).mkdir(parents=True, exist_ok=True)
    gitignore = state / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(AGENT_GITIGNORE, encoding="utf-8")
    config = state / "config.toml"
    if not config.exists():
        config.write_text(DEFAULT_PROJECT_CONFIG, encoding="utf-8")
    return state


class Runtime:
    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        *,
        interactive: bool = False,
        approvals: ApprovalBroker | None = None,
        questions: QuestionBroker | None = None,
        registry: ProviderRegistry | None = None,
        session_id: str | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.settings = settings
        self.interactive = interactive
        self.session_id = session_id or new_id("sess")
        self.state_dir = init_project_dir(self.workspace)
        self.redactor = Redactor()
        self.bus = EventBus(self.redactor)
        self.bus.context["session_id"] = self.session_id
        configure_logging(settings.observability.log_level, self.state_dir / "logs" / "agent.log", self.redactor, settings.observability.debug)
        self.trace: TraceWriter | None = None
        if settings.observability.trace:
            self.trace = TraceWriter(self.state_dir / "logs" / f"trace-{self.session_id}.jsonl", debug=settings.observability.debug)
            self.trace.attach(self.bus)
        self.store = StateStore(self.state_dir / "state.db")
        self.bus.subscribe(self.store.add_event)
        self.metrics = MetricsCollector()
        self.metrics.attach(self.bus)
        self.audit = AuditLog(self.state_dir / "logs" / "audit.jsonl", self.redactor)
        self.router = ModelRouter(settings.models, registry=registry, bus=self.bus)
        self.registry = ToolRegistry([*default_tools(), *(cls() for cls in CONTROL_TOOLS)])
        self.policy = PermissionPolicy(settings.permissions)
        self.approvals: ApprovalBroker = approvals or DenyAllBroker(self.bus)
        if self.approvals.bus is None:
            self.approvals.bus = self.bus
        self.questions: QuestionBroker = questions or NoQuestions()
        self.executor = ToolExecutor(self.registry, self.policy, self.approvals, self.audit, self.bus, settings.agent.tool_result_max_chars)
        self.git: Any = None
        self.memory: Any = None
        self.index: Any = None
        self.profile: Any = None
        self.lease_owner = lease_owner_id(self.session_id)
        self._open_optional_services()
        from .tester.engine import ValidationEngine

        self.validation = ValidationEngine(self.workspace, settings, self.profile)
        self.tool_ctx = make_context(
            self.workspace, settings, bus=self.bus, redactor=self.redactor,
            repo_index=self.index, memory=self.memory, validation=self.validation, git=self.git, profile=self.profile,
        )
        self.context = ContextBuilder(self.workspace, self.profile, self.index, self.memory, self.redactor)
        self.checkpoints = CheckpointManager(self.workspace, self.store, self.tool_ctx.files, self.state_dir, self.git, self.bus)
        recovered = self.store.recover_interrupted()
        for task in recovered:
            self.bus.emit(EventType.TASK_INTERRUPTED, f"recovered interrupted task {task.id}", task_id=task.id)
        from .orchestrator.pipeline import Orchestrator

        self.orchestrator = Orchestrator(self)

    @classmethod
    def open(
        cls,
        workspace: Path,
        overrides: dict[str, Any] | None = None,
        *,
        interactive: bool = False,
        approvals: ApprovalBroker | None = None,
        questions: QuestionBroker | None = None,
        registry: ProviderRegistry | None = None,
        use_global_config: bool = True,
    ) -> Runtime:
        workspace = workspace.resolve()
        load_dotenv(workspace / ".env")
        settings = load_settings(workspace, overrides, use_global=use_global_config)
        if registry is not None:
            registry.settings = settings.models
        return cls(workspace, settings, interactive=interactive, approvals=approvals, questions=questions, registry=registry)

    # ---- services ---------------------------------------------------------------------

    def _open_optional_services(self) -> None:
        from .git.repo import git_available

        if git_available() and ((self.workspace / ".git").exists() or self._inside_git()):
            from .git.repo import GitRepo

            self.git = GitRepo(self.workspace, self.settings.git.author_name, self.settings.git.author_email)
        try:
            from .memory.manager import MemoryManager
            from .memory.store import MemoryStore

            global_dir = Path(self.settings.memory.global_dir) if self.settings.memory.global_dir else global_data_dir()
            global_dir.mkdir(parents=True, exist_ok=True)
            self.memory = MemoryManager(
                MemoryStore(self.state_dir / "memory.db", self.workspace, self.redactor),
                MemoryStore(global_dir / "engineering-memory.db", None, self.redactor),
            )
        except Exception as exc:
            log.warning("memory unavailable: %s", exc)
            self.memory = None
        try:
            from .repo.index import RepoIndex

            self.index = RepoIndex(self.workspace, self.state_dir / "indexes" / "index.db")
        except Exception as exc:
            log.warning("repository index unavailable: %s", exc)
            self.index = None
        from .repo.discovery import load_profile

        self.profile = load_profile(self.state_dir / "project.json")

    def _inside_git(self) -> bool:
        import subprocess

        try:
            out = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=self.workspace, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return out.stdout.strip() == "true"

    def _set_profile(self, profile: Any) -> None:
        self.profile = profile
        self.tool_ctx.profile = profile
        self.context.profile = profile
        self.validation.refresh_checks(profile)

    async def ensure_profile(self, refresh: bool = False) -> Any:
        if self.profile is not None and not refresh:
            return self.profile
        from .repo.discovery import discover, save_profile

        profile = await asyncio.to_thread(discover, self.workspace)
        await asyncio.to_thread(save_profile, profile, self.state_dir / "project.json")
        self._set_profile(profile)
        return profile

    async def refresh_index(self) -> dict[str, Any]:
        if self.index is None:
            return {}
        stats = await asyncio.to_thread(self.index.refresh)
        return stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)

    def export_snapshots(self) -> None:
        with contextlib.suppress(Exception):
            self.store.export_tasks_json(self.state_dir / "tasks.json")
        if self.memory is not None:
            with contextlib.suppress(Exception):
                self.memory.write_snapshot(self.state_dir / "context.json", extra={"profile_summary": getattr(self.profile, "summary", "")})
                decisions = [i.model_dump() for i in self.memory.project.list(layer="decision", limit=500)]
                from .core.util import atomic_write_json

                atomic_write_json(self.state_dir / "decisions.json", {"decisions": decisions})

    # ---- tasks ------------------------------------------------------------------------------

    def create_task(self, description: str, *, title: str | None = None, priority: int = 50, queue: bool = False, depends_on: list[str] | None = None) -> Task:
        description = description.strip()
        if not description:
            raise ValueError("task description is empty")
        first = description.splitlines()[0]
        task = Task(
            title=(title or first)[:120],
            description=description,
            priority=priority,
            status=TaskStatus.QUEUED if queue else TaskStatus.PENDING,
            mode=str(self.settings.permissions.mode),
            depends_on=depends_on or [],
        )
        self.store.create_task(task)
        self.bus.emit(EventType.TASK_CREATED, f"created task {task.id}: {task.title}", task_id=task.id)
        self.export_snapshots()
        return task

    async def run_task(self, task_id: str) -> Task:
        return await self.orchestrator.run(task_id)

    def request_stop(self, task_id: str, cancel: bool = False) -> None:
        self.store.request_control(task_id, "cancel" if cancel else "stop")

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self.tool_ctx.processes.stop_all()
        browser = self.tool_ctx.state.get("browser")
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        await self.router.aclose()
        if self.memory is not None:
            with contextlib.suppress(Exception):
                self.memory.close()
        if self.index is not None:
            with contextlib.suppress(Exception):
                self.index.close()
        if self.trace is not None:
            self.trace.close()
        self.store.close()
