"""Tool abstraction."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..config.settings import PermissionLevel, Settings
from ..core.cancel import CancellationToken
from ..core.events import EventBus
from ..core.types import ToolSpec
from ..security.command_risk import Risk
from ..security.paths import PathGuard
from ..security.secrets import Redactor

if TYPE_CHECKING:
    from .file_state import FileStateTracker
    from .process import ProcessManager


class SideEffect(StrEnum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"


class ToolInput(BaseModel):
    """Base class for tool argument models: unknown arguments are rejected."""

    model_config = ConfigDict(extra="forbid")


class ToolResult(BaseModel):
    ok: bool = True
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    truncated: bool = False
    duration_s: float = 0.0

    @classmethod
    def fail(cls, message: str, **data: Any) -> ToolResult:
        return cls(ok=False, content=message, error=message, data=data)


@dataclass
class ActionAssessment:
    """What a tool call would do, for the permission policy and approval prompts."""

    level: PermissionLevel
    summary: str
    risk: Risk | None = None
    read_only: bool = True
    command: str | None = None
    paths: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """Everything a tool may use. Optional services are attached by the runtime."""

    workspace: Path
    settings: Settings
    guard: PathGuard
    redactor: Redactor
    bus: EventBus
    cancel: CancellationToken
    files: FileStateTracker
    processes: ProcessManager
    task_id: str = ""
    subtask_id: str = ""
    # Optional services (typed as Any to keep this module dependency-free).
    repo_index: Any = None
    memory: Any = None
    validation: Any = None
    git: Any = None
    profile: Any = None
    state: dict[str, Any] = field(default_factory=dict)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic noise (titles) that wastes tokens; keep everything semantic."""
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node.pop("title", None)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        return node

    walk(schema)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Input: ClassVar[type[ToolInput]]
    level: ClassVar[PermissionLevel] = PermissionLevel.READ_ONLY
    side_effect: ClassVar[SideEffect] = SideEffect.READ
    timeout_s: ClassVar[float] = 60.0
    # Tools that only make sense in some stages can be filtered by tag.
    tags: ClassVar[frozenset[str]] = frozenset()

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description.strip(), input_schema=_clean_schema(self.Input.model_json_schema()))

    def summarize(self, args: ToolInput) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in args.model_dump(exclude_defaults=True).items())
        return f"{self.name}({fields[:200]})"

    def assess(self, args: ToolInput, ctx: ToolContext) -> ActionAssessment:
        return ActionAssessment(
            level=self.level,
            summary=self.summarize(args),
            read_only=self.side_effect in (SideEffect.NONE, SideEffect.READ),
        )

    def effective_timeout(self, args: ToolInput, ctx: ToolContext) -> float:
        return self.timeout_s

    @property
    def concurrency_safe(self) -> bool:
        return self.side_effect in (SideEffect.NONE, SideEffect.READ)

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> ToolResult:
        """Execute the tool. Raise ToolError for expected failures."""
