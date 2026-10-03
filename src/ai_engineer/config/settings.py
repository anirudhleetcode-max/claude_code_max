"""Typed configuration.

Precedence (lowest to highest): built-in defaults → global config file →
project ``.agent/config.toml`` → environment variables → explicit overrides
(e.g. CLI flags). Credentials are never stored in config files: providers name
the environment variable that holds their key (``api_key_env``).
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PermissionLevel(IntEnum):
    READ_ONLY = 0
    SAFE_WRITE = 1
    DEVELOPMENT = 2
    PRIVILEGED = 3

    @classmethod
    def parse(cls, value: str | int | PermissionLevel) -> PermissionLevel:
        if isinstance(value, PermissionLevel):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper()]


class Mode(StrEnum):
    SAFE = "safe"
    ASSISTED = "assisted"
    DEVELOPER = "developer"
    AUTONOMOUS = "autonomous"


class GateMode(StrEnum):
    REQUIRED = "required"
    IF_AVAILABLE = "if_available"
    DISABLED = "disabled"


# --- models ----------------------------------------------------------------------


class ModelOptions(_Section):
    """Per-model tuning. Unset values are not sent to the provider."""

    max_output_tokens: int = 16000
    context_window: int = 128000
    temperature: float | None = None
    native_tools: bool = True
    structured_output: Literal["prompt", "native"] = "prompt"
    # Provider-specific knobs, passed through by the adapter that understands them
    # (e.g. Anthropic: {"effort": "high"}; Ollama: {"num_ctx": 32768}).
    params: dict[str, Any] = Field(default_factory=dict)


class ProviderConfig(_Section):
    type: Literal["anthropic", "openai", "openai_compatible", "google", "ollama", "scripted"]
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 600.0
    headers: dict[str, str] = Field(default_factory=dict)
    # Default model options for every model of this provider, overridable per model.
    defaults: ModelOptions = Field(default_factory=ModelOptions)
    models: dict[str, ModelOptions] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    def model_options(self, model: str) -> ModelOptions:
        if model in self.models:
            merged = self.defaults.model_dump()
            merged.update(self.models[model].model_dump(exclude_unset=True))
            return ModelOptions(**merged)
        return self.defaults


class RetryConfig(_Section):
    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay_s: float = Field(default=1.0, ge=0)
    max_delay_s: float = Field(default=30.0, ge=0)
    jitter: float = Field(default=0.25, ge=0, le=1)


class CircuitBreakerConfig(_Section):
    failure_threshold: int = Field(default=3, ge=1)
    reset_after_s: float = Field(default=60.0, ge=0)


ROLE_NAMES = (
    "default",
    "planner",
    "coder",
    "debugger",
    "reviewer",
    "fast",
    "classifier",
    "summarizer",
    "embeddings",
)


class ModelsSettings(_Section):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    # role -> ordered fallback chain of "provider:model" references
    roles: dict[str, list[str]] = Field(default_factory=dict)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    request_timeout_s: float = 600.0

    @field_validator("roles")
    @classmethod
    def _check_refs(cls, roles: dict[str, list[str]]) -> dict[str, list[str]]:
        for role, chain in roles.items():
            if isinstance(chain, str):
                raise ValueError(f"models.roles.{role} must be a list of 'provider:model' strings")
            for ref in chain:
                if ":" not in ref:
                    raise ValueError(f"models.roles.{role}: '{ref}' must look like 'provider:model'")
        return roles

    def chain_for(self, role: str) -> list[str]:
        if self.roles.get(role):
            return list(self.roles[role])
        fallbacks = {
            "debugger": "coder",
            "reviewer": "coder",
            "planner": "coder",
            "summarizer": "fast",
            "classifier": "fast",
            "fast": "default",
            "coder": "default",
        }
        nxt = fallbacks.get(role)
        if nxt:
            return self.chain_for(nxt)
        return list(self.roles.get("default", []))


# --- safety ------------------------------------------------------------------------


class PermissionsSettings(_Section):
    mode: Mode = Mode.DEVELOPER
    # Hard ceiling regardless of mode. PRIVILEGED actions always require approval
    # (or an explicit allow_commands match); set DEVELOPMENT to deny them outright.
    max_level: PermissionLevel = PermissionLevel.PRIVILEGED
    # fnmatch-style patterns matched against the full command string.
    allow_commands: list[str] = Field(default_factory=list)
    deny_commands: list[str] = Field(default_factory=list)
    # Workspace-relative glob patterns that tools may never write (matched case-insensitively).
    # "**/.git/**" and ".git" also cover nested repositories, submodules and worktree gitfiles.
    protected_paths: list[str] = Field(
        default_factory=lambda: [
            "**/.git/**",
            ".git",
            ".agent/**",
            ".env",
            ".env.*",
            "**/*.pem",
            "**/*.key",
            "**/id_rsa*",
            "**/id_ed25519*",
            "**/.ssh/**",
        ]
    )
    # Patterns of files whose *contents* are redacted when read (matched case-insensitively).
    secret_files: list[str] = Field(
        default_factory=lambda: [".env", ".env.*", "**/*.pem", "**/*.key", "**/id_rsa*", "**/credentials*"]
    )
    approval_timeout_s: float = 900.0

    @field_validator("max_level", mode="before")
    @classmethod
    def _parse_level(cls, v: Any) -> PermissionLevel:
        return PermissionLevel.parse(v)


class TerminalSettings(_Section):
    shell: str | None = None
    default_timeout_s: float = 300.0
    max_timeout_s: float = 3600.0
    max_output_chars: int = 30000
    strip_secret_env: bool = True
    env_allow: list[str] = Field(default_factory=list)
    sandbox: Literal["none", "docker"] = "none"
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"


# --- agent behaviour -----------------------------------------------------------------


class AgentSettings(_Section):
    max_steps: int = Field(default=60, ge=1)
    max_repair_iterations: int = Field(default=4, ge=0)
    max_review_iterations: int = Field(default=2, ge=0)
    max_subtasks: int = Field(default=12, ge=1)
    task_time_budget_s: float = 4 * 3600.0
    on_questions: Literal["ask", "assume", "block"] = "ask"
    stop_on_subtask_failure: bool = False
    # Fraction of the model context window the agent may fill before resetting
    # the conversation with a hand-off summary.
    context_fill_ratio: float = Field(default=0.6, gt=0.1, le=0.95)
    tool_result_max_chars: int = 20000
    repeated_call_limit: int = 3


class GatesSettings(_Section):
    requirements: GateMode = GateMode.REQUIRED
    implementation: GateMode = GateMode.REQUIRED
    typecheck: GateMode = GateMode.IF_AVAILABLE
    lint: GateMode = GateMode.IF_AVAILABLE
    tests: GateMode = GateMode.REQUIRED
    build: GateMode = GateMode.IF_AVAILABLE
    security: GateMode = GateMode.REQUIRED
    review: GateMode = GateMode.REQUIRED
    docs: GateMode = GateMode.IF_AVAILABLE
    git_state: GateMode = GateMode.REQUIRED


class GitSettings(_Section):
    checkpoints: bool = True
    # "if_clean": create a task branch and commit verified subtasks only when the
    # working tree was clean when the task started; "never": leave changes uncommitted.
    auto_commit: Literal["never", "if_clean"] = "if_clean"
    branch_prefix: str = "aie/"
    commit_prefix: str = "aie: "
    author_name: str | None = None
    author_email: str | None = None


class ValidationSettings(_Section):
    # Explicit overrides for detected commands.
    test_command: str | None = None
    lint_command: str | None = None
    format_command: str | None = None
    typecheck_command: str | None = None
    build_command: str | None = None
    test_timeout_s: float = 1200.0
    # Run the checks once before any change so pre-existing failures are not
    # attributed to (or hidden by) the agent.
    baseline: bool = True
    # After targeted tests pass, also run the full suite at the end of each subtask.
    full_suite_per_subtask: bool = True
    run_full_suite_at_end: bool = True
    dependency_audit: bool = True


class MemorySettings(_Section):
    global_dir: str | None = None
    max_items_in_context: int = 8


class WebSettings(_Section):
    search_backend: Literal["none", "searxng", "brave", "tavily"] = "none"
    searxng_url: str | None = None
    api_key_env: str | None = None
    fetch_max_bytes: int = 2_000_000
    allow_domains: list[str] = Field(default_factory=list)
    block_domains: list[str] = Field(default_factory=list)
    timeout_s: float = 30.0
    # Use an existing Chrome/Chromium instead of Playwright's bundled browser
    # (also settable with AIE_BROWSER_EXECUTABLE).
    browser_executable: str | None = None


class UISettings(_Section):
    host: str = "127.0.0.1"
    port: int = 8765


class ObservabilitySettings(_Section):
    debug: bool = False
    trace: bool = True
    log_level: str = "WARNING"


class Settings(_Section):
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    permissions: PermissionsSettings = Field(default_factory=PermissionsSettings)
    terminal: TerminalSettings = Field(default_factory=TerminalSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    gates: GatesSettings = Field(default_factory=GatesSettings)
    git: GitSettings = Field(default_factory=GitSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    web: WebSettings = Field(default_factory=WebSettings)
    ui: UISettings = Field(default_factory=UISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
