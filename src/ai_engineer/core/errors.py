"""Error taxonomy shared across the system."""

from __future__ import annotations

from typing import Any


class AIEngineerError(Exception):
    """Base class for all errors raised by AI Engineer."""


class ConfigError(AIEngineerError):
    """Invalid or missing configuration."""


class StateError(AIEngineerError):
    """Persistent state is missing, corrupt, or inconsistent."""


class CancelledByUser(AIEngineerError):
    """The user (or a stop request) cancelled the operation."""


# --- model / provider errors -------------------------------------------------


class ProviderError(AIEngineerError):
    """A model provider call failed."""

    retryable: bool = False
    # Whether the router should try the next model in the chain after this error.
    fallback: bool = True
    # Whether the error says the model/endpoint itself is unhealthy (counts toward its circuit
    # breaker). Request-specific problems (too long, invalid, refused, unparsable) do not.
    unhealthy: bool = True

    def __init__(self, message: str, *, provider: str = "", model: str = "", status: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status = status

    def __str__(self) -> str:
        where = f"{self.provider}:{self.model}" if self.provider else ""
        status = f" (HTTP {self.status})" if self.status else ""
        base = super().__str__()
        return f"[{where}]{status} {base}" if where else f"{base}{status}"


class RetryableProviderError(ProviderError):
    """Transient failure: timeouts, connection resets, 5xx, overload."""

    retryable = True


class RateLimitError(RetryableProviderError):
    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class ProviderUnavailableError(RetryableProviderError):
    """The provider endpoint cannot be reached."""


class AuthenticationError(ProviderError):
    """Missing or invalid credentials. Not retryable; fall back to the next model."""


class InvalidRequestError(ProviderError):
    """The provider rejected the request as malformed (HTTP 400/404/422)."""

    unhealthy = False


class ContextLengthError(InvalidRequestError):
    """The request exceeded the model's context window."""

    fallback = False


class RefusalError(ProviderError):
    """The model declined the request."""

    unhealthy = False


class CapabilityNotSupported(ProviderError):
    """The provider or model does not support the requested capability."""

    fallback = True
    unhealthy = False


class MalformedOutputError(ProviderError):
    """The model output could not be parsed or validated."""

    unhealthy = False


class AllModelsFailedError(ProviderError):
    """Every model in a fallback chain failed."""

    fallback = False

    def __init__(self, message: str, *, attempts: list[str] | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.attempts = attempts or []


# --- tool errors ---------------------------------------------------------------


class ToolError(AIEngineerError):
    """A tool failed in a way the model should be told about."""


class ToolValidationError(ToolError):
    """Tool arguments did not match the schema."""


class PermissionDenied(ToolError):
    """Policy denied the action."""


class ApprovalDenied(PermissionDenied):
    """A human (or approval policy) declined the action."""


class PathViolation(PermissionDenied):
    """A path escaped the workspace or touched a protected location."""


class ToolTimeout(ToolError):
    """A tool exceeded its timeout."""
