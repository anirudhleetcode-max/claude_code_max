"""The provider-independent model interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from ..config.settings import ModelOptions, ProviderConfig
from ..core.errors import CapabilityNotSupported
from ..core.types import (
    Block,
    Message,
    StopReason,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
    message_char_size,
)
from ..core.util import estimate_tokens

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    system: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] = Field(default_factory=list)
    # JSON schema for native structured output (only used when the model is
    # configured with structured_output = "native").
    response_schema: dict[str, Any] | None = None
    # Not sent to providers: used for routing, scripted providers and tracing.
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float | None = None


class ModelResponse(BaseModel):
    content: list[Block] = Field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    latency_s: float = 0.0
    request_id: str = ""
    refusal_detail: str = ""

    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def to_message(self) -> Message:
        return Message(role="assistant", content=list(self.content))


class StreamEvent(BaseModel):
    type: Literal["text_delta", "done"]
    text: str = ""
    response: ModelResponse | None = None


class HealthStatus(BaseModel):
    ok: bool
    provider: str
    detail: str = ""
    latency_s: float = 0.0
    models: list[str] = Field(default_factory=list)


class StructuredResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    value: Any
    response: ModelResponse
    repairs: int = 0


class ModelProvider(ABC):
    """Base class for model providers.

    Subclasses implement :meth:`_generate` (and optionally :meth:`_stream`,
    :meth:`embeddings`, :meth:`list_models`). The public :meth:`generate`
    adds prompted tool-calling for models without native tool support, timing,
    and token estimation when the provider does not report usage.
    """

    type: str = "base"

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config

    # ---- capabilities -----------------------------------------------------------

    def options_for(self, model: str) -> ModelOptions:
        return self.config.model_options(model)

    @property
    def supports_embeddings(self) -> bool:
        return False

    # ---- public API -----------------------------------------------------------------

    async def generate(self, req: ModelRequest) -> ModelResponse:
        from .prompted_tools import adapt_request, parse_response  # local import avoids a cycle

        opts = self.options_for(req.model)
        use_prompted = bool(req.tools) and not opts.native_tools
        send = adapt_request(req) if use_prompted else req
        if send.max_tokens is None:
            send = send.model_copy(update={"max_tokens": opts.max_output_tokens})
        if send.temperature is None and opts.temperature is not None:
            send = send.model_copy(update={"temperature": opts.temperature})
        started = time.monotonic()
        resp = await self._generate(send)
        resp.latency_s = time.monotonic() - started
        resp.provider = resp.provider or self.name
        resp.model = resp.model or req.model
        if use_prompted:
            resp = parse_response(resp, {t.name for t in req.tools})
        if resp.usage.input_tokens == 0 and resp.usage.output_tokens == 0:
            resp.usage = estimate_usage(send, resp)
        return resp

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream text deltas, then a final ``done`` event carrying the full response."""
        resp = await self.generate(req)
        text = resp.text()
        if text:
            yield StreamEvent(type="text_delta", text=text)
        yield StreamEvent(type="done", response=resp)

    async def tool_call(self, req: ModelRequest) -> ModelResponse:
        if not req.tools:
            raise ValueError("tool_call() requires at least one tool")
        return await self.generate(req)

    async def structured_output(
        self, req: ModelRequest, schema: type[SchemaT], max_repairs: int = 2
    ) -> StructuredResult:
        from .structured import structured_generate

        return await structured_generate(self.generate, req, schema, self.options_for(req.model), max_repairs)

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise CapabilityNotSupported("embeddings are not supported", provider=self.name, model=model or "")

    async def list_models(self) -> list[str]:
        raise CapabilityNotSupported("model listing is not supported", provider=self.name)

    async def health_check(self) -> HealthStatus:
        started = time.monotonic()
        try:
            models = await self.list_models()
            return HealthStatus(
                ok=True, provider=self.name, detail="reachable", latency_s=time.monotonic() - started, models=models
            )
        except CapabilityNotSupported:
            return HealthStatus(ok=True, provider=self.name, detail="no health endpoint; not verified")
        except Exception as exc:
            return HealthStatus(ok=False, provider=self.name, detail=str(exc), latency_s=time.monotonic() - started)

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""

    # ---- for subclasses -------------------------------------------------------------------

    @abstractmethod
    async def _generate(self, req: ModelRequest) -> ModelResponse:
        """Perform one non-streaming model call."""


def estimate_usage(req: ModelRequest, resp: ModelResponse) -> Usage:
    in_chars = len(req.system) + sum(message_char_size(m) for m in req.messages)
    in_chars += sum(len(t.description) + len(str(t.input_schema)) for t in req.tools)
    out_chars = message_char_size(resp.to_message())
    return Usage(input_tokens=estimate_tokens("x" * in_chars), output_tokens=estimate_tokens("x" * out_chars), estimated=True)


def request_tokens(req: ModelRequest) -> int:
    """Estimated prompt size of a request in tokens."""
    chars = len(req.system) + sum(message_char_size(m) for m in req.messages)
    chars += sum(len(t.description) + len(str(t.input_schema)) for t in req.tools)
    return max(1, chars // 4)
