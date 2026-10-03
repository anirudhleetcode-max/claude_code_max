"""Anthropic Messages API adapter, built on the official ``anthropic`` SDK.

Install with ``pip install 'ai-engineer[anthropic]'``.

Behaviour:
- Reasoning blocks returned by the API are kept as :class:`OpaqueBlock` and
  echoed back unchanged on later turns to the same model (conversations are
  append-only, so they stay valid).
- ``tool_choice`` is never forced; the agent steers tool use through prompts.
- Sampling parameters are only sent when explicitly configured.
- Optional server-side refusal fallback (``options.server_side_fallback``);
  automatically disabled for a model that rejects it.
- Long requests use streaming internally to avoid HTTP timeouts.
"""

from __future__ import annotations

import os
from typing import Any

from ..config.settings import ProviderConfig
from ..core.errors import (
    AuthenticationError,
    CapabilityNotSupported,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    RetryableProviderError,
)
from ..core.types import OpaqueBlock, StopReason, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from ..models.base import ModelProvider, ModelRequest, ModelResponse
from ._http import parse_retry_after

FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "pause_turn": StopReason.PAUSE,
    "refusal": StopReason.REFUSAL,
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
}


def _import_sdk() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise CapabilityNotSupported(
            "the 'anthropic' package is not installed; run: pip install 'ai-engineer[anthropic]'",
            provider="anthropic",
        ) from exc
    return anthropic


class AnthropicProvider(ModelProvider):
    type = "anthropic"

    def __init__(self, name: str, config: ProviderConfig, http_client: Any = None) -> None:
        super().__init__(name, config)
        self._client: Any = None
        self._http_client = http_client
        self.server_side_fallback: bool = bool(config.options.get("server_side_fallback", False))
        self.prompt_caching: bool = bool(config.options.get("prompt_caching", True))
        self.stream_threshold: int = int(config.options.get("stream_threshold_tokens", 16000))
        self._fallback_rejected: set[str] = set()

    # ---- client ---------------------------------------------------------------------

    def client(self) -> Any:
        if self._client is None:
            sdk = _import_sdk()
            key_env = self.config.api_key_env or "ANTHROPIC_API_KEY"
            key = os.environ.get(key_env)
            if not key:
                raise AuthenticationError(f"environment variable {key_env} is not set", provider=self.name)
            kwargs: dict[str, Any] = {
                "api_key": key,
                "max_retries": 0,  # retries are handled uniformly by the router
                "timeout": self.config.timeout_s,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            if self.config.headers:
                kwargs["default_headers"] = dict(self.config.headers)
            if self._http_client is not None:
                kwargs["http_client"] = self._http_client
            self._client = sdk.AsyncAnthropic(**kwargs)
        return self._client

    # ---- translation -------------------------------------------------------------------

    def to_messages(self, req: ModelRequest) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in req.messages:
            blocks: list[dict[str, Any]] = []
            if msg.role == "assistant":
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        if block.text:
                            blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUseBlock):
                        blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                    elif isinstance(block, OpaqueBlock):
                        if block.provider == self.name and block.model == req.model:
                            blocks.append(dict(block.payload))
            else:
                # tool_result blocks must precede any text in the user turn
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id,
                                "content": block.content or "(no output)",
                                "is_error": block.is_error,
                            }
                        )
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text:
                        blocks.append({"type": "text", "text": block.text})
            if not blocks:
                blocks.append({"type": "text", "text": "(empty)"})
            if out and out[-1]["role"] == msg.role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": msg.role, "content": blocks})
        return out

    def build_params(self, req: ModelRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens or self.options_for(req.model).max_output_tokens,
            "messages": self.to_messages(req),
        }
        if req.system:
            params["system"] = req.system
        if req.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in req.tools
            ]
        if req.stop:
            params["stop_sequences"] = req.stop
        if self.prompt_caching:
            params["cache_control"] = {"type": "ephemeral"}
        extra = dict(self.options_for(req.model).params)
        output_config = dict(extra.pop("output_config", {}))
        if "effort" in extra:
            output_config["effort"] = extra.pop("effort")
        if req.response_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": req.response_schema}
        if output_config:
            params["output_config"] = output_config
        if "thinking" in extra:
            params["thinking"] = extra.pop("thinking")
        extra_body = dict(extra.pop("extra_body", {}))
        if req.temperature is not None:
            extra_body["temperature"] = req.temperature
        extra_body.update(extra)
        if extra_body:
            params["extra_body"] = extra_body
        return params

    def from_message(self, message: Any, model: str) -> ModelResponse:
        content: list[Any] = []
        for block in message.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                content.append(TextBlock(text=block.text))
            elif btype == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input or {})))
            else:
                # thinking, redacted_thinking, fallback, server-tool blocks: replay verbatim
                payload = block.model_dump(mode="json", exclude_none=True) if hasattr(block, "model_dump") else dict(block)
                content.append(OpaqueBlock(provider=self.name, model=model, payload=payload))
        stop = _STOP.get(str(message.stop_reason or ""), StopReason.OTHER)
        refusal_detail = ""
        details = getattr(message, "stop_details", None)
        if stop == StopReason.REFUSAL and details is not None:
            refusal_detail = f"{getattr(details, 'category', '') or ''} {getattr(details, 'explanation', '') or ''}".strip()
        u = message.usage
        usage = Usage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        )
        return ModelResponse(
            content=content,
            stop_reason=stop,
            usage=usage,
            model=getattr(message, "model", model) or model,
            request_id=str(getattr(message, "_request_id", "") or getattr(message, "id", "")),
            refusal_detail=refusal_detail,
        )

    def map_error(self, exc: Exception, model: str) -> ProviderError:
        sdk = _import_sdk()
        kw: dict[str, Any] = {"provider": self.name, "model": model}
        status = getattr(exc, "status_code", None)
        if status is not None:
            kw["status"] = status
        message = str(getattr(exc, "message", "") or exc)
        if isinstance(exc, sdk.APITimeoutError):
            return RetryableProviderError(f"timeout: {message}", **kw)
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderUnavailableError(f"connection error: {message}", **kw)
        if isinstance(exc, sdk.AuthenticationError | sdk.PermissionDeniedError):
            return AuthenticationError(message, **kw)
        if isinstance(exc, sdk.RateLimitError):
            headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
            return RateLimitError(message, retry_after=parse_retry_after(headers.get("retry-after")), **kw)
        if isinstance(exc, sdk.APIStatusError):
            if status is not None and (status >= 500 or status in (408, 409, 529)):
                return RetryableProviderError(message, **kw)
            lowered = message.lower()
            if "prompt is too long" in lowered or "context window" in lowered or "too many tokens" in lowered:
                return ContextLengthError(message, **kw)
            return InvalidRequestError(message, **kw)
        return ProviderError(f"unexpected error: {exc}", **kw)

    # ---- API ------------------------------------------------------------------------------

    async def _call(self, params: dict[str, Any], use_fallback: bool) -> Any:
        client = self.client()
        streaming = int(params.get("max_tokens", 0)) > self.stream_threshold
        if use_fallback:
            api = client.beta.messages
            params = {**params, "betas": [FALLBACK_BETA], "fallbacks": "default"}
        else:
            api = client.messages
        if streaming:
            async with api.stream(**params) as stream:
                return await stream.get_final_message()
        return await api.create(**params)

    async def _generate(self, req: ModelRequest) -> ModelResponse:
        params = self.build_params(req)
        use_fallback = self.server_side_fallback and req.model not in self._fallback_rejected
        sdk = _import_sdk()
        try:
            try:
                message = await self._call(params, use_fallback)
            except sdk.BadRequestError as exc:
                if use_fallback and "fallback" in str(exc).lower():
                    # This model does not accept server-side fallbacks: remember and retry without.
                    self._fallback_rejected.add(req.model)
                    message = await self._call(params, False)
                else:
                    raise
        except ProviderError:
            raise
        except Exception as exc:
            raise self.map_error(exc, req.model) from exc
        return self.from_message(message, req.model)

    async def list_models(self) -> list[str]:
        try:
            page = self.client().models.list(limit=100)
            ids = [m.id async for m in page]
        except ProviderError:
            raise
        except Exception as exc:
            raise self.map_error(exc, "") from exc
        return sorted(ids)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None
