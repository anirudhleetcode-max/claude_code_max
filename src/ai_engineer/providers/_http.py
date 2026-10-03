"""HTTP helpers shared by REST-based providers."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ..config.settings import ProviderConfig
from ..core.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    RetryableProviderError,
)

_CONTEXT_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "prompt is too long",
    "too many tokens",
    "input is too long",
    "exceeds the maximum number of tokens",
)


def api_key(cfg: ProviderConfig, required: bool, provider: str) -> str | None:
    if not cfg.api_key_env:
        if required:
            raise AuthenticationError(
                "no api_key_env configured for this provider", provider=provider
            )
        return None
    value = os.environ.get(cfg.api_key_env)
    if not value and required:
        raise AuthenticationError(
            f"environment variable {cfg.api_key_env} is not set", provider=provider
        )
    return value or None


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:1000]
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("message") or err.get("status") or err)[:1000]
        return str(err)[:1000]
    return str(body)[:1000]


def map_http_error(response: httpx.Response, provider: str, model: str) -> ProviderError:
    status = response.status_code
    text = _error_text(response)
    kw: dict[str, Any] = {"provider": provider, "model": model, "status": status}
    if status in (401, 403):
        return AuthenticationError(text or "authentication failed", **kw)
    if status == 429:
        return RateLimitError(text or "rate limited", retry_after=parse_retry_after(response.headers.get("retry-after")), **kw)
    if status in (408, 409, 425) or status >= 500:
        return RetryableProviderError(text or f"server error {status}", **kw)
    lowered = text.lower()
    if any(marker in lowered for marker in _CONTEXT_MARKERS):
        return ContextLengthError(text, **kw)
    return InvalidRequestError(text or f"request rejected ({status})", **kw)


def map_transport_error(exc: Exception, provider: str, model: str) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return RetryableProviderError(f"timeout: {exc}", provider=provider, model=model)
    if isinstance(exc, httpx.ConnectError):
        return ProviderUnavailableError(f"cannot connect: {exc}", provider=provider, model=model)
    if isinstance(exc, httpx.HTTPError):
        return RetryableProviderError(f"transport error: {exc}", provider=provider, model=model)
    return ProviderError(f"unexpected error: {exc}", provider=provider, model=model)


class HttpProviderMixin:
    """Lazily-created ``httpx.AsyncClient`` with an injectable transport for tests."""

    name: str
    config: ProviderConfig
    _client: httpx.AsyncClient | None = None
    _transport: httpx.AsyncBaseTransport | None = None

    def set_transport(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport
        self._client = None

    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_s, connect=15.0),
                transport=self._transport,
            )
        return self._client

    async def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str], model: str) -> dict[str, Any]:
        try:
            response = await self.http().post(url, json=payload, headers=headers)
        except Exception as exc:
            raise map_transport_error(exc, self.name, model) from exc
        if response.status_code >= 400:
            raise map_http_error(response, self.name, model)
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise RetryableProviderError(
                f"invalid JSON from provider: {response.text[:200]}", provider=self.name, model=model
            ) from exc
        if not isinstance(data, dict):
            raise RetryableProviderError("unexpected response shape", provider=self.name, model=model)
        return data

    async def _get_json(self, url: str, headers: dict[str, str]) -> dict[str, Any]:
        try:
            response = await self.http().get(url, headers=headers)
        except Exception as exc:
            raise map_transport_error(exc, self.name, "") from exc
        if response.status_code >= 400:
            raise map_http_error(response, self.name, "")
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
