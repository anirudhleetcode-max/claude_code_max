"""A deterministic provider driven by a script.

Used for tests, offline demos and harness benchmarks. It returns pre-defined
responses (optionally chosen per agent role) and records every request it
receives. It has no intelligence of its own and must never be presented as a
real model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
from ..core.ids import short_id
from ..core.types import StopReason, TextBlock, ToolUseBlock, Usage
from ..models.base import ModelProvider, ModelRequest, ModelResponse

Step = Any  # ModelResponse | dict | Exception | Callable[[ModelRequest], Step]

_ERRORS: dict[str, type[ProviderError]] = {
    "rate_limit": RateLimitError,
    "unavailable": ProviderUnavailableError,
    "retryable": RetryableProviderError,
    "auth": AuthenticationError,
    "invalid": InvalidRequestError,
    "context": ContextLengthError,
    "error": ProviderError,
}


def response_from_dict(data: dict[str, Any], model: str = "scripted") -> ModelResponse:
    content: list[Any] = []
    if data.get("text"):
        content.append(TextBlock(text=data["text"]))
    for call in data.get("tool_calls", []):
        content.append(
            ToolUseBlock(id=call.get("id") or f"call_{short_id(12)}", name=call["name"], input=call.get("input", {}))
        )
    stop = data.get("stop_reason")
    if stop is None:
        stop = StopReason.TOOL_USE if data.get("tool_calls") else StopReason.END_TURN
    usage = Usage(**data["usage"]) if data.get("usage") else Usage()
    return ModelResponse(content=content, stop_reason=StopReason(stop), usage=usage, model=model)


class ScriptedProvider(ModelProvider):
    type = "scripted"

    def __init__(
        self,
        name: str = "scripted",
        config: ProviderConfig | None = None,
        steps: list[Step] | None = None,
        by_role: dict[str, list[Step]] | None = None,
        responder: Callable[[ModelRequest], Step] | None = None,
        embed_dim: int = 64,
        models: list[str] | None = None,
    ) -> None:
        super().__init__(name, config or ProviderConfig(type="scripted"))
        self.steps: list[Step] = list(steps or [])
        self.by_role: dict[str, list[Step]] = {k: list(v) for k, v in (by_role or {}).items()}
        self.responder = responder
        self.embed_dim = embed_dim
        self.models = models or ["scripted"]
        self.requests: list[ModelRequest] = []

    @classmethod
    def from_config(cls, name: str, cfg: ProviderConfig) -> ScriptedProvider:
        steps: list[Step] = []
        by_role: dict[str, list[Step]] = {}
        script_file = cfg.options.get("script_file")
        if script_file:
            data = json.loads(Path(script_file).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                by_role = data.get("by_role", {})
                steps = data.get("steps", [])
            else:
                steps = data
        return cls(name, cfg, steps=steps, by_role=by_role, models=cfg.options.get("models"))

    @property
    def supports_embeddings(self) -> bool:
        return True

    def remaining(self) -> int:
        return len(self.steps) + sum(len(v) for v in self.by_role.values())

    async def _resolve(self, step: Step, req: ModelRequest) -> ModelResponse:
        while callable(step) and not isinstance(step, ModelResponse | type):
            step = step(req)
            if asyncio.iscoroutine(step):
                step = await step
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, ModelResponse):
            return step.model_copy(deep=True)
        if isinstance(step, dict):
            if step.get("delay_s"):
                await asyncio.sleep(float(step["delay_s"]))
            if "raise" in step:
                spec = step["raise"]
                err_cls = _ERRORS.get(spec.get("type", "error"), ProviderError)
                raise err_cls(spec.get("message", "scripted failure"), provider=self.name, model=req.model)
            return response_from_dict(step, req.model)
        if isinstance(step, str):
            return response_from_dict({"text": step}, req.model)
        raise ProviderError(f"invalid scripted step: {step!r}", provider=self.name, model=req.model)

    async def _generate(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        role = str(req.metadata.get("role", ""))
        if self.responder is not None:
            step: Step = self.responder(req)
        elif self.by_role.get(role):
            step = self.by_role[role].pop(0)
        elif self.steps:
            step = self.steps.pop(0)
        else:
            raise ProviderError(
                f"scripted provider has no response left (role={role or '-'})", provider=self.name, model=req.model
            )
        resp = await self._resolve(step, req)
        resp.provider = self.name
        return resp

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [hash_embedding(t, self.embed_dim) for t in texts]

    async def list_models(self) -> list[str]:
        return list(self.models)


def hash_embedding(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words feature hashing. Lexical, not semantic."""
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode(), usedforsecurity=False).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
