"""Ollama native REST adapter (local models, works offline)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..config.settings import ProviderConfig
from ..core.errors import ProviderError
from ..core.ids import short_id
from ..core.types import StopReason, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from ..models.base import ModelProvider, ModelRequest, ModelResponse, StreamEvent
from ._http import HttpProviderMixin, map_http_error, map_transport_error

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def to_ollama_messages(req: ModelRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    for msg in req.messages:
        if msg.role == "assistant":
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(b.text for b in msg.content if isinstance(b, TextBlock)),
            }
            calls = [
                {"function": {"name": b.name, "arguments": b.input}} for b in msg.content if isinstance(b, ToolUseBlock)
            ]
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        else:
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    content = block.content if not block.is_error else f"ERROR: {block.content}"
                    out.append({"role": "tool", "content": content, "tool_name": block.name})
            text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
            if text:
                out.append({"role": "user", "content": text})
    return out


def from_ollama_response(data: dict[str, Any], model: str) -> ModelResponse:
    message = data.get("message") or {}
    content: list[Any] = []
    if message.get("content"):
        content.append(TextBlock(text=message["content"]))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        content.append(ToolUseBlock(id=call.get("id") or f"call_{short_id(12)}", name=fn.get("name", ""), input=args))
    if any(isinstance(b, ToolUseBlock) for b in content):
        stop = StopReason.TOOL_USE
    elif data.get("done_reason") == "length":
        stop = StopReason.MAX_TOKENS
    else:
        stop = StopReason.END_TURN
    usage = Usage(input_tokens=int(data.get("prompt_eval_count") or 0), output_tokens=int(data.get("eval_count") or 0))
    return ModelResponse(content=content, stop_reason=stop, usage=usage, model=data.get("model") or model)


class OllamaProvider(HttpProviderMixin, ModelProvider):
    type = "ollama"

    def __init__(self, name: str, config: ProviderConfig) -> None:
        super().__init__(name, config)
        self.base_url = (config.base_url or DEFAULT_OLLAMA_URL).rstrip("/")

    @property
    def supports_embeddings(self) -> bool:
        return True

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", **self.config.headers}

    def build_payload(self, req: ModelRequest, stream: bool = False) -> dict[str, Any]:
        options: dict[str, Any] = {}
        params = dict(self.options_for(req.model).params)
        if req.max_tokens:
            options["num_predict"] = req.max_tokens
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.stop:
            options["stop"] = req.stop
        options.update(params.pop("options", {}))
        for key in ("num_ctx", "top_p", "top_k", "seed", "repeat_penalty"):
            if key in params:
                options[key] = params.pop(key)
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": to_ollama_messages(req),
            "stream": stream,
            "options": options,
        }
        if req.tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in req.tools
            ]
        if req.response_schema is not None:
            payload["format"] = req.response_schema
        payload.update(params)
        return payload

    async def _generate(self, req: ModelRequest) -> ModelResponse:
        data = await self._post_json(f"{self.base_url}/api/chat", self.build_payload(req), self._headers(), req.model)
        if data.get("error"):
            raise ProviderError(str(data["error"]), provider=self.name, model=req.model)
        return from_ollama_response(data, req.model)

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        opts = self.options_for(req.model)
        if req.tools and not opts.native_tools:
            async for event in super().stream(req):
                yield event
            return
        if req.max_tokens is None:
            req = req.model_copy(update={"max_tokens": opts.max_output_tokens})
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        try:
            async with self.http().stream(
                "POST", f"{self.base_url}/api/chat", json=self.build_payload(req, stream=True), headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise map_http_error(response, self.name, req.model)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message") or {}
                    if msg.get("content"):
                        text_parts.append(msg["content"])
                        yield StreamEvent(type="text_delta", text=msg["content"])
                    tool_calls.extend(msg.get("tool_calls") or [])
                    if chunk.get("done"):
                        final = chunk
        except ProviderError:
            raise
        except Exception as exc:
            raise map_transport_error(exc, self.name, req.model) from exc
        final = {**final, "message": {"content": "".join(text_parts), "tool_calls": tool_calls}}
        resp = from_ollama_response(final, req.model)
        resp.provider = self.name
        yield StreamEvent(type="done", response=resp)

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.config.options.get("embedding_model", "")
        data = await self._post_json(f"{self.base_url}/api/embed", {"model": model, "input": texts}, self._headers(), model)
        return [list(map(float, v)) for v in data.get("embeddings", [])]

    async def list_models(self) -> list[str]:
        data = await self._get_json(f"{self.base_url}/api/tags", self._headers())
        return sorted(str(m.get("name")) for m in data.get("models", []) if isinstance(m, dict) and m.get("name"))
