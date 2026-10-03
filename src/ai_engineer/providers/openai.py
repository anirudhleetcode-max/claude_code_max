"""OpenAI Chat Completions adapter.

Also used for any OpenAI-compatible server (vLLM, llama.cpp server, LM Studio,
Ollama's /v1 endpoint, hosted gateways) via ``type = "openai_compatible"`` and
``base_url``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..config.settings import ProviderConfig
from ..core.errors import ProviderError, RetryableProviderError
from ..core.types import StopReason, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from ..models.base import ModelProvider, ModelRequest, ModelResponse, StreamEvent
from ..models.prompted_tools import INVALID_TOOL_CALL
from ._http import HttpProviderMixin, api_key, map_http_error, map_transport_error

DEFAULT_OPENAI_URL = "https://api.openai.com/v1"

_FINISH = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


def to_openai_messages(req: ModelRequest, system_role: str = "system") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if req.system:
        out.append({"role": system_role, "content": req.system})
    for msg in req.messages:
        if msg.role == "assistant":
            text = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(b.input, ensure_ascii=False)},
                }
                for b in msg.content
                if isinstance(b, ToolUseBlock)
            ]
            if calls:
                entry["tool_calls"] = calls
            if entry["content"] is None and not calls:
                entry["content"] = ""
            out.append(entry)
        else:
            # Tool results must directly follow the assistant message that requested them.
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    content = block.content if not block.is_error else f"ERROR: {block.content}"
                    out.append({"role": "tool", "tool_call_id": block.tool_use_id, "content": content})
            text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
            if text:
                out.append({"role": "user", "content": text})
    return out


def to_openai_tools(req: ModelRequest) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in req.tools
    ]


def _parse_tool_call(call: dict[str, Any]) -> ToolUseBlock:
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    raw = fn.get("arguments")
    call_id = call.get("id") or f"call_{abs(hash(json.dumps(call, sort_keys=True))) % 10**12}"
    if isinstance(raw, dict):
        return ToolUseBlock(id=call_id, name=name, input=raw)
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return ToolUseBlock(
            id=call_id, name=INVALID_TOOL_CALL, input={"raw": str(raw)[:2000], "error": f"{name}: {exc}"}
        )
    if not isinstance(args, dict):
        args = {"_raw": args}
    return ToolUseBlock(id=call_id, name=name, input=args)


def from_openai_response(data: dict[str, Any], model: str) -> ModelResponse:
    choices = data.get("choices") or []
    if not choices:
        raise RetryableProviderError("response contained no choices", model=model)
    choice = choices[0]
    message = choice.get("message") or {}
    content: list[Any] = []
    text = message.get("content")
    if isinstance(text, list):  # some servers return content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    if text:
        content.append(TextBlock(text=text))
    for call in message.get("tool_calls") or []:
        content.append(_parse_tool_call(call))
    stop = _FINISH.get(choice.get("finish_reason") or "", StopReason.OTHER)
    refusal = message.get("refusal")
    if refusal:
        stop = StopReason.REFUSAL
    if any(isinstance(b, ToolUseBlock) for b in content):
        stop = StopReason.TOOL_USE
    usage_data = data.get("usage") or {}
    details = usage_data.get("prompt_tokens_details") or {}
    usage = Usage(
        input_tokens=int(usage_data.get("prompt_tokens") or 0),
        output_tokens=int(usage_data.get("completion_tokens") or 0),
        cache_read_tokens=int(details.get("cached_tokens") or 0),
    )
    return ModelResponse(
        content=content,
        stop_reason=stop,
        usage=usage,
        model=data.get("model") or model,
        request_id=str(data.get("id") or ""),
        refusal_detail=str(refusal or ""),
    )


class OpenAIProvider(HttpProviderMixin, ModelProvider):
    provider_type = "openai"

    def __init__(self, name: str, config: ProviderConfig) -> None:
        super().__init__(name, config)
        self.compatible = config.type == "openai_compatible"
        self.base_url = (config.base_url or DEFAULT_OPENAI_URL).rstrip("/")
        default_field = "max_tokens" if self.compatible else "max_completion_tokens"
        self.max_tokens_field: str = config.options.get("max_tokens_field", default_field)
        self.system_role: str = config.options.get("system_role", "system")

    @property
    def supports_embeddings(self) -> bool:
        return True

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        key = api_key(self.config, required=not self.compatible, provider=self.name)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def build_payload(self, req: ModelRequest, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": req.model, "messages": to_openai_messages(req, self.system_role)}
        if req.tools:
            payload["tools"] = to_openai_tools(req)
        if req.max_tokens:
            payload[self.max_tokens_field] = req.max_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.stop:
            payload["stop"] = req.stop
        if req.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": req.response_schema},
            }
        payload.update(self.options_for(req.model).params)
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def _generate(self, req: ModelRequest) -> ModelResponse:
        data = await self._post_json(
            f"{self.base_url}/chat/completions", self.build_payload(req), self._headers(), req.model
        )
        return from_openai_response(data, req.model)

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        opts = self.options_for(req.model)
        if req.tools and not opts.native_tools:
            async for event in super().stream(req):
                yield event
            return
        if req.max_tokens is None:
            req = req.model_copy(update={"max_tokens": opts.max_output_tokens})
        payload = self.build_payload(req, stream=True)
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish = ""
        usage: dict[str, Any] = {}
        try:
            async with self.http().stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise map_http_error(response, self.name, req.model)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if data.get("usage"):
                        usage = data["usage"]
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            text_parts.append(delta["content"])
                            yield StreamEvent(type="text_delta", text=delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(int(tc.get("index", 0)), {"id": "", "name": "", "args": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["args"] += fn.get("arguments") or ""
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise map_transport_error(exc, self.name, req.model) from exc
        message: dict[str, Any] = {"content": "".join(text_parts)}
        if calls:
            message["tool_calls"] = [
                {"id": c["id"], "function": {"name": c["name"], "arguments": c["args"]}} for _, c in sorted(calls.items())
            ]
        resp = from_openai_response(
            {"choices": [{"message": message, "finish_reason": finish}], "usage": usage, "model": req.model}, req.model
        )
        resp.provider = self.name
        yield StreamEvent(type="done", response=resp)

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.config.options.get("embedding_model", "")
        data = await self._post_json(
            f"{self.base_url}/embeddings", {"model": model, "input": texts}, self._headers(), model
        )
        items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        return [list(map(float, item["embedding"])) for item in items]

    async def list_models(self) -> list[str]:
        data = await self._get_json(f"{self.base_url}/models", self._headers())
        return sorted(str(m.get("id")) for m in data.get("data", []) if isinstance(m, dict) and m.get("id"))
