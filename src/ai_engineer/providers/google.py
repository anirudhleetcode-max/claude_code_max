"""Google Gemini (Generative Language REST API) adapter."""

from __future__ import annotations

from typing import Any

from ..config.settings import ProviderConfig
from ..core.ids import short_id
from ..core.types import Message, OpaqueBlock, StopReason, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from ..models.base import ModelProvider, ModelRequest, ModelResponse
from ._http import HttpProviderMixin, api_key
from ._schema import to_openapi_subset

DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"

_REFUSAL_REASONS = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


class GoogleProvider(HttpProviderMixin, ModelProvider):
    type = "google"

    def __init__(self, name: str, config: ProviderConfig) -> None:
        super().__init__(name, config)
        self.base_url = (config.base_url or DEFAULT_GEMINI_URL).rstrip("/")

    @property
    def supports_embeddings(self) -> bool:
        return True

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        key = api_key(self.config, required=True, provider=self.name)
        if key:
            headers["x-goog-api-key"] = key
        return headers

    # ---- translation ----------------------------------------------------------------

    def _assistant_parts(self, msg: Message, model: str) -> list[dict[str, Any]]:
        # Replay the provider's raw parts verbatim when they came from this model:
        # this preserves thought signatures exactly.
        for block in msg.content:
            if isinstance(block, OpaqueBlock) and block.provider == self.name and block.model == model:
                parts = block.payload.get("gemini_parts")
                if isinstance(parts, list) and parts:
                    return parts
        parts: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlock) and block.text:
                parts.append({"text": block.text})
            elif isinstance(block, ToolUseBlock):
                call: dict[str, Any] = {"name": block.name, "args": block.input}
                meta = block.meta.get(self.name) or {}
                if meta.get("id"):
                    call["id"] = meta["id"]
                part: dict[str, Any] = {"functionCall": call}
                if meta.get("thoughtSignature") and meta.get("model") == model:
                    part["thoughtSignature"] = meta["thoughtSignature"]
                parts.append(part)
        return parts

    def to_contents(self, req: ModelRequest) -> list[dict[str, Any]]:
        call_ids: dict[str, str] = {}
        for msg in req.messages:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    native = (block.meta.get(self.name) or {}).get("id")
                    if native:
                        call_ids[block.id] = native
        contents: list[dict[str, Any]] = []
        for msg in req.messages:
            if msg.role == "assistant":
                role, parts = "model", self._assistant_parts(msg, req.model)
            else:
                role, parts = "user", []
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        key = "error" if block.is_error else "result"
                        response: dict[str, Any] = {"name": block.name, "response": {key: block.content}}
                        if block.tool_use_id in call_ids:
                            response["id"] = call_ids[block.tool_use_id]
                        parts.append({"functionResponse": response})
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text:
                        parts.append({"text": block.text})
            if not parts:
                continue
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        return contents

    def build_payload(self, req: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {"contents": self.to_contents(req)}
        if req.system:
            payload["systemInstruction"] = {"parts": [{"text": req.system}]}
        if req.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": t.name, "description": t.description, "parameters": to_openapi_subset(t.input_schema)}
                        for t in req.tools
                    ]
                }
            ]
        gen: dict[str, Any] = {}
        if req.max_tokens:
            gen["maxOutputTokens"] = req.max_tokens
        if req.temperature is not None:
            gen["temperature"] = req.temperature
        if req.stop:
            gen["stopSequences"] = req.stop
        if req.response_schema is not None:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = to_openapi_subset(req.response_schema)
        params = dict(self.options_for(req.model).params)
        gen.update(params.pop("generationConfig", {}))
        if gen:
            payload["generationConfig"] = gen
        payload.update(params)
        return payload

    def from_response(self, data: dict[str, Any], model: str) -> ModelResponse:
        usage_data = data.get("usageMetadata") or {}
        usage = Usage(
            input_tokens=int(usage_data.get("promptTokenCount") or 0),
            output_tokens=int(usage_data.get("candidatesTokenCount") or 0) + int(usage_data.get("thoughtsTokenCount") or 0),
            cache_read_tokens=int(usage_data.get("cachedContentTokenCount") or 0),
        )
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            return ModelResponse(stop_reason=StopReason.REFUSAL, usage=usage, model=model, refusal_detail=str(reason))
        cand = candidates[0]
        raw_parts = (cand.get("content") or {}).get("parts") or []
        content: list[Any] = []
        for part in raw_parts:
            if part.get("thought"):
                continue  # reasoning summaries are never surfaced
            if part.get("text"):
                content.append(TextBlock(text=part["text"]))
            elif "functionCall" in part:
                call = part["functionCall"]
                meta: dict[str, Any] = {"model": model}
                if call.get("id"):
                    meta["id"] = call["id"]
                if part.get("thoughtSignature"):
                    meta["thoughtSignature"] = part["thoughtSignature"]
                content.append(
                    ToolUseBlock(
                        id=f"call_{short_id(12)}", name=call.get("name", ""), input=call.get("args") or {},
                        meta={self.name: meta},
                    )
                )
        if raw_parts:
            content.insert(0, OpaqueBlock(provider=self.name, model=model, payload={"gemini_parts": raw_parts}))
        reason = cand.get("finishReason") or "STOP"
        if any(isinstance(b, ToolUseBlock) for b in content):
            stop = StopReason.TOOL_USE
        elif reason == "MAX_TOKENS":
            stop = StopReason.MAX_TOKENS
        elif reason in _REFUSAL_REASONS:
            stop = StopReason.REFUSAL
        elif reason == "STOP":
            stop = StopReason.END_TURN
        else:
            stop = StopReason.OTHER
        return ModelResponse(
            content=content, stop_reason=stop, usage=usage, model=model,
            refusal_detail=reason if stop == StopReason.REFUSAL else "",
        )

    # ---- API ---------------------------------------------------------------------------

    async def _generate(self, req: ModelRequest) -> ModelResponse:
        url = f"{self.base_url}/models/{req.model}:generateContent"
        data = await self._post_json(url, self.build_payload(req), self._headers(), req.model)
        return self.from_response(data, req.model)

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.config.options.get("embedding_model", "")
        url = f"{self.base_url}/models/{model}:batchEmbedContents"
        body = {"requests": [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in texts]}
        data = await self._post_json(url, body, self._headers(), model)
        return [list(map(float, e.get("values", []))) for e in data.get("embeddings", [])]

    async def list_models(self) -> list[str]:
        data = await self._get_json(f"{self.base_url}/models", self._headers())
        names = [str(m.get("name", "")) for m in data.get("models", []) if isinstance(m, dict)]
        return sorted(n.removeprefix("models/") for n in names if n)
