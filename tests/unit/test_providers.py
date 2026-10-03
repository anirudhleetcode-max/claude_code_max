"""Provider adapters against mocked HTTP transports.

These verify request construction and response parsing against the documented
wire formats. They do not prove compatibility with the live services.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ai_engineer.config.settings import ModelOptions, ProviderConfig
from ai_engineer.core.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitError,
    RetryableProviderError,
)
from ai_engineer.core.types import Message, OpaqueBlock, StopReason, ToolResultBlock, ToolSpec, ToolUseBlock
from ai_engineer.models.base import ModelRequest
from ai_engineer.models.prompted_tools import INVALID_TOOL_CALL
from ai_engineer.providers._schema import to_openapi_subset
from ai_engineer.providers.google import GoogleProvider
from ai_engineer.providers.ollama import OllamaProvider
from ai_engineer.providers.openai import OpenAIProvider

TOOL = ToolSpec(
    name="read_file",
    description="Read a file",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
)


def conversation() -> list[Message]:
    return [
        Message.user("read a.py"),
        Message(role="assistant", content=[ToolUseBlock(id="call_1", name="read_file", input={"path": "a.py"})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", name="read_file", content="x = 1")]),
    ]


class Recorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


# ---- OpenAI -------------------------------------------------------------------------------


def openai_provider(rec: Recorder, compatible: bool = False, monkeypatch: pytest.MonkeyPatch | None = None) -> OpenAIProvider:
    cfg = ProviderConfig(
        type="openai_compatible" if compatible else "openai",
        api_key_env="TEST_OPENAI_KEY",
        base_url="http://llm.test/v1" if compatible else None,
    )
    p = OpenAIProvider("openai", cfg)
    p.set_transport(httpx.MockTransport(rec))
    return p


async def test_openai_request_and_tool_call_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-123")
    rec = Recorder(
        [
            httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "model": "m",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {"id": "call_9", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}}
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4}},
                },
            )
        ]
    )
    p = openai_provider(rec)
    resp = await p.generate(ModelRequest(model="m", system="sys", tools=[TOOL], messages=conversation(), max_tokens=100))
    body = rec.body()
    assert rec.requests[0].url == "https://api.openai.com/v1/chat/completions"
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test-123"
    assert body["max_completion_tokens"] == 100 and "temperature" not in body
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}'
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "x = 1"}
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert resp.stop_reason == StopReason.TOOL_USE
    assert resp.tool_uses()[0].input == {"path": "b.py"}
    assert resp.usage.input_tokens == 12 and resp.usage.cache_read_tokens == 4 and not resp.usage.estimated


async def test_openai_compatible_uses_max_tokens_and_optional_key() -> None:
    rec = Recorder([httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})])
    p = openai_provider(rec, compatible=True)
    resp = await p.generate(ModelRequest(model="local", messages=[Message.user("x")], max_tokens=50))
    assert str(rec.requests[0].url) == "http://llm.test/v1/chat/completions"
    assert "authorization" not in rec.requests[0].headers
    assert rec.body()["max_tokens"] == 50
    assert resp.text() == "hi" and resp.stop_reason == StopReason.END_TURN


async def test_openai_malformed_arguments_become_invalid_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "k")
    rec = Recorder(
        [
            httpx.Response(
                200,
                json={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"id": "c", "function": {"name": "read_file", "arguments": "{bad"}}]}}]},
            )
        ]
    )
    resp = await openai_provider(rec).generate(ModelRequest(model="m", messages=[Message.user("x")]))
    assert resp.tool_uses()[0].name == INVALID_TOOL_CALL


@pytest.mark.parametrize(
    ("status", "body", "headers", "exc"),
    [
        (401, {"error": {"message": "bad key"}}, {}, AuthenticationError),
        (429, {"error": {"message": "slow down"}}, {"retry-after": "7"}, RateLimitError),
        (503, {"error": {"message": "overloaded"}}, {}, RetryableProviderError),
        (400, {"error": {"message": "This model's maximum context length is 8192 tokens"}}, {}, ContextLengthError),
        (400, {"error": {"message": "bad param"}}, {}, InvalidRequestError),
    ],
)
async def test_openai_error_mapping(monkeypatch: pytest.MonkeyPatch, status, body, headers, exc) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "k")
    rec = Recorder([httpx.Response(status, json=body, headers=headers)])
    with pytest.raises(exc) as info:
        await openai_provider(rec).generate(ModelRequest(model="m", messages=[Message.user("x")]))
    if exc is RateLimitError:
        assert info.value.retry_after == 7.0


async def test_openai_missing_key_is_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
    rec = Recorder([])
    with pytest.raises(AuthenticationError, match="TEST_OPENAI_KEY"):
        await openai_provider(rec).generate(ModelRequest(model="m", messages=[Message.user("x")]))


async def test_openai_connection_error_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "k")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    p = OpenAIProvider("openai", ProviderConfig(type="openai", api_key_env="TEST_OPENAI_KEY"))
    p.set_transport(httpx.MockTransport(boom))
    with pytest.raises(ProviderUnavailableError):
        await p.generate(ModelRequest(model="m", messages=[Message.user("x")]))


async def test_openai_streaming_accumulates_text_and_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "k")
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_", "arguments": '{"pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": 'th": "q"}'}}]}, "finish_reason": "tool_calls"}]},
        {"usage": {"prompt_tokens": 3, "completion_tokens": 2}, "choices": []},
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    rec = Recorder([httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})])
    p = openai_provider(rec)
    events = [e async for e in p.stream(ModelRequest(model="m", messages=[Message.user("x")], tools=[TOOL]))]
    assert "".join(e.text for e in events if e.type == "text_delta") == "Hello"
    final = events[-1].response
    assert final is not None and final.tool_uses()[0].name == "read_file"
    assert final.tool_uses()[0].input == {"path": "q"}
    assert final.usage.output_tokens == 2
    assert rec.body()["stream"] is True


async def test_openai_list_models_and_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "k")
    rec = Recorder(
        [
            httpx.Response(200, json={"data": [{"id": "b"}, {"id": "a"}]}),
            httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}),
        ]
    )
    p = openai_provider(rec)
    assert await p.list_models() == ["a", "b"]
    assert await p.embeddings(["x", "y"], "emb") == [[1.0, 0.0], [0.0, 1.0]]
    health = await p.health_check()
    assert health.ok is False  # recorder exhausted -> error surfaced, not hidden


# ---- Ollama --------------------------------------------------------------------------------


async def test_ollama_request_and_response() -> None:
    rec = Recorder(
        [
            httpx.Response(
                200,
                json={
                    "model": "q",
                    "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "c"}}}]},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 20,
                    "eval_count": 7,
                },
            )
        ]
    )
    cfg = ProviderConfig(type="ollama", defaults=ModelOptions(params={"num_ctx": 16384}))
    p = OllamaProvider("ollama", cfg)
    p.set_transport(httpx.MockTransport(rec))
    resp = await p.generate(ModelRequest(model="q", system="s", tools=[TOOL], messages=conversation(), max_tokens=64))
    body = rec.body()
    assert str(rec.requests[0].url) == "http://localhost:11434/api/chat"
    assert body["stream"] is False and body["options"] == {"num_predict": 64, "num_ctx": 16384}
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert body["messages"][3] == {"role": "tool", "content": "x = 1", "tool_name": "read_file"}
    assert resp.stop_reason == StopReason.TOOL_USE and resp.tool_uses()[0].input == {"path": "c"}
    assert resp.usage.input_tokens == 20


async def test_ollama_length_stop_and_models() -> None:
    rec = Recorder(
        [
            httpx.Response(200, json={"message": {"content": "partial"}, "done": True, "done_reason": "length"}),
            httpx.Response(200, json={"models": [{"name": "z:1b"}, {"name": "a:7b"}]}),
        ]
    )
    p = OllamaProvider("ollama", ProviderConfig(type="ollama", base_url="http://gpu:11434/"))
    p.set_transport(httpx.MockTransport(rec))
    resp = await p.generate(ModelRequest(model="q", messages=[Message.user("x")]))
    assert resp.stop_reason == StopReason.MAX_TOKENS
    assert await p.list_models() == ["a:7b", "z:1b"]


# ---- Google ---------------------------------------------------------------------------------


async def test_google_round_trip_preserves_thought_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_GEMINI_KEY", "g-key")
    raw_parts = [
        {"text": "thinking summary", "thought": True},
        {"functionCall": {"name": "read_file", "args": {"path": "d"}}, "thoughtSignature": "SIG"},
    ]
    rec = Recorder(
        [
            httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"role": "model", "parts": raw_parts}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 3, "thoughtsTokenCount": 2},
                },
            ),
            httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}]}),
        ]
    )
    p = GoogleProvider("google", ProviderConfig(type="google", api_key_env="TEST_GEMINI_KEY"))
    p.set_transport(httpx.MockTransport(rec))
    first = await p.generate(ModelRequest(model="gem", system="s", tools=[TOOL], messages=[Message.user("go")]))
    assert rec.requests[0].headers["x-goog-api-key"] == "g-key"
    assert str(rec.requests[0].url).endswith("/models/gem:generateContent")
    assert first.stop_reason == StopReason.TOOL_USE
    assert first.text() == ""  # thought parts are never surfaced
    assert first.usage.output_tokens == 5
    call = first.tool_uses()[0]
    # tool declarations use the OpenAPI subset (no additionalProperties)
    assert "additionalProperties" not in rec.body(0)["tools"][0]["functionDeclarations"][0]["parameters"]

    history = [
        Message.user("go"),
        first.to_message(),
        Message(role="user", content=[ToolResultBlock(tool_use_id=call.id, name="read_file", content="data")]),
    ]
    second = await p.generate(ModelRequest(model="gem", tools=[TOOL], messages=history))
    assert second.text() == "done"
    contents = rec.body(1)["contents"]
    assert contents[1] == {"role": "model", "parts": raw_parts}  # replayed verbatim incl. signature
    assert contents[2]["parts"][0] == {"functionResponse": {"name": "read_file", "response": {"result": "data"}}}

    # A different model must not receive this model's opaque parts.
    p2_req = ModelRequest(model="other", messages=history)
    model_parts = p.to_contents(p2_req)[1]["parts"]
    assert model_parts == [{"functionCall": {"name": "read_file", "args": {"path": "d"}}}]


async def test_google_blocked_prompt_is_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_GEMINI_KEY", "g")
    rec = Recorder([httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})])
    p = GoogleProvider("google", ProviderConfig(type="google", api_key_env="TEST_GEMINI_KEY"))
    p.set_transport(httpx.MockTransport(rec))
    resp = await p.generate(ModelRequest(model="gem", messages=[Message.user("x")]))
    assert resp.stop_reason == StopReason.REFUSAL and resp.refusal_detail == "SAFETY"


def test_openapi_subset_inlines_refs_and_nullable() -> None:
    schema = {
        "type": "object",
        "$defs": {"Item": {"type": "object", "title": "Item", "properties": {"n": {"type": "integer"}}}},
        "properties": {
            "item": {"$ref": "#/$defs/Item"},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "additionalProperties": False,
    }
    out = to_openapi_subset(schema)
    assert out["properties"]["item"]["properties"]["n"] == {"type": "integer"}
    assert out["properties"]["note"] == {"type": "string", "nullable": True}
    assert "additionalProperties" not in out and "$defs" not in out


# ---- Anthropic (official SDK over a mocked transport) ------------------------------------------


def anthropic_provider(handler, monkeypatch: pytest.MonkeyPatch, **options):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    from ai_engineer.providers.anthropic import AnthropicProvider

    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test")
    http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    cfg = ProviderConfig(type="anthropic", api_key_env="TEST_ANTHROPIC_KEY", options=options)
    return AnthropicProvider("anthropic", cfg, http_client=http_client)


def _anthropic_message(content, stop_reason="end_turn", usage=None):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 10, "output_tokens": 4},
    }


async def test_anthropic_request_shape_and_thinking_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx2 = pytest.importorskip("httpx2")
    seen: list[dict] = []
    replies = [
        _anthropic_message(
            [
                {"type": "thinking", "thinking": "", "signature": "sig-abc"},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "e"}},
            ],
            stop_reason="tool_use",
            usage={"input_tokens": 50, "output_tokens": 9, "cache_read_input_tokens": 40},
        ),
        _anthropic_message([{"type": "text", "text": "all done"}]),
    ]

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=replies.pop(0))

    p = anthropic_provider(handler, monkeypatch)
    p.options_for("claude-test").params.update({})
    first = await p.generate(ModelRequest(model="claude-test", system="sys", tools=[TOOL], messages=[Message.user("go")]))
    body = seen[0]
    assert body["system"] == "sys" and body["tools"][0]["input_schema"]["required"] == ["path"]
    assert body["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in body and "tool_choice" not in body
    assert first.stop_reason == StopReason.TOOL_USE
    assert isinstance(first.content[0], OpaqueBlock)
    assert first.usage.cache_read_tokens == 40

    history = [
        Message.user("go"),
        first.to_message(),
        Message(role="user", content=[ToolResultBlock(tool_use_id="toolu_1", name="read_file", content="contents")]),
    ]
    second = await p.generate(ModelRequest(model="claude-test", tools=[TOOL], messages=history))
    assert second.text() == "all done"
    replay = seen[1]["messages"][1]["content"]
    assert replay[0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    assert replay[1]["type"] == "tool_use"
    assert seen[1]["messages"][2]["content"][0]["type"] == "tool_result"

    # A different model never receives this model's reasoning blocks.
    other = p.to_messages(ModelRequest(model="claude-other", messages=history))
    assert [b["type"] for b in other[1]["content"]] == ["tool_use"]


async def test_anthropic_errors_and_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx2 = pytest.importorskip("httpx2")
    replies = [
        httpx2.Response(429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}, headers={"retry-after": "3"}),
        httpx2.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long: 300000 tokens"}}),
        httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
        httpx2.Response(200, json=_anthropic_message([], stop_reason="refusal")),
    ]
    p = anthropic_provider(lambda r: replies.pop(0), monkeypatch)
    r = ModelRequest(model="claude-test", messages=[Message.user("x")])
    with pytest.raises(RateLimitError) as info:
        await p.generate(r)
    assert info.value.retry_after == 3.0
    with pytest.raises(ContextLengthError):
        await p.generate(r)
    with pytest.raises(RetryableProviderError):
        await p.generate(r)
    assert (await p.generate(r)).stop_reason == StopReason.REFUSAL


async def test_anthropic_server_side_fallback_auto_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx2 = pytest.importorskip("httpx2")
    seen: list[tuple[dict, str]] = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((body, request.headers.get("anthropic-beta", "")))
        if "fallbacks" in body:
            return httpx2.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks not supported for this model"}})
        return httpx2.Response(200, json=_anthropic_message([{"type": "text", "text": "ok"}]))

    p = anthropic_provider(handler, monkeypatch, server_side_fallback=True)
    r = ModelRequest(model="claude-test", messages=[Message.user("x")])
    assert (await p.generate(r)).text() == "ok"
    assert seen[0][0]["fallbacks"] == "default" and "server-side-fallback" in seen[0][1]
    assert "fallbacks" not in seen[1][0]
    await p.generate(r)
    assert len(seen) == 3 and "fallbacks" not in seen[2][0]  # remembered


async def test_anthropic_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    from ai_engineer.providers.anthropic import AnthropicProvider

    monkeypatch.delenv("NOPE_KEY", raising=False)
    p = AnthropicProvider("anthropic", ProviderConfig(type="anthropic", api_key_env="NOPE_KEY"))
    with pytest.raises(AuthenticationError):
        await p.generate(ModelRequest(model="m", messages=[Message.user("x")]))


async def test_anthropic_large_requests_stream_and_assemble(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx2 = pytest.importorskip("httpx2")
    events = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_s", "type": "message", "role": "assistant", "model": "claude-test", "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 30, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "stream"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "read_file", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"path\": \"z.py\"}"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 12}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    sse = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    seen: list[dict] = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    p = anthropic_provider(handler, monkeypatch, stream_threshold_tokens=1000)
    resp = await p.generate(ModelRequest(model="claude-test", messages=[Message.user("x")], tools=[TOOL], max_tokens=64000))
    assert seen[0]["stream"] is True and seen[0]["max_tokens"] == 64000
    assert resp.text() == "Hello stream"
    assert resp.tool_uses()[0].input == {"path": "z.py"}
    assert resp.stop_reason == StopReason.TOOL_USE and resp.usage.output_tokens == 12


async def test_anthropic_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx2 = pytest.importorskip("httpx2")

    def handler(request):
        assert request.url.path.endswith("/v1/models")
        return httpx2.Response(200, json={"data": [
            {"type": "model", "id": "model-b", "display_name": "B", "created_at": "2026-01-01T00:00:00Z"},
            {"type": "model", "id": "model-a", "display_name": "A", "created_at": "2026-01-01T00:00:00Z"},
        ], "has_more": False, "first_id": "model-b", "last_id": "model-a"})

    p = anthropic_provider(handler, monkeypatch)
    assert await p.list_models() == ["model-a", "model-b"]


async def test_ollama_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        {"message": {"role": "assistant", "content": "Hi "}, "done": False},
        {"message": {"role": "assistant", "content": "there", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "q"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 3},
    ]
    body = "\n".join(json.dumps(x) for x in lines) + "\n"
    rec = Recorder([httpx.Response(200, content=body.encode())])
    p = OllamaProvider("ollama", ProviderConfig(type="ollama"))
    p.set_transport(httpx.MockTransport(rec))
    events = [e async for e in p.stream(ModelRequest(model="q", messages=[Message.user("x")], tools=[TOOL]))]
    assert "".join(e.text for e in events if e.type == "text_delta") == "Hi there"
    final = events[-1].response
    assert final.tool_uses()[0].input == {"path": "q"} and final.usage.input_tokens == 5
    assert rec.body()["stream"] is True
