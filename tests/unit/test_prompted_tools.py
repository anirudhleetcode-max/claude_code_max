from __future__ import annotations

from ai_engineer.config.settings import ModelOptions, ProviderConfig
from ai_engineer.core.types import Message, StopReason, TextBlock, ToolResultBlock, ToolSpec, ToolUseBlock
from ai_engineer.models.base import ModelRequest
from ai_engineer.models.prompted_tools import INVALID_TOOL_CALL, adapt_request, parse_response
from ai_engineer.providers.scripted import ScriptedProvider, response_from_dict

TOOLS = [ToolSpec(name="read_file", description="Read a file", input_schema={"type": "object", "properties": {"path": {"type": "string"}}})]


def test_adapt_request_moves_tools_into_system_and_renders_history() -> None:
    req = ModelRequest(
        system="base",
        tools=TOOLS,
        messages=[
            Message.user("go"),
            Message(role="assistant", content=[ToolUseBlock(id="1", name="read_file", input={"path": "a.py"})]),
            Message(role="user", content=[ToolResultBlock(tool_use_id="1", name="read_file", content="print(1)")]),
        ],
    )
    out = adapt_request(req)
    assert out.tools == []
    assert out.system.startswith("base") and "read_file" in out.system and "<tool_call>" in out.system
    assert '"path": "a.py"' in out.messages[1].text()
    assert '<tool_result name="read_file"' in out.messages[2].text()


def test_parse_response_extracts_calls_and_text() -> None:
    resp = response_from_dict(
        {"text": 'Let me look.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "x"}}\n</tool_call>'}
    )
    parsed = parse_response(resp, {"read_file"})
    assert parsed.stop_reason == StopReason.TOOL_USE
    assert parsed.text() == "Let me look."
    call = parsed.tool_uses()[0]
    assert call.name == "read_file" and call.input == {"path": "x"}


def test_parse_response_flags_malformed_calls() -> None:
    resp = response_from_dict({"text": "<tool_call>{not json</tool_call>"})
    parsed = parse_response(resp, {"read_file"})
    assert parsed.tool_uses()[0].name == INVALID_TOOL_CALL


def test_parse_response_accepts_bare_json_call_for_known_tool() -> None:
    resp = response_from_dict({"text": '{"name": "read_file", "arguments": {"path": "y"}}'})
    parsed = parse_response(resp, {"read_file"})
    assert parsed.tool_uses()[0].input == {"path": "y"}
    other = parse_response(response_from_dict({"text": '{"name": "nope", "arguments": {}}'}), {"read_file"})
    assert other.tool_uses() == []


async def test_provider_without_native_tools_uses_prompt_protocol() -> None:
    cfg = ProviderConfig(type="scripted", defaults=ModelOptions(native_tools=False))
    p = ScriptedProvider("s", cfg, steps=['<tool_call>{"name":"read_file","arguments":{"path":"z"}}</tool_call>'])
    resp = await p.generate(ModelRequest(model="m", tools=TOOLS, messages=[Message.user("hi")]))
    assert resp.tool_uses()[0].input == {"path": "z"}
    sent = p.requests[0]
    assert sent.tools == [] and "read_file" in sent.system
    assert isinstance(sent.messages[0].content[0], TextBlock)
    assert resp.usage.estimated
