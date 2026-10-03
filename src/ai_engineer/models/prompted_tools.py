"""Tool calling for models without native tool support.

Tools are described in the system prompt and the model emits
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks. History is
converted so earlier tool calls/results appear in the same textual protocol.
"""

from __future__ import annotations

import json
import re

from ..core.ids import short_id
from ..core.types import Message, StopReason, TextBlock, ToolResultBlock, ToolUseBlock
from .base import ModelRequest, ModelResponse

INVALID_TOOL_CALL = "invalid_tool_call"

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def tools_prompt(req: ModelRequest) -> str:
    lines = [
        "# Tools",
        "You can call tools. To call a tool, write a block exactly like this:",
        "<tool_call>",
        '{"name": "<tool name>", "arguments": {<arguments as JSON>}}',
        "</tool_call>",
        "You may write several tool_call blocks in one reply. After writing tool calls, stop and",
        "wait: the results arrive in <tool_result> blocks in the next message. Never invent tool",
        "results. When you need no more tools, reply normally without any tool_call block.",
        "",
        "Available tools:",
    ]
    for tool in req.tools:
        lines.append(f"## {tool.name}")
        lines.append(tool.description.strip())
        lines.append("Arguments JSON schema: " + json.dumps(tool.input_schema, separators=(",", ":")))
        lines.append("")
    return "\n".join(lines)


def _render_message(msg: Message) -> Message:
    blocks: list[TextBlock] = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            blocks.append(block)
        elif isinstance(block, ToolUseBlock):
            payload = json.dumps({"name": block.name, "arguments": block.input}, ensure_ascii=False)
            blocks.append(TextBlock(text=f"<tool_call>\n{payload}\n</tool_call>"))
        elif isinstance(block, ToolResultBlock):
            status = "error" if block.is_error else "ok"
            blocks.append(
                TextBlock(
                    text=f'<tool_result name="{block.name}" id="{block.tool_use_id}" status="{status}">\n'
                    f"{block.content}\n</tool_result>"
                )
            )
        # opaque blocks are provider-native and meaningless in this protocol
    if not blocks:
        blocks.append(TextBlock(text="(empty)"))
    return Message(role=msg.role, content=list(blocks))


def adapt_request(req: ModelRequest) -> ModelRequest:
    system = (req.system + "\n\n" if req.system else "") + tools_prompt(req)
    return req.model_copy(
        update={"system": system, "tools": [], "messages": [_render_message(m) for m in req.messages]}
    )


def _parse_call(raw: str) -> ToolUseBlock:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        from ..core.util import extract_json

        try:
            data = extract_json(raw)
        except ValueError as exc:
            return ToolUseBlock(id=f"call_{short_id(12)}", name=INVALID_TOOL_CALL, input={"raw": raw[:2000], "error": str(exc)})
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return ToolUseBlock(
            id=f"call_{short_id(12)}", name=INVALID_TOOL_CALL, input={"raw": raw[:2000], "error": "missing 'name'"}
        )
    args = data.get("arguments", data.get("input", data.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return ToolUseBlock(id=f"call_{short_id(12)}", name=data["name"], input=args)


def parse_response(resp: ModelResponse, tool_names: set[str]) -> ModelResponse:
    text = resp.text()
    calls = [_parse_call(m.group(1)) for m in _CALL_RE.finditer(text)]
    if not calls:
        # Some models emit a bare JSON object instead of the tagged block.
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidate = _parse_call(stripped)
            if candidate.name in tool_names:
                calls = [candidate]
                text = ""
    remaining = _CALL_RE.sub("", text).strip()
    content: list = []
    if remaining:
        content.append(TextBlock(text=remaining))
    content.extend(calls)
    stop = StopReason.TOOL_USE if calls else resp.stop_reason
    return resp.model_copy(update={"content": content, "stop_reason": stop})
