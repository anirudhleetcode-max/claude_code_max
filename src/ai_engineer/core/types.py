"""Provider-neutral conversation types.

Every provider adapter translates to and from these types, so nothing outside
``ai_engineer.providers`` needs to know any provider's wire format.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(_Block):
    """A request from the model to call a tool."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    # Per-provider round-trip data (e.g. {"google": {"thoughtSignature": "..."}}).
    # Only the provider named by the key ever sees its entry.
    meta: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(_Block):
    """The result of executing a tool, sent back to the model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    name: str = ""
    content: str
    is_error: bool = False


class OpaqueBlock(_Block):
    """Provider-specific content that must be echoed back unchanged.

    Example: reasoning blocks that a provider requires to be replayed verbatim on
    the same model. Adapters drop opaque blocks that belong to a different
    provider or model.
    """

    type: Literal["opaque"] = "opaque"
    provider: str
    model: str
    payload: dict[str, Any]


Block = Annotated[TextBlock | ToolUseBlock | ToolResultBlock | OpaqueBlock, Field(discriminator="type")]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: list[Block] = Field(default_factory=list)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=[TextBlock(text=text)])

    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]


class ToolSpec(BaseModel):
    """Description of a tool as presented to a model."""

    name: str
    description: str
    input_schema: dict[str, Any]


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"
    PAUSE = "pause"
    OTHER = "other"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated: bool = False

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            estimated=self.estimated or other.estimated,
        )


def message_char_size(message: Message) -> int:
    """Approximate serialized size of a message, used for token estimation."""
    total = 0
    for block in message.content:
        if isinstance(block, TextBlock):
            total += len(block.text)
        elif isinstance(block, ToolUseBlock):
            total += len(block.name) + len(json.dumps(block.input, ensure_ascii=False))
        elif isinstance(block, ToolResultBlock):
            total += len(block.content)
        elif isinstance(block, OpaqueBlock):
            total += len(json.dumps(block.payload, ensure_ascii=False)) // 4
    return total
