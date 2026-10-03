"""Structured output: request JSON, validate against a pydantic schema, repair."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..config.settings import ModelOptions
from ..core.errors import MalformedOutputError
from ..core.types import Message, StopReason
from ..core.util import extract_json
from .base import ModelRequest, ModelResponse, StructuredResult

T = TypeVar("T", bound=BaseModel)

GenerateFn = Callable[[ModelRequest], Awaitable[ModelResponse]]


def schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object and nothing else (no prose, no code fences). "
        "It must validate against this JSON schema:\n"
        + json.dumps(schema.model_json_schema(), separators=(",", ":"))
    )


def parse_structured(text: str, schema: type[T]) -> T:
    data = extract_json(text)
    return schema.model_validate(data)


async def structured_generate(
    generate: GenerateFn,
    req: ModelRequest,
    schema: type[BaseModel],
    options: ModelOptions,
    max_repairs: int = 2,
) -> StructuredResult:
    """Ask for JSON matching ``schema``; on invalid output, show the errors and retry.

    Raises :class:`MalformedOutputError` after ``max_repairs`` failed repairs.
    """
    system = (req.system + "\n\n" if req.system else "") + schema_instructions(schema)
    update: dict = {"system": system, "tools": []}
    if options.structured_output == "native":
        update["response_schema"] = schema.model_json_schema()
    current = req.model_copy(update=update)
    last_error = ""
    resp: ModelResponse | None = None
    for attempt in range(max_repairs + 1):
        resp = await generate(current)
        if resp.stop_reason == StopReason.REFUSAL:
            raise MalformedOutputError("model refused to produce structured output", model=resp.model, provider=resp.provider)
        text = resp.text()
        try:
            value = parse_structured(text, schema)
            return StructuredResult(value=value, response=resp, repairs=attempt)
        except (ValueError, ValidationError) as exc:
            last_error = _summarize_error(exc)
            messages = [
                *current.messages,
                Message.assistant(text or "(empty response)"),
                Message.user(
                    "Your previous reply was not valid JSON for the required schema.\n"
                    f"Problems: {last_error}\n"
                    "Reply again with only the corrected JSON object."
                ),
            ]
            current = current.model_copy(update={"messages": messages})
    raise MalformedOutputError(
        f"structured output invalid after {max_repairs} repairs: {last_error}",
        model=resp.model if resp else req.model,
        provider=resp.provider if resp else "",
    )


def _summarize_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:8]:
            loc = ".".join(str(p) for p in err.get("loc", ()))
            parts.append(f"{loc or '<root>'}: {err.get('msg')}")
        return "; ".join(parts)
    return str(exc)[:500]
