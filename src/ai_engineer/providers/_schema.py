"""JSON-schema helpers for providers with restricted schema dialects."""

from __future__ import annotations

import copy
from typing import Any

# Keywords understood by OpenAPI-subset schema dialects (e.g. Gemini function declarations).
_ALLOWED = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "properties",
    "required",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "propertyOrdering",
}


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local ``$ref`` pointers into ``$defs``/``definitions`` (no recursion support)."""
    schema = copy.deepcopy(schema)
    defs = {**schema.pop("$defs", {}), **schema.pop("definitions", {})}

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 32:
            return {"type": "object"}
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                name = ref.rsplit("/", 1)[-1]
                target = defs.get(name, {"type": "object"})
                merged = {**resolve(target, depth + 1), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: resolve(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    return resolve(schema)


def to_openapi_subset(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON schema to the OpenAPI subset accepted by restrictive providers."""

    def convert(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        node = dict(node)
        # Optional[X] in pydantic → anyOf [X, null] → X with nullable
        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            non_null = [s for s in any_of if not (isinstance(s, dict) and s.get("type") == "null")]
            if len(non_null) == 1 and len(non_null) < len(any_of):
                base = convert(non_null[0])
                base["nullable"] = True
                if "description" in node:
                    base.setdefault("description", node["description"])
                return base
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _ALLOWED:
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {k: convert(v) for k, v in value.items()}
            elif key == "items":
                out[key] = convert(value)
            elif key == "anyOf" and isinstance(value, list):
                out[key] = [convert(v) for v in value]
            elif key == "type" and isinstance(value, list):
                types = [t for t in value if t != "null"]
                out["type"] = types[0] if types else "string"
                if "null" in value:
                    out["nullable"] = True
            else:
                out[key] = value
        if out.get("type") == "object" and "properties" not in out:
            out["properties"] = {}
        if not out.get("required"):
            out.pop("required", None)
        return out

    return convert(inline_refs(schema))
