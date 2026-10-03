"""Regenerate the reference sections of the documentation from the code.

Sections between ``<!-- BEGIN GENERATED: name -->`` and ``<!-- END GENERATED: name -->``
markers are replaced. Run ``python scripts/gen_docs.py`` after changing tools,
settings or CLI commands; ``--check`` exits non-zero when docs are out of date
(used by the test suite).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_engineer.config.settings import Settings  # noqa: E402
from ai_engineer.executor.control import CONTROL_TOOLS  # noqa: E402
from ai_engineer.tools.factory import default_tools  # noqa: E402
from ai_engineer.ui.cli import build_parser  # noqa: E402


def _type_of(schema: dict[str, Any]) -> str:
    if "anyOf" in schema:
        return " | ".join(_type_of(s) for s in schema["anyOf"])
    if "enum" in schema:
        return " | ".join(repr(v) for v in schema["enum"])
    if schema.get("type") == "array":
        return f"list[{_type_of(schema.get('items', {}))}]"
    if "const" in schema:
        return repr(schema["const"])
    return str(schema.get("type", "any"))


def tools_section() -> str:
    tools = [*default_tools(), *(cls() for cls in CONTROL_TOOLS)]
    from ai_engineer.tools.base import Tool

    lines = ["| Tool | Permission level | Side effect | Timeout | Purpose |", "|---|---|---|---|---|"]
    for tool in sorted(tools, key=lambda t: t.name):
        purpose = " ".join(tool.description.split()).replace("|", "\\|")
        assessed = type(tool).assess is not Tool.assess
        level = f"{tool.level.name} + per-call risk" if assessed else tool.level.name
        lines.append(f"| `{tool.name}` | {level} | {tool.side_effect} | {tool.timeout_s:.0f}s | {purpose} |")
    lines.append("")
    lines.append(
        "\"+ per-call risk\": the tool assesses each call (command risk classification, SQL "
        "classification, network access, writes) and the effective level can be higher than the base level."
    )
    lines.append("")
    for tool in sorted(tools, key=lambda t: t.name):
        spec = tool.spec()
        props = spec.input_schema.get("properties", {})
        required = set(spec.input_schema.get("required", []))
        lines.append(f"### `{tool.name}`")
        if not props:
            lines.append("No arguments.")
        else:
            lines.append("| Argument | Type | Required | Default | Description |")
            lines.append("|---|---|---|---|---|")
            for name, prop in props.items():
                default = prop.get("default", "")
                default_text = f"`{default!r}`" if name not in required and "default" in prop else ""
                desc = str(prop.get("description", "")).replace("|", "\\|")
                type_text = _type_of(prop).replace("|", "\\|")
                needed = "yes" if name in required else "no"
                lines.append(f"| `{name}` | {type_text} | {needed} | {default_text} | {desc} |")
        lines.append("")
    return "\n".join(lines).rstrip()


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '"" # unset'
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{}" if not value else "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
    return str(value)


def settings_section() -> str:
    from enum import IntEnum

    data = Settings().model_dump(mode="json")
    defaults = Settings()
    for section, values in data.items():
        for key in values:
            raw = getattr(getattr(defaults, section), key)
            if isinstance(raw, IntEnum):
                values[key] = raw.name
    lines = ["```toml"]
    for section, values in data.items():
        if section == "models":
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    models = data["models"]
    lines.append("[models]")
    lines.append(f"request_timeout_s = {models['request_timeout_s']}")
    lines.append("")
    for sub in ("retry", "circuit_breaker"):
        lines.append(f"[models.{sub}]")
        for key, value in models[sub].items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    lines.append("```")
    return "\n".join(lines)


def cli_section() -> str:
    parser = build_parser()
    lines = ["| Command | Description |", "|---|---|"]
    for action in parser._subparsers._group_actions:  # noqa: SLF001 - argparse has no public API for this
        for choice in action._choices_actions:  # noqa: SLF001
            lines.append(f"| `aie {choice.dest}` | {choice.help} |")
    return "\n".join(lines)


SECTIONS = {"tools": tools_section, "settings": settings_section, "cli": cli_section}
TARGETS = {
    "tools": [ROOT / "docs" / "TOOL_GUIDE.md"],
    "settings": [ROOT / "docs" / "CONFIGURATION.md"],
    "cli": [ROOT / "docs" / "AGENT_GUIDE.md"],
}


def render(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"markers for '{name}' not found")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated sections are stale")
    args = parser.parse_args()
    stale = []
    for name, build in SECTIONS.items():
        body = build()
        for path in TARGETS[name]:
            original = path.read_text(encoding="utf-8")
            updated = render(original, name, body)
            if updated != original:
                if args.check:
                    stale.append(f"{path.relative_to(ROOT)} [{name}]")
                else:
                    path.write_text(updated, encoding="utf-8")
                    print(f"updated {path.relative_to(ROOT)} [{name}]")
    if stale:
        print("stale generated docs (run python scripts/gen_docs.py): " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
