"""Layered configuration loading and provider auto-detection."""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..core.errors import ConfigError
from ..core.util import deep_merge
from .paths import global_config_dir, project_state_dir
from .settings import Settings

ENV_PREFIX = "AIE__"

# Environment variables conventionally used by each provider type.
PROVIDER_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def _unescape(match: re.Match[str]) -> str:
    return _ESCAPES.get(match.group(1), "\\" + match.group(1))


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse a ``.env`` file: KEY=VALUE lines, optional ``export``, quotes, comments."""
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            quote = value[0]
            value = value[1:-1]
            if quote == '"' and "\\" in value:
                value = re.sub(r"\\(.)", _unescape, value)
        else:
            # strip inline comments for unquoted values
            hash_pos = value.find(" #")
            if hash_pos != -1:
                value = value[:hash_pos].rstrip()
        result[key] = value
    return result


def load_dotenv(path: Path, override: bool = False) -> list[str]:
    """Load variables from ``path`` into ``os.environ``. Returns the names loaded."""
    if not path.is_file():
        return []
    loaded = []
    for key, value in parse_dotenv(path.read_text(encoding="utf-8")).items():
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """``AIE__SECTION__KEY=value`` → nested dict. Values are parsed as JSON when possible."""
    out: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in name[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    # convenience shortcuts
    if environ.get("AIE_MODE"):
        out.setdefault("permissions", {})["mode"] = environ["AIE_MODE"]
    if environ.get("AIE_MODEL"):
        chain = [m.strip() for m in environ["AIE_MODEL"].split(",") if m.strip()]
        out.setdefault("models", {}).setdefault("roles", {}).setdefault("default", chain)
    if environ.get("AIE_REVIEW_MODEL"):
        chain = [m.strip() for m in environ["AIE_REVIEW_MODEL"].split(",") if m.strip()]
        out.setdefault("models", {}).setdefault("roles", {}).setdefault("reviewer", chain)
    if environ.get("AIE_FAST_MODEL"):
        chain = [m.strip() for m in environ["AIE_FAST_MODEL"].split(",") if m.strip()]
        out.setdefault("models", {}).setdefault("roles", {}).setdefault("fast", chain)
    return out


def autodetect_providers(environ: dict[str, str]) -> dict[str, Any]:
    """Register providers whose credentials are present, plus local Ollama.

    Explicitly configured providers always take precedence over these.
    """
    providers: dict[str, Any] = {}
    for ptype, envs in PROVIDER_KEY_ENVS.items():
        for env in envs:
            if environ.get(env):
                providers[ptype] = {"type": ptype, "api_key_env": env}
                break
    if environ.get("OPENAI_COMPATIBLE_BASE_URL"):
        providers["local"] = {
            "type": "openai_compatible",
            "base_url": environ["OPENAI_COMPATIBLE_BASE_URL"],
            "api_key_env": "OPENAI_COMPATIBLE_API_KEY" if environ.get("OPENAI_COMPATIBLE_API_KEY") else None,
        }
    providers["ollama"] = {"type": "ollama", "base_url": environ.get("OLLAMA_HOST") or "http://localhost:11434"}
    return providers


def load_settings(
    workspace: Path | None = None,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    use_global: bool = True,
) -> Settings:
    env = dict(os.environ if environ is None else environ)
    data: dict[str, Any] = {"models": {"providers": autodetect_providers(env)}}
    if use_global:
        data = deep_merge(data, _read_toml(global_config_dir() / "config.toml"))
    if workspace is not None:
        data = deep_merge(data, _read_toml(project_state_dir(workspace) / "config.toml"))
    data = deep_merge(data, _env_overrides(env))
    if overrides:
        data = deep_merge(data, overrides)
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc


DEFAULT_PROJECT_CONFIG = """\
# AI Engineer project configuration.
# Credentials never go here: providers reference environment variables.
# Documentation: docs/CONFIGURATION.md

[permissions]
# safe | assisted | developer | autonomous
mode = "developer"
# Commands matching these patterns are auto-approved / always denied (fnmatch).
allow_commands = []
deny_commands = []

[agent]
max_steps = 60
max_repair_iterations = 4
max_review_iterations = 2

[gates]
# required | if_available | disabled
tests = "required"
lint = "if_available"
typecheck = "if_available"
build = "if_available"
security = "required"
review = "required"

[git]
checkpoints = true
auto_commit = "if_clean"   # or "never"

# --- Models -------------------------------------------------------------------
# Providers with keys in the environment are detected automatically
# (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, OLLAMA_HOST).
# Run `aie providers models` to list model ids your providers offer, then map
# roles to ordered fallback chains of "provider:model":
#
# [models.roles]
# default  = ["anthropic:<model-id>", "ollama:<model-id>"]
# reviewer = ["openai:<model-id>"]
# fast     = ["ollama:<model-id>"]
#
# [models.providers.local]
# type = "openai_compatible"
# base_url = "http://localhost:8000/v1"
"""
