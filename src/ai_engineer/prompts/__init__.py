"""Role system prompts, stored as editable Markdown files next to this module."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8").strip()
