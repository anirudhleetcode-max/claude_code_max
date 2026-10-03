"""Helpers for end-to-end tests: real repositories, real commands, scripted models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ai_engineer.config.settings import ModelsSettings
from ai_engineer.models.registry import ProviderRegistry
from ai_engineer.providers.scripted import ScriptedProvider
from ai_engineer.runtime import Runtime
from tests.conftest import git

PYTEST = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'


def make_calc_repo(root: Path, buggy: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "calc").mkdir()
    (root / "calc" / "__init__.py").write_text(
        '"""Tiny calculator."""\n\n\ndef add(a, b):\n    return a ' + ("-" if buggy else "+") + " b\n\n\ndef sub(a, b):\n    return a - b\n"
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_calc.py").write_text(
        "from calc import add, sub\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\ndef test_sub():\n    assert sub(5, 3) == 2\n"
    )
    (root / "pyproject.toml").write_text('[project]\nname = "calc"\nversion = "0.0.1"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    return root


def j(data: dict[str, Any]) -> str:
    return json.dumps(data)


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": name, "input": args}]}


def understanding(task_type: str = "change", complexity: str = "small", **extra: Any) -> str:
    data = {
        "summary": extra.pop("summary", "Fix add() so it returns the sum"),
        "task_type": task_type,
        "complexity": complexity,
        "requirements": extra.pop("requirements", ["add(a, b) returns a + b"]),
        "acceptance_criteria": extra.pop("acceptance_criteria", ["tests/test_calc.py passes"]),
        "needs": extra.pop("needs", {"tests": True}),
        **extra,
    }
    return j(data)


APPROVE = j({
    "verdict": "approve",
    "summary": "Change is correct and minimal.",
    "issues": [],
    "requirements": [{"criterion": "tests/test_calc.py passes", "status": "met", "evidence": "pytest passed"}],
})


def open_runtime(workspace: Path, providers: dict[str, ScriptedProvider], roles: dict[str, list[str]] | None = None, **overrides: Any) -> Runtime:
    registry = ProviderRegistry(ModelsSettings())
    for name, provider in providers.items():
        registry.register_instance(name, provider)
    first = next(iter(providers))
    config: dict[str, Any] = {
        "models": {
            "providers": {name: {"type": "scripted"} for name in providers},
            "roles": roles or {"default": [f"{first}:m"]},
            "retry": {"max_attempts": 2, "base_delay_s": 0.0, "max_delay_s": 0.0},
        },
        "validation": {"test_command": PYTEST, "dependency_audit": False},
        "agent": {"max_steps": 12},
        "permissions": {"mode": "developer"},
    }
    for key, value in overrides.items():
        config.setdefault(key, {}).update(value)
    return Runtime.open(workspace, config, registry=registry, use_global_config=False)
