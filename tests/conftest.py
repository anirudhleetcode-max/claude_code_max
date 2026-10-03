from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ai_engineer.config.settings import ModelsSettings, ProviderConfig
from ai_engineer.models.registry import ProviderRegistry
from ai_engineer.models.router import ModelRouter
from ai_engineer.providers.scripted import ScriptedProvider


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real user's global config/memory during tests."""
    home = tmp_path_factory.mktemp("aie_home")
    monkeypatch.setenv("AIE_HOME", str(home))
    for var in list(os.environ):
        if var.startswith("AIE__") or var in ("AIE_MODEL", "AIE_MODE", "AIE_REVIEW_MODEL", "AIE_FAST_MODEL"):
            monkeypatch.delenv(var, raising=False)


async def _no_sleep(_: float) -> None:
    return None


def make_router(*providers: ScriptedProvider, roles: dict[str, list[str]] | None = None, **settings_kw) -> ModelRouter:
    cfg = ModelsSettings(
        providers={p.name: ProviderConfig(type="scripted") for p in providers},
        roles=roles if roles is not None else {"default": [f"{providers[0].name}:m"]},
        **settings_kw,
    )
    registry = ProviderRegistry(cfg)
    for p in providers:
        registry.register_instance(p.name, p)
    return ModelRouter(cfg, registry, sleep=_no_sleep)


def git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("# demo\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial")
    return repo
