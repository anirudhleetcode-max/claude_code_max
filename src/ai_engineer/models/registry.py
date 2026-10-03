"""Provider registry: maps provider types to factories and instantiates providers lazily."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..config.settings import ModelsSettings, ProviderConfig
from ..core.errors import ConfigError
from .base import ModelProvider

ProviderFactory = Callable[[str, ProviderConfig], ModelProvider]


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @classmethod
    def parse(cls, ref: str) -> ModelRef:
        # Split on the first colon only: model ids may contain colons (e.g. "qwen2.5:7b").
        provider, sep, model = ref.partition(":")
        if not sep or not provider or not model:
            raise ConfigError(f"model reference '{ref}' must look like 'provider:model'")
        return cls(provider.strip(), model.strip())

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def _builtin_factories() -> dict[str, ProviderFactory]:
    # Imported lazily so optional dependencies are only needed when used.
    def anthropic(name: str, cfg: ProviderConfig) -> ModelProvider:
        from ..providers.anthropic import AnthropicProvider

        return AnthropicProvider(name, cfg)

    def openai(name: str, cfg: ProviderConfig) -> ModelProvider:
        from ..providers.openai import OpenAIProvider

        return OpenAIProvider(name, cfg)

    def google(name: str, cfg: ProviderConfig) -> ModelProvider:
        from ..providers.google import GoogleProvider

        return GoogleProvider(name, cfg)

    def ollama(name: str, cfg: ProviderConfig) -> ModelProvider:
        from ..providers.ollama import OllamaProvider

        return OllamaProvider(name, cfg)

    def scripted(name: str, cfg: ProviderConfig) -> ModelProvider:
        from ..providers.scripted import ScriptedProvider

        return ScriptedProvider.from_config(name, cfg)

    return {
        "anthropic": anthropic,
        "openai": openai,
        "openai_compatible": openai,
        "google": google,
        "ollama": ollama,
        "scripted": scripted,
    }


class ProviderRegistry:
    def __init__(self, settings: ModelsSettings) -> None:
        self.settings = settings
        self._factories: dict[str, ProviderFactory] = _builtin_factories()
        self._instances: dict[str, ModelProvider] = {}

    def register_type(self, type_name: str, factory: ProviderFactory) -> None:
        """Register a custom provider type (see docs/MODEL_PROVIDERS.md)."""
        self._factories[type_name] = factory

    def register_instance(self, name: str, provider: ModelProvider) -> None:
        """Register an already-constructed provider (used by tests and embedders)."""
        self._instances[name] = provider

    def names(self) -> list[str]:
        names = {n for n, c in self.settings.providers.items() if c.enabled}
        return sorted(names | set(self._instances))

    def get(self, name: str) -> ModelProvider:
        if name in self._instances:
            return self._instances[name]
        cfg = self.settings.providers.get(name)
        if cfg is None:
            raise ConfigError(f"unknown provider '{name}'. Configured: {', '.join(self.names()) or 'none'}")
        if not cfg.enabled:
            raise ConfigError(f"provider '{name}' is disabled")
        factory = self._factories.get(cfg.type)
        if factory is None:
            raise ConfigError(f"no factory for provider type '{cfg.type}'")
        provider = factory(name, cfg)
        self._instances[name] = provider
        return provider

    async def aclose(self) -> None:
        for provider in self._instances.values():
            try:
                await provider.aclose()
            except Exception:  # noqa: S110 - best-effort shutdown
                pass
