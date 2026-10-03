"""Role-based routing with retries, circuit breakers and fallback chains."""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from ..config.settings import ModelOptions, ModelsSettings
from ..core.cancel import CancellationToken, run_cancellable
from ..core.errors import (
    AllModelsFailedError,
    ConfigError,
    MalformedOutputError,
    ProviderError,
    RateLimitError,
)
from ..core.events import EventBus, EventType
from ..core.types import StopReason
from .base import ModelProvider, ModelRequest, ModelResponse, StructuredResult
from .registry import ModelRef, ProviderRegistry

R = TypeVar("R")

SleepFn = Callable[[float], Awaitable[None]]


@dataclass
class _BreakerState:
    failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """Skip a model after repeated failures; allow a trial call after a cool-down."""

    def __init__(self, threshold: int, reset_after_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold = threshold
        self.reset_after_s = reset_after_s
        self._clock = clock
        self._states: dict[str, _BreakerState] = {}

    def allow(self, key: str) -> bool:
        state = self._states.get(key)
        if state is None or state.opened_at is None:
            return True
        return self._clock() - state.opened_at >= self.reset_after_s  # half-open trial

    def is_open(self, key: str) -> bool:
        return not self.allow(key)

    def record_success(self, key: str) -> None:
        self._states[key] = _BreakerState()

    def record_failure(self, key: str) -> None:
        state = self._states.setdefault(key, _BreakerState())
        state.failures += 1
        if state.failures >= self.threshold:
            state.opened_at = self._clock()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {k: {"failures": v.failures, "open": self.is_open(k)} for k, v in self._states.items()}


@dataclass
class CallStats:
    calls: int = 0
    failures: int = 0
    retries: int = 0
    fallbacks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_tokens: bool = False
    latency_s: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, ref: str, resp: ModelResponse) -> None:
        self.calls += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        self.estimated_tokens = self.estimated_tokens or resp.usage.estimated
        self.latency_s += resp.latency_s
        entry = self.by_model.setdefault(ref, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0})
        entry["calls"] += 1
        entry["input_tokens"] += resp.usage.input_tokens
        entry["output_tokens"] += resp.usage.output_tokens
        entry["latency_s"] += resp.latency_s


class FallbackModel:
    """A model handle for one role: tries each configured model in order."""

    def __init__(
        self,
        role: str,
        chain: list[ModelRef],
        registry: ProviderRegistry,
        settings: ModelsSettings,
        breaker: CircuitBreaker,
        bus: EventBus | None = None,
        stats: CallStats | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        if not chain:
            raise ConfigError(
                f"no model configured for role '{role}'. Set AIE_MODEL=provider:model or "
                "[models.roles] in .agent/config.toml (see `aie providers models`)."
            )
        self.role = role
        self.chain = chain
        self.registry = registry
        self.settings = settings
        self.breaker = breaker
        self.bus = bus
        self.stats = stats or CallStats()
        self._sleep = sleep
        self.last_ref: ModelRef | None = None

    # ---- helpers ------------------------------------------------------------------

    def primary_options(self) -> ModelOptions:
        for ref in self.chain:
            if self.breaker.allow(str(ref)):
                try:
                    return self.registry.get(ref.provider).options_for(ref.model)
                except ConfigError:
                    continue
        ref = self.chain[0]
        return self.registry.get(ref.provider).options_for(ref.model)

    def describe(self) -> str:
        return " → ".join(str(r) for r in self.chain)

    def _emit(self, etype: EventType, message: str, **data: Any) -> None:
        if self.bus is not None:
            self.bus.emit(etype, message, data={"role": self.role, **data})

    async def _sleep_cancellable(self, delay: float, cancel: CancellationToken | None) -> None:
        if cancel is None:
            await self._sleep(delay)
            return
        cancel.raise_if_cancelled()
        sleeper = asyncio.ensure_future(self._sleep(delay))
        waiter = asyncio.ensure_future(cancel.wait())
        try:
            await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            sleeper.cancel()
            waiter.cancel()
        cancel.raise_if_cancelled()

    def _backoff(self, attempt: int, error: ProviderError) -> float:
        cfg = self.settings.retry
        delay = min(cfg.max_delay_s, cfg.base_delay_s * (2**attempt))
        if isinstance(error, RateLimitError) and error.retry_after:
            delay = min(max(delay, error.retry_after), cfg.max_delay_s * 2)
        jitter = delay * cfg.jitter
        return max(0.0, delay + random.uniform(-jitter, jitter))  # noqa: S311 - not security sensitive

    async def _with_fallback(
        self,
        op: Callable[[ModelProvider, ModelRef], Awaitable[R]],
        cancel: CancellationToken | None,
        is_refusal: Callable[[R], bool] | None = None,
    ) -> R:
        attempts: list[str] = []
        candidates = [r for r in self.chain if self.breaker.allow(str(r))] or list(self.chain)
        last_refusal: R | None = None
        for index, ref in enumerate(candidates):
            key = str(ref)
            try:
                provider = self.registry.get(ref.provider)
            except ConfigError as exc:
                attempts.append(f"{key}: {exc}")
                continue
            for attempt in range(self.settings.retry.max_attempts):
                try:
                    if cancel is None:
                        result = await op(provider, ref)
                    else:  # a stop/cancel interrupts the in-flight request instead of waiting for it
                        result = await run_cancellable(functools.partial(op, provider, ref), cancel)
                except ProviderError as exc:
                    self.stats.failures += 1
                    if exc.retryable and attempt + 1 < self.settings.retry.max_attempts:
                        delay = self._backoff(attempt, exc)
                        self.stats.retries += 1
                        self._emit(
                            EventType.MODEL_RETRY,
                            f"{key} failed ({type(exc).__name__}); retrying in {delay:.1f}s",
                            model=key, attempt=attempt + 1, error=str(exc)[:300],
                        )
                        await self._sleep_cancellable(delay, cancel)
                        continue
                    self.breaker.record_failure(key)
                    attempts.append(f"{key}: {type(exc).__name__}: {str(exc)[:200]}")
                    if not exc.fallback:
                        raise
                    break
                else:
                    if is_refusal is not None and is_refusal(result) and index + 1 < len(candidates):
                        attempts.append(f"{key}: refused")
                        last_refusal = result
                        break
                    self.breaker.record_success(key)
                    self.last_ref = ref
                    if attempts:
                        self.stats.fallbacks += 1
                    return result
            if index + 1 < len(candidates):
                nxt = candidates[index + 1]
                self._emit(
                    EventType.MODEL_FALLBACK,
                    f"switching {self.role} model {key} → {nxt}",
                    from_model=key, to_model=str(nxt), reason=attempts[-1] if attempts else "",
                )
        if last_refusal is not None:
            return last_refusal
        raise AllModelsFailedError(
            f"all models failed for role '{self.role}': " + " | ".join(attempts), attempts=attempts
        )

    # ---- public API -----------------------------------------------------------------

    async def generate(self, req: ModelRequest, cancel: CancellationToken | None = None) -> ModelResponse:
        async def op(provider: ModelProvider, ref: ModelRef) -> ModelResponse:
            self._emit(EventType.MODEL_CALLED, f"calling {ref} ({self.role})", model=str(ref))
            call = provider.generate(req.model_copy(update={"model": ref.model}))
            timeout = req.timeout_s or self.settings.request_timeout_s
            try:
                resp = await asyncio.wait_for(call, timeout=timeout)
            except TimeoutError as exc:
                from ..core.errors import RetryableProviderError

                raise RetryableProviderError(
                    f"request timed out after {timeout:.0f}s", provider=ref.provider, model=ref.model
                ) from exc
            self.stats.record(str(ref), resp)
            self._emit(
                EventType.MODEL_RESULT,
                f"{ref} responded ({resp.stop_reason}, {resp.latency_s:.1f}s)",
                model=str(ref), stop_reason=str(resp.stop_reason), latency_s=round(resp.latency_s, 3),
                input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                estimated=resp.usage.estimated, tool_calls=len(resp.tool_uses()),
            )
            return resp

        return await self._with_fallback(op, cancel, lambda r: r.stop_reason == StopReason.REFUSAL)

    async def structured_output(
        self,
        req: ModelRequest,
        schema: type[BaseModel],
        cancel: CancellationToken | None = None,
        max_repairs: int = 2,
    ) -> StructuredResult:
        from .structured import structured_generate

        async def op(provider: ModelProvider, ref: ModelRef) -> StructuredResult:
            async def gen(r: ModelRequest) -> ModelResponse:
                self._emit(EventType.MODEL_CALLED, f"calling {ref} ({self.role}, structured)", model=str(ref))
                resp = await asyncio.wait_for(
                    provider.generate(r.model_copy(update={"model": ref.model})),
                    timeout=req.timeout_s or self.settings.request_timeout_s,
                )
                self.stats.record(str(ref), resp)
                self._emit(
                    EventType.MODEL_RESULT, f"{ref} responded ({resp.latency_s:.1f}s)",
                    model=str(ref), latency_s=round(resp.latency_s, 3),
                    input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                    estimated=resp.usage.estimated,
                )
                return resp

            try:
                return await structured_generate(gen, req, schema, provider.options_for(ref.model), max_repairs)
            except TimeoutError as exc:
                from ..core.errors import RetryableProviderError

                raise RetryableProviderError("request timed out", provider=ref.provider, model=ref.model) from exc
            except MalformedOutputError as exc:
                exc.provider, exc.model = ref.provider, ref.model
                raise

        return await self._with_fallback(op, cancel)

    async def embeddings(self, texts: list[str], cancel: CancellationToken | None = None) -> list[list[float]]:
        async def op(provider: ModelProvider, ref: ModelRef) -> list[list[float]]:
            return await provider.embeddings(texts, ref.model)

        return await self._with_fallback(op, cancel)


class ModelRouter:
    """Creates :class:`FallbackModel` handles per role from configuration."""

    def __init__(
        self,
        settings: ModelsSettings,
        registry: ProviderRegistry | None = None,
        bus: EventBus | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.registry = registry or ProviderRegistry(settings)
        self.bus = bus
        self.breaker = CircuitBreaker(settings.circuit_breaker.failure_threshold, settings.circuit_breaker.reset_after_s)
        self.stats = CallStats()
        self._sleep = sleep
        self._handles: dict[str, FallbackModel] = {}

    def has_role(self, role: str) -> bool:
        return bool(self.settings.chain_for(role))

    def for_role(self, role: str) -> FallbackModel:
        if role not in self._handles:
            chain = [ModelRef.parse(r) for r in self.settings.chain_for(role)]
            self._handles[role] = FallbackModel(
                role, chain, self.registry, self.settings, self.breaker, self.bus, self.stats, self._sleep
            )
        return self._handles[role]

    def configured_roles(self) -> dict[str, list[str]]:
        from ..config.settings import ROLE_NAMES

        return {role: self.settings.chain_for(role) for role in ROLE_NAMES if self.settings.chain_for(role)}

    async def aclose(self) -> None:
        await self.registry.aclose()
