from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from ai_engineer.core.cancel import CancellationToken
from ai_engineer.core.errors import (
    AllModelsFailedError,
    CancelledByUser,
    ConfigError,
    ContextLengthError,
    MalformedOutputError,
)
from ai_engineer.core.events import EventBus, EventType
from ai_engineer.core.types import Message, StopReason
from ai_engineer.models.base import ModelRequest
from ai_engineer.models.registry import ModelRef
from ai_engineer.models.router import CircuitBreaker
from ai_engineer.providers.scripted import ScriptedProvider
from tests.conftest import make_router


def req(text: str = "hi") -> ModelRequest:
    return ModelRequest(messages=[Message.user(text)])


def test_model_ref_parse_keeps_colons_in_model() -> None:
    ref = ModelRef.parse("ollama:qwen2.5:7b")
    assert ref.provider == "ollama" and ref.model == "qwen2.5:7b"
    with pytest.raises(ConfigError):
        ModelRef.parse("no-colon")


async def test_retries_transient_errors_then_succeeds() -> None:
    p = ScriptedProvider("a", steps=[{"raise": {"type": "rate_limit"}}, {"raise": {"type": "unavailable"}}, "ok"])
    router = make_router(p)
    resp = await router.for_role("default").generate(req())
    assert resp.text() == "ok"
    assert router.stats.retries == 2
    assert len(p.requests) == 3


async def test_falls_back_to_next_model_after_retries_exhausted() -> None:
    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    a = ScriptedProvider("a", steps=[{"raise": {"type": "retryable"}}] * 3)
    b = ScriptedProvider("b", steps=["from b"])
    router = make_router(a, b, roles={"default": ["a:m1", "b:m2"]})
    router.bus = bus
    model = router.for_role("default")
    model.bus = bus
    resp = await model.generate(req())
    assert resp.text() == "from b"
    assert resp.provider == "b"
    assert any(e.type == EventType.MODEL_FALLBACK for e in events)
    assert str(model.last_ref) == "b:m2"


async def test_auth_error_falls_back_without_retry() -> None:
    a = ScriptedProvider("a", steps=[{"raise": {"type": "auth"}}])
    b = ScriptedProvider("b", steps=["fine"])
    router = make_router(a, b, roles={"default": ["a:x", "b:y"]})
    resp = await router.for_role("default").generate(req())
    assert resp.text() == "fine"
    assert len(a.requests) == 1


async def test_context_length_error_is_not_masked_by_fallback() -> None:
    a = ScriptedProvider("a", steps=[{"raise": {"type": "context"}}])
    b = ScriptedProvider("b", steps=["should not be used"])
    router = make_router(a, b, roles={"default": ["a:x", "b:y"]})
    with pytest.raises(ContextLengthError):
        await router.for_role("default").generate(req())
    assert b.requests == []


async def test_all_models_failed() -> None:
    a = ScriptedProvider("a", steps=[{"raise": {"type": "invalid"}}])
    b = ScriptedProvider("b", steps=[{"raise": {"type": "auth"}}])
    router = make_router(a, b, roles={"default": ["a:x", "b:y"]})
    with pytest.raises(AllModelsFailedError) as info:
        await router.for_role("default").generate(req())
    assert len(info.value.attempts) == 2


async def test_refusal_falls_back_but_last_refusal_is_returned() -> None:
    a = ScriptedProvider("a", steps=[{"text": "", "stop_reason": "refusal"}])
    b = ScriptedProvider("b", steps=["answer"])
    router = make_router(a, b, roles={"default": ["a:x", "b:y"]})
    assert (await router.for_role("default").generate(req())).text() == "answer"

    c = ScriptedProvider("c", steps=[{"text": "", "stop_reason": "refusal"}])
    router2 = make_router(c)
    resp = await router2.for_role("default").generate(req())
    assert resp.stop_reason == StopReason.REFUSAL


async def test_circuit_breaker_skips_open_model() -> None:
    a = ScriptedProvider("a", steps=[{"raise": {"type": "auth"}}] * 5)
    b = ScriptedProvider("b", steps=["1", "2", "3"])
    router = make_router(a, b, roles={"default": ["a:x", "b:y"]})
    router.breaker.threshold = 1
    model = router.for_role("default")
    await model.generate(req())
    await model.generate(req())
    assert len(a.requests) == 1  # second call skipped the open breaker
    assert router.breaker.is_open("a:x")


def test_circuit_breaker_half_open_after_cooldown() -> None:
    now = [0.0]
    breaker = CircuitBreaker(2, 10.0, clock=lambda: now[0])
    breaker.record_failure("k")
    assert breaker.allow("k")
    breaker.record_failure("k")
    assert not breaker.allow("k")
    now[0] = 11.0
    assert breaker.allow("k")
    breaker.record_success("k")
    assert breaker.allow("k")


async def test_request_timeout_is_retryable_and_bounded() -> None:
    a = ScriptedProvider("a", steps=[{"delay_s": 5, "text": "slow"}] * 3)
    b = ScriptedProvider("b", steps=["fast"])
    router = make_router(a, b, roles={"default": ["a:x", "b:y"]})
    resp = await router.for_role("default").generate(req().model_copy(update={"timeout_s": 0.05}))
    assert resp.text() == "fast"


async def test_cancellation_during_backoff() -> None:
    a = ScriptedProvider("a", steps=[{"raise": {"type": "retryable"}}] * 3)
    router = make_router(a)

    async def slow_sleep(_: float) -> None:
        await asyncio.sleep(10)

    router.for_role("default")._sleep = slow_sleep
    token = CancellationToken()
    task = asyncio.create_task(router.for_role("default").generate(req(), cancel=token))
    await asyncio.sleep(0.05)
    token.cancel("stop")
    with pytest.raises(CancelledByUser):
        await asyncio.wait_for(task, 2)



async def test_cancellation_interrupts_in_flight_request() -> None:
    # Regression: the cancel token was only checked between attempts, so a stop waited for the
    # whole model call (up to its timeout) to finish.
    a = ScriptedProvider("a", steps=[{"delay_s": 30, "text": "late"}])
    router = make_router(a)
    token = CancellationToken()
    task = asyncio.create_task(router.for_role("default").generate(req(), cancel=token))
    await asyncio.sleep(0.05)
    started = asyncio.get_running_loop().time()
    token.cancel("stop")
    with pytest.raises(CancelledByUser):
        await asyncio.wait_for(task, 2)
    assert asyncio.get_running_loop().time() - started < 1.0
    assert router.for_role("default").breaker.allow("a:x")  # a cancel is not a model failure

class Verdict(BaseModel):
    ok: bool
    reason: str


async def test_structured_output_repairs_invalid_json() -> None:
    a = ScriptedProvider("a", steps=["not json", '{"ok": "maybe"}', 'Sure: ```json\n{"ok": true, "reason": "x"}\n```'])
    router = make_router(a)
    result = await router.for_role("default").structured_output(req(), Verdict)
    assert result.value == Verdict(ok=True, reason="x")
    assert result.repairs == 2
    # The repair prompt shows the validation problem to the model.
    assert "Problems" in a.requests[1].messages[-1].text()


async def test_structured_output_falls_back_when_unrepairable() -> None:
    a = ScriptedProvider("a", steps=["junk", "junk", "junk"])
    b = ScriptedProvider("b", steps=['{"ok": false, "reason": "fallback"}'])
    router = make_router(a, b, roles={"default": ["a:x", "b:y"]})
    result = await router.for_role("default").structured_output(req(), Verdict)
    assert result.value.reason == "fallback"

    c = ScriptedProvider("c", steps=["junk"] * 3)
    with pytest.raises((MalformedOutputError, AllModelsFailedError)):
        await make_router(c).for_role("default").structured_output(req(), Verdict)


def test_missing_role_configuration_is_explained() -> None:
    a = ScriptedProvider("a")
    router = make_router(a, roles={})
    with pytest.raises(ConfigError, match="AIE_MODEL"):
        router.for_role("default")


def test_role_chain_inherits_from_default() -> None:
    a = ScriptedProvider("a")
    router = make_router(a, roles={"default": ["a:x"], "reviewer": ["a:r"]})
    assert router.settings.chain_for("planner") == ["a:x"]
    assert router.settings.chain_for("reviewer") == ["a:r"]
    assert router.settings.chain_for("classifier") == ["a:x"]
