"""Approval brokers: how ASK decisions reach a human."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..core.events import EventBus, EventType
from ..core.ids import new_id
from ..core.util import utcnow_iso


class ApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: new_id("apr"))
    created: str = Field(default_factory=utcnow_iso)
    tool: str
    summary: str
    reason: str
    risk: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    task_id: str = ""

    def signature(self) -> str:
        return f"{self.tool}::{self.summary}"


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""
    remember: bool = False  # approve identical requests for the rest of the session
    by: str = ""


class ApprovalBroker(ABC):
    """Base broker with a per-session memory of remembered approvals."""

    interactive: bool = False

    def __init__(self, bus: EventBus | None = None) -> None:
        self.bus = bus
        self._remembered: set[str] = set()

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        if req.signature() in self._remembered:
            return ApprovalDecision(approved=True, reason="previously approved for this session", by="memory")
        if self.bus:
            self.bus.emit(
                EventType.APPROVAL_REQUIRED, f"approval needed: {req.summary}",
                data={"approval_id": req.id, "tool": req.tool, "reason": req.reason, "risk": req.risk},
            )
        decision = await self._decide(req)
        if decision.approved and decision.remember:
            self._remembered.add(req.signature())
        if self.bus:
            self.bus.emit(
                EventType.APPROVAL_RESOLVED,
                f"{'approved' if decision.approved else 'denied'}: {req.summary}",
                data={"approval_id": req.id, "approved": decision.approved, "reason": decision.reason, "by": decision.by},
            )
        return decision

    @abstractmethod
    async def _decide(self, req: ApprovalRequest) -> ApprovalDecision: ...


class DenyAllBroker(ApprovalBroker):
    """Non-interactive runs: nothing that needs a human is approved."""

    async def _decide(self, req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=False, reason="no human is attached to approve this action", by="policy")


class AllowAllBroker(ApprovalBroker):
    """Approves everything. Only for tests and explicitly trusted automation."""

    async def _decide(self, req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, reason="auto-approved", by="allow-all")


class CallbackBroker(ApprovalBroker):
    interactive = True

    def __init__(self, callback: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]], bus: EventBus | None = None) -> None:
        super().__init__(bus)
        self._callback = callback

    async def _decide(self, req: ApprovalRequest) -> ApprovalDecision:
        return await self._callback(req)


class QueueBroker(ApprovalBroker):
    """Holds requests until resolved externally (web UI / CLI), with a timeout."""

    interactive = True

    def __init__(self, timeout_s: float = 900.0, bus: EventBus | None = None) -> None:
        super().__init__(bus)
        self.timeout_s = timeout_s
        self._pending: dict[str, tuple[ApprovalRequest, asyncio.Future[ApprovalDecision]]] = {}

    def pending(self) -> list[ApprovalRequest]:
        return [req for req, _ in self._pending.values()]

    def resolve(self, approval_id: str, approved: bool, reason: str = "", remember: bool = False, by: str = "user") -> bool:
        entry = self._pending.get(approval_id)
        if entry is None or entry[1].done():
            return False
        entry[1].set_result(ApprovalDecision(approved=approved, reason=reason, remember=remember, by=by))
        return True

    def deny_all(self, reason: str = "stopped") -> None:
        for approval_id in list(self._pending):
            self.resolve(approval_id, False, reason, by="system")

    async def _decide(self, req: ApprovalRequest) -> ApprovalDecision:
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self._pending[req.id] = (req, future)
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_s)
        except TimeoutError:
            return ApprovalDecision(approved=False, reason=f"no response within {self.timeout_s:.0f}s", by="timeout")
        finally:
            self._pending.pop(req.id, None)
            if not future.done():
                future.cancel()
