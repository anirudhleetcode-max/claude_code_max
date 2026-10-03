"""Cooperative cancellation tokens."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .errors import CancelledByUser

T = TypeVar("T")


class CancellationToken:
    """A cancellation signal that can be checked, awaited, and chained.

    Cancelling a parent cancels all children; cancelling a child does not
    affect the parent.
    """

    def __init__(self, parent: CancellationToken | None = None) -> None:
        self._cancelled = False
        self.reason: str = ""
        self._callbacks: list[Callable[[str], None]] = []
        self._waiters: list[asyncio.Event] = []
        if parent is not None:
            parent.on_cancel(self.cancel)
            if parent.cancelled:
                self.cancel(parent.reason)

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self, reason: str = "cancelled") -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self.reason = reason
        for event in self._waiters:
            event.set()
        for cb in list(self._callbacks):
            try:
                cb(reason)
            except Exception:  # noqa: S110 - a misbehaving callback must not block cancellation
                pass

    def on_cancel(self, callback: Callable[[str], None]) -> None:
        if self._cancelled:
            callback(self.reason)
        else:
            self._callbacks.append(callback)

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledByUser(self.reason or "cancelled")

    def child(self) -> CancellationToken:
        return CancellationToken(parent=self)

    async def wait(self) -> str:
        if self._cancelled:
            return self.reason
        event = asyncio.Event()
        self._waiters.append(event)
        try:
            await event.wait()
        finally:
            self._waiters.remove(event)
        return self.reason


async def run_cancellable(coro_factory: Callable[[], Awaitable[T]], token: CancellationToken) -> T:
    """Run an awaitable, cancelling it promptly if ``token`` fires.

    Raises :class:`CancelledByUser` when the token wins.
    """
    token.raise_if_cancelled()
    task: asyncio.Future[T] = asyncio.ensure_future(coro_factory())
    waiter = asyncio.ensure_future(token.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: S110 - we are discarding the cancelled task
            pass
        raise CancelledByUser(token.reason or "cancelled")
    finally:
        waiter.cancel()
