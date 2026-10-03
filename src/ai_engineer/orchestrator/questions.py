"""How the agent asks the user genuinely blocking questions."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from ..core.ids import new_id


class QuestionBroker(ABC):
    @abstractmethod
    async def ask(self, questions: list[str], context: str = "") -> list[str] | None:
        """Return one answer per question, or None if no answer is available."""


class NoQuestions(QuestionBroker):
    async def ask(self, questions: list[str], context: str = "") -> list[str] | None:
        return None


class CallbackQuestionBroker(QuestionBroker):
    def __init__(self, callback: Callable[[list[str], str], Awaitable[list[str] | None]]) -> None:
        self._callback = callback

    async def ask(self, questions: list[str], context: str = "") -> list[str] | None:
        return await self._callback(questions, context)


class PendingQuestions(BaseModel):
    id: str = Field(default_factory=lambda: new_id("q"))
    questions: list[str]
    context: str = ""


class QueueQuestionBroker(QuestionBroker):
    """Questions are answered from outside (web UI) within a timeout."""

    def __init__(self, timeout_s: float = 1800.0) -> None:
        self.timeout_s = timeout_s
        self._pending: dict[str, tuple[PendingQuestions, asyncio.Future[list[str]]]] = {}

    def pending(self) -> list[PendingQuestions]:
        return [p for p, _ in self._pending.values()]

    def answer(self, qid: str, answers: list[str]) -> bool:
        entry = self._pending.get(qid)
        if entry is None or entry[1].done():
            return False
        entry[1].set_result(answers)
        return True

    async def ask(self, questions: list[str], context: str = "") -> list[str] | None:
        item = PendingQuestions(questions=questions, context=context)
        fut: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
        self._pending[item.id] = (item, fut)
        try:
            return await asyncio.wait_for(fut, timeout=self.timeout_s)
        except TimeoutError:
            return None
        finally:
            self._pending.pop(item.id, None)
