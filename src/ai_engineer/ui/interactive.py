"""Interactive terminal prompts for approvals and clarifying questions."""

from __future__ import annotations

import asyncio
import sys

from ..orchestrator.questions import CallbackQuestionBroker
from ..tools.approval import ApprovalDecision, ApprovalRequest, CallbackBroker

_LOCK = asyncio.Lock()


async def _input(prompt: str) -> str:
    try:
        return await asyncio.to_thread(input, prompt)
    except EOFError:
        return ""


async def prompt_approval(req: ApprovalRequest) -> ApprovalDecision:
    async with _LOCK:
        print(file=sys.stderr)
        print(f"  ⚠ Approval required ({req.risk or 'policy'}): {req.summary}", file=sys.stderr)
        print(f"    Reason: {req.reason}", file=sys.stderr)
        reasons = req.details.get("risk_reasons") if isinstance(req.details, dict) else None
        if reasons:
            print(f"    Risk factors: {'; '.join(reasons)}", file=sys.stderr)
        while True:
            answer = (await _input("    Approve? [y]es / [n]o / [a]lways this session: ")).strip().lower()
            if answer in ("y", "yes"):
                return ApprovalDecision(approved=True, reason="approved by user", by="user")
            if answer in ("a", "always"):
                return ApprovalDecision(approved=True, reason="approved by user for this session", remember=True, by="user")
            if answer in ("n", "no", ""):
                note = (await _input("    Optional note for the agent (why / what to do instead): ")).strip()
                return ApprovalDecision(approved=False, reason=note or "declined by user", by="user")


async def prompt_questions(questions: list[str], context: str) -> list[str] | None:
    async with _LOCK:
        print(file=sys.stderr)
        print(f"  ? The agent needs clarification{f' ({context})' if context else ''}:", file=sys.stderr)
        answers = []
        for q in questions:
            answers.append((await _input(f"    {q}\n    > ")).strip())
        if not any(answers):
            return None
        return answers


def terminal_approvals() -> CallbackBroker:
    return CallbackBroker(prompt_approval)


def terminal_questions() -> CallbackQuestionBroker:
    return CallbackQuestionBroker(prompt_questions)
