"""Stage-control tools: how a model signals that its loop is finished.

The agent loop intercepts these calls; their claims are recorded but never
trusted — the orchestrator verifies changes and test results independently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..tools.base import SideEffect, Tool, ToolContext, ToolInput, ToolResult


class SubmitWorkInput(ToolInput):
    summary: str = Field(min_length=5, description="What you changed and why")
    files_changed: list[str] = Field(default_factory=list)
    verification: str = Field(default="", description="Commands/tests you ran and their actual results")
    unresolved: str = Field(default="", description="Anything incomplete, unverified, or uncertain")
    blocked: bool = Field(default=False, description="True if you could not complete the work (explain in unresolved)")


class SubmitWorkTool(Tool):
    name = "submit_work"
    description = (
        "Call when the subtask is complete and you have verified it (or you are blocked). Be honest: report "
        "exactly what you ran and what is still unverified. Your claims will be independently checked."
    )
    Input = SubmitWorkInput
    side_effect = SideEffect.NONE

    async def run(self, args: SubmitWorkInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(content="submission received")


class SubmitAnswerInput(ToolInput):
    answer: str = Field(min_length=1, description="Direct, well-structured answer in Markdown")
    evidence: list[str] = Field(default_factory=list, description="path:line locations supporting the answer")
    confidence: Literal["high", "medium", "low"] = "medium"


class SubmitAnswerTool(Tool):
    name = "submit_answer"
    description = "Call once you can answer the question from evidence in the repository."
    Input = SubmitAnswerInput
    side_effect = SideEffect.NONE

    async def run(self, args: SubmitAnswerInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(content="answer received")


CONTROL_TOOLS: list[type[Tool]] = [SubmitWorkTool, SubmitAnswerTool]
