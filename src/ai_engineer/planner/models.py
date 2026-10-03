"""Schemas for task understanding and plans."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Needs(BaseModel):
    research: bool = Field(default=False, description="external documentation must be consulted")
    security_review: bool = Field(default=False, description="touches auth, secrets, crypto, input handling or permissions")
    performance_review: bool = Field(default=False, description="performance is an explicit concern")
    docs_update: bool = Field(default=False, description="user-facing behaviour, CLI, API or configuration changes")
    tests: bool = Field(default=True, description="tests should be added or updated")


class TaskUnderstanding(BaseModel):
    summary: str = Field(description="one or two sentences restating the task precisely")
    task_type: Literal["question", "change"] = Field(description="'question' = answer/explain only, no code changes; 'change' = modify the repository")
    complexity: Literal["trivial", "small", "medium", "large"]
    requirements: list[str] = Field(default_factory=list, description="explicit and clearly implied requirements")
    acceptance_criteria: list[str] = Field(default_factory=list, description="objectively checkable conditions for done")
    assumptions: list[str] = Field(default_factory=list)
    blocking_questions: list[str] = Field(default_factory=list, description="ONLY questions that make the task impossible to do correctly without an answer")
    needs: Needs = Field(default_factory=Needs)
    risk_areas: list[str] = Field(default_factory=list)
    relevant_paths: list[str] = Field(default_factory=list, description="files/directories likely involved, if known")

    @field_validator("blocking_questions")
    @classmethod
    def _limit_questions(cls, v: list[str]) -> list[str]:
        return [q for q in v if q.strip()][:5]


SubtaskKind = Literal["implement", "test", "docs", "refactor", "config", "research", "investigate"]


class Subtask(BaseModel):
    id: str = Field(description="short stable id such as 's1'")
    title: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    files_hint: list[str] = Field(default_factory=list)
    kind: SubtaskKind = "implement"
    validation: list[str] = Field(default_factory=list, description="how to verify: test files, commands or checks")


class Plan(BaseModel):
    goal: str
    approach: str = Field(description="architecture/approach in a few sentences, including key decisions")
    subtasks: list[Subtask]
    risks: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list, description="significant design decisions with a one-line rationale each")


def validate_plan(plan: Plan, max_subtasks: int) -> tuple[list[Subtask], list[str]]:
    """Return (topologically ordered subtasks, problems). Problems empty means valid."""
    problems: list[str] = []
    if not plan.subtasks:
        problems.append("plan has no subtasks")
    if len(plan.subtasks) > max_subtasks:
        problems.append(f"plan has {len(plan.subtasks)} subtasks; the maximum is {max_subtasks} — merge related work")
    ids = [s.id for s in plan.subtasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"duplicate subtask ids: {sorted(dupes)}")
    known = set(ids)
    for s in plan.subtasks:
        missing = [d for d in s.depends_on if d not in known]
        if missing:
            problems.append(f"subtask {s.id} depends on unknown ids {missing}")
        if s.id in s.depends_on:
            problems.append(f"subtask {s.id} depends on itself")
    if problems:
        return [], problems
    order: list[Subtask] = []
    by_id = {s.id: s for s in plan.subtasks}
    state: dict[str, int] = {}

    def visit(sid: str, path: list[str]) -> None:
        if state.get(sid) == 2:
            return
        if state.get(sid) == 1:
            problems.append("dependency cycle: " + " -> ".join([*path, sid]))
            return
        state[sid] = 1
        for dep in by_id[sid].depends_on:
            visit(dep, [*path, sid])
        state[sid] = 2
        order.append(by_id[sid])

    for s in plan.subtasks:
        visit(s.id, [])
    return (order if not problems else []), problems
