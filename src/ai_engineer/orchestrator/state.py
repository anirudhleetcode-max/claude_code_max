"""Persisted pipeline state (serialized into the task record after every stage transition)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

STAGES = ("understand", "inspect", "plan", "execute", "final_qa", "report", "done")


class SubtaskState(BaseModel):
    id: str
    title: str
    status: str = "pending"  # pending | running | completed | completed_unverified | failed | blocked | skipped
    attempts: int = 0
    checkpoint_before: str | None = None
    checkpoint_after: str | None = None
    files_changed: list[str] = Field(default_factory=list)
    submission: dict[str, Any] | None = None
    loop_status: str | None = None
    validation: list[dict[str, Any]] = Field(default_factory=list)
    review: dict[str, Any] | None = None
    gates: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = Field(default_factory=list)
    repair_iterations: int = 0
    review_iterations: int = 0
    commit: str | None = None
    notes: list[str] = Field(default_factory=list)


class PipelineState(BaseModel):
    stage: str = "understand"
    understanding: dict[str, Any] | None = None
    understanding_source: str = ""
    clarifications: list[dict[str, str]] = Field(default_factory=list)
    # blocking questions awaiting an answer (on_questions = "block"), and answers supplied on resume
    pending_questions: list[str] = Field(default_factory=list)
    supplied_answers: list[str] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    plan_source: str = ""
    order: list[str] = Field(default_factory=list)
    subtasks: dict[str, SubtaskState] = Field(default_factory=dict)
    baseline: dict[str, dict[str, Any]] = Field(default_factory=dict)
    start_checkpoint: str | None = None
    started_clean: bool | None = None
    auto_commit: bool = False
    branch: str | None = None
    original_branch: str | None = None
    answer: dict[str, Any] | None = None
    final_validation: list[dict[str, Any]] = Field(default_factory=list)
    final_review: dict[str, Any] | None = None
    security_findings: list[dict[str, Any]] = Field(default_factory=list)
    gates: dict[str, Any] | None = None
    docs: dict[str, str] | None = None
    git_state: dict[str, Any] | None = None
    report_path: str | None = None
    resumed: int = 0
    notes: list[str] = Field(default_factory=list)
