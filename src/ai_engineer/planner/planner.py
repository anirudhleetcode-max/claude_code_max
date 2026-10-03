"""Understanding (triage) and planning stages."""

from __future__ import annotations

import re
from typing import Any

from ..core.errors import AllModelsFailedError, MalformedOutputError
from ..core.types import Message
from ..models.base import ModelRequest
from .models import Needs, Plan, Subtask, TaskUnderstanding, validate_plan

_CHANGE_VERBS = re.compile(
    r"\b(add|implement|create|build|fix|refactor|rename|remove|delete|update|upgrade|migrate|write|change|modify|replace|make|convert|support|introduce|optimi[sz]e|improve|generate|set up|setup|configure|port)\b",
    re.I,
)
_SECURITY = re.compile(r"\b(auth\w*|login|password|passwd|token|jwt|oauth|session|cookie|crypt\w*|secret|permission|role|acl|sanitiz\w*|inject\w*|xss|csrf|ssrf|vulnerab\w*|cve|encrypt\w*|hash\w*|tls|ssl|cors)\b", re.I)
_PERF = re.compile(r"\b(perf\w*|slow|fast(er)?|latency|throughput|optimi[sz]\w*|memory|cpu|cache|caching|n\+1|scal\w*|benchmark)\b", re.I)
_DOCS = re.compile(r"\b(readme|docs?|documentation|cli|command[- ]line|api|endpoint|config\w*|option|flag)\b", re.I)


_REQUEST_PREFIX = re.compile(r"^\s*(can|could|would|will)\s+you\s+(please\s+)?|^\s*please\s+", re.I)
_INFO_START = re.compile(r"^\s*(what|where|which|who|why|how|when|explain|describe|summari[sz]e|tell me|show me|is|are|does|do)\b", re.I)


def looks_like_question(line: str) -> bool:
    """'How do I add X?' is a question; 'Can you add X?' and 'Add X' are change requests."""
    request = _REQUEST_PREFIX.match(line)
    if request:
        return not _CHANGE_VERBS.match(line[request.end():])
    if _INFO_START.match(line):
        return True
    return line.rstrip().endswith("?") and not _CHANGE_VERBS.search(line)


def heuristic_understanding(task: str) -> TaskUnderstanding:
    """Deterministic fallback used when no model can produce an understanding."""
    first_line = task.strip().splitlines()[0] if task.strip() else task
    is_question = looks_like_question(first_line)
    words = len(task.split())
    complexity = "small" if words < 40 else ("medium" if words < 150 else "large")
    return TaskUnderstanding(
        summary=first_line[:300],
        task_type="question" if is_question else "change",
        complexity=complexity,  # type: ignore[arg-type]
        requirements=[task.strip()[:2000]],
        acceptance_criteria=[] if is_question else ["the requested change is implemented and existing tests still pass"],
        assumptions=["understanding derived heuristically because no model response was available"],
        needs=Needs(
            security_review=bool(_SECURITY.search(task)),
            performance_review=bool(_PERF.search(task)),
            docs_update=bool(_DOCS.search(task)) and not is_question,
            tests=not is_question,
        ),
    )


def apply_routing_heuristics(u: TaskUnderstanding, task: str) -> TaskUnderstanding:
    """Deterministic signals can only *add* review stages, never remove ones the model requested."""
    needs = u.needs.model_copy()
    if _SECURITY.search(task):
        needs.security_review = True
    if _PERF.search(task):
        needs.performance_review = True
    if u.task_type == "question":
        needs.tests = False
        needs.docs_update = False
    return u.model_copy(update={"needs": needs})


UNDERSTAND_SYSTEM = """You are the lead engineer triaging a software task for an autonomous engineering agent.
Restate the task precisely, classify it, extract requirements and objectively checkable acceptance
criteria, list assumptions, and decide which extra review stages are needed.

Rules:
- task_type "question" only when the user wants information/explanation and no repository change.
- complexity: trivial (one obvious edit), small (one focused change), medium (several files or steps),
  large (multiple components, new subsystems).
- blocking_questions: ONLY when the task cannot be done correctly without the answer (e.g. two
  incompatible interpretations with different outcomes, missing credentials). Prefer reasonable,
  stated assumptions. Most tasks have no blocking questions.
- acceptance_criteria must be verifiable by tests, commands or inspection."""


async def understand(model: Any, task: str, project_brief: str, cancel: Any = None) -> tuple[TaskUnderstanding, str]:
    """Returns (understanding, source) where source is 'model' or 'heuristic'."""
    content = f"## Task\n{task}\n\n## Repository\n{project_brief}"
    req = ModelRequest(messages=[Message.user(content)], system=UNDERSTAND_SYSTEM, metadata={"role": "classifier", "stage": "understand"})
    try:
        result = await model.structured_output(req, TaskUnderstanding, cancel=cancel)
        return apply_routing_heuristics(result.value, task), "model"
    except (AllModelsFailedError, MalformedOutputError):
        return heuristic_understanding(task), "heuristic"


PLAN_SYSTEM = """You are a principal engineer planning work for an autonomous coding agent that will
execute your plan step by step, running tests after every step.

Produce a plan of independently verifiable subtasks:
- Each subtask must leave the repository in a working state and have concrete acceptance criteria.
- Order with depends_on; keep the plan as small as the task allows (merge trivial steps).
- Put tests in the same subtask as the code they verify unless a separate test subtask is clearer.
- Prefer existing project conventions, frameworks and libraries found in the repository; do not
  invent dependencies or APIs. Note significant design decisions with their rationale.
- Use files_hint for files you expect to touch (only real paths or clearly new files)."""


def single_subtask_plan(u: TaskUnderstanding) -> Plan:
    return Plan(
        goal=u.summary,
        approach="Single focused change.",
        subtasks=[
            Subtask(
                id="s1",
                title=u.summary[:120],
                description="\n".join(u.requirements) or u.summary,
                acceptance_criteria=u.acceptance_criteria,
                files_hint=u.relevant_paths,
                kind="implement",
            )
        ],
    )


async def make_plan(
    model: Any, task: str, u: TaskUnderstanding, project_brief: str, context: str, max_subtasks: int, cancel: Any = None
) -> tuple[Plan, list[Subtask], str]:
    """Returns (plan, ordered_subtasks, source). Falls back to a single-subtask plan."""
    if u.complexity in ("trivial", "small"):
        plan = single_subtask_plan(u)
        return plan, plan.subtasks, "direct"
    content = (
        f"## Task\n{task}\n\n## Understanding\nSummary: {u.summary}\nRequirements:\n"
        + "\n".join(f"- {r}" for r in u.requirements)
        + "\nAcceptance criteria:\n"
        + "\n".join(f"- {c}" for c in u.acceptance_criteria)
        + (("\nAssumptions:\n" + "\n".join(f"- {a}" for a in u.assumptions)) if u.assumptions else "")
        + f"\n\n## Repository\n{project_brief}\n\n## Relevant context\n{context}\n\nMaximum subtasks: {max_subtasks}"
    )
    messages = [Message.user(content)]
    for _ in range(2):
        req = ModelRequest(messages=messages, system=PLAN_SYSTEM, metadata={"role": "planner", "stage": "plan"})
        try:
            result = await model.structured_output(req, Plan, cancel=cancel)
        except (AllModelsFailedError, MalformedOutputError):
            break
        ordered, problems = validate_plan(result.value, max_subtasks)
        if not problems:
            return result.value, ordered, "model"
        messages = [
            *messages,
            Message.assistant(result.value.model_dump_json()),
            Message.user("The plan is invalid:\n" + "\n".join(f"- {p}" for p in problems) + "\nReturn a corrected plan."),
        ]
    plan = single_subtask_plan(u)
    return plan, plan.subtasks, "fallback"
