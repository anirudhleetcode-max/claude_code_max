"""The end-to-end engineering pipeline.

UNDERSTAND → INSPECT → PLAN → for each subtask: CHECKPOINT → IMPLEMENT → VALIDATE ⇄ REPAIR →
REVIEW ⇄ FIX → GATES → CHECKPOINT/COMMIT → FINAL QA → REPORT → RETROSPECTIVE

State is persisted after every transition. On resume nothing is assumed: an
interrupted subtask with changes on disk is re-validated before it can complete.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import traceback
from typing import TYPE_CHECKING, Any

from ..config.settings import GateMode, Mode
from ..core.cancel import CancellationToken
from ..core.errors import AllModelsFailedError, CancelledByUser, ConfigError, ToolError
from ..core.events import EventType
from ..debugger.failures import FailureTracker
from ..executor.loop import AgentLoop, LoopLimits, LoopResult
from ..gates.evaluate import GateInputs, GateReport, compare_with_baseline, evaluate_gates
from ..planner.models import Plan, Subtask, TaskUnderstanding
from ..planner.planner import make_plan, understand
from ..prompts import load_prompt
from ..reviewer.review import ReviewResult, review_change
from ..security.secrets import scan_text
from ..security.static_rules import added_lines_by_file
from ..tasks.store import Task, TaskStatus
from ..tools.approval import ApprovalRequest
from ..tools.builtin.git_tools import stage_for_commit
from .state import PipelineState, SubtaskState

if TYPE_CHECKING:
    from ..runtime import Runtime

log = logging.getLogger(__name__)

READ_TOOLS = [
    "read_file", "list_directory", "find_files", "search_text", "repo_overview", "find_symbol",
    "find_dependents", "related_tests", "code_search", "git_status", "git_diff", "git_log",
    "environment_info", "memory_search", "db_schema",
]
WRITE_TOOLS = [
    "write_file", "edit_file", "delete_file", "run_command", "process_list", "process_output", "process_stop",
    "run_tests", "run_linter", "run_formatter", "run_typecheck", "run_build", "db_query", "memory_record",
]
NETWORK_TOOLS = ["web_fetch", "web_search", "browser"]

_DOC_FILE = re.compile(r"(^|/)(readme|changelog|contributing|docs?/)|\.(md|rst|adoc)$", re.I)


class BlockedError(Exception):
    """The pipeline cannot continue without outside help (model, human, environment)."""


class _StopAfter(Exception):
    """Planned stop after a stage (e.g. `aie plan`)."""


class Orchestrator:
    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.settings = rt.settings
        self.bus = rt.bus
        self.store = rt.store
        self._stop_watch: asyncio.Task[None] | None = None
        self._started = 0.0
        self._stop_after: str | None = None

    # ------------------------------------------------------------------ utilities

    def _emit(self, etype: EventType, message: str, **data: Any) -> None:
        self.bus.emit(etype, message, data=data)

    def _save(self, task: Task, state: PipelineState, **fields: Any) -> Task:
        return self.store.update_task(task.id, state=state.model_dump(mode="json"), stage=state.stage, **fields)

    def _tools(self, *groups: list[str]) -> list[str]:
        names: list[str] = []
        for group in groups:
            names += group
        web = self.settings.web
        if "web_search" in names and web.search_backend == "none":
            names.remove("web_search")
        if "browser" in names:
            try:
                import playwright  # noqa: F401
            except ImportError:
                names.remove("browser")
        return [n for n in dict.fromkeys(names) if self.rt.registry.get(n) is not None]

    def _check_budget(self) -> None:
        if time.monotonic() - self._started > self.settings.agent.task_time_budget_s:
            raise BlockedError(f"task time budget of {self.settings.agent.task_time_budget_s:.0f}s exhausted")

    def _limits(self, model: Any) -> LoopLimits:
        opts = model.primary_options()
        budget = int(opts.context_window * self.settings.agent.context_fill_ratio) - opts.max_output_tokens
        return LoopLimits(
            max_steps=self.settings.agent.max_steps,
            max_seconds=max(60.0, self.settings.agent.task_time_budget_s - (time.monotonic() - self._started)),
            context_budget_tokens=max(8000, budget),
            repeated_call_limit=self.settings.agent.repeated_call_limit,
        )

    async def _run_loop(self, *, role: str, stage: str, prompt: str, message: str, finish: str, tools: list[str]) -> LoopResult:
        model = self.rt.router.for_role(role)
        system = load_prompt(prompt)
        brief = self.rt.context.project_brief()
        if brief:
            system += "\n\n# Repository\n" + brief
        loop = AgentLoop(
            model, self.rt.executor, self.rt.tool_ctx, role=role, stage=stage, system=system,
            tools=tools, finish_tool=finish, limits=self._limits(model),
        )
        self.bus.context["stage"] = stage
        result = await loop.run(message)
        if result.status == "model_unavailable":
            raise BlockedError(f"no model available for role '{role}': {result.error}")
        return result

    async def _watch_controls(self, task_id: str, token: CancellationToken, owner: str) -> None:
        while not token.cancelled:
            await asyncio.sleep(2.0)
            self.store.heartbeat(task_id, owner)
            request = self.store.pop_control(task_id)
            if request in ("stop", "cancel"):
                token.cancel(f"{request} requested")
                if hasattr(self.rt.approvals, "deny_all"):
                    self.rt.approvals.deny_all("task stopped")
                return

    # ------------------------------------------------------------------ entry point

    async def run(self, task_id: str, stop_after: str | None = None) -> Task:
        self._stop_after = stop_after
        task = self.store.require_task(task_id)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.COMPLETED_UNVERIFIED, TaskStatus.CANCELLED):
            raise ConfigError(f"task {task.id} is already {task.status}")
        owner = self.rt.lease_owner
        if not self.store.acquire_lease(task.id, owner):
            raise ConfigError(f"task {task.id} is being run by another process ({task.lease_owner})")
        state = PipelineState.model_validate(task.state) if task.state else PipelineState()
        resuming = bool(task.state)
        if resuming:
            state.resumed += 1
        self._started = time.monotonic()
        token = self.rt.tool_ctx.cancel
        self.rt.tool_ctx.task_id = task.id
        self.bus.context.update({"task_id": task.id, "stage": state.stage, "subtask_id": ""})
        task = self.store.update_task(task.id, status=TaskStatus.RUNNING, attempts=task.attempts + 1, started=task.started or task.created, error=None)
        self._emit(EventType.TASK_RESUMED if resuming else EventType.TASK_STARTED, f"{'Resuming' if resuming else 'Starting'}: {task.title}", mode=str(self.settings.permissions.mode))
        self._stop_watch = asyncio.create_task(self._watch_controls(task.id, token, owner))
        final_status = TaskStatus.FAILED
        error: str | None = None
        try:
            final_status = await self._pipeline(task, state)
        except _StopAfter:
            final_status = TaskStatus.PENDING
        except CancelledByUser as exc:
            final_status = TaskStatus.CANCELLED if "cancel" in str(exc) else TaskStatus.INTERRUPTED
            error = str(exc)
        except (BlockedError, AllModelsFailedError) as exc:
            final_status = TaskStatus.BLOCKED
            error = str(exc)
        except Exception as exc:
            log.debug("pipeline crashed:\n%s", traceback.format_exc())
            final_status = TaskStatus.FAILED
            error = f"internal error: {type(exc).__name__}: {exc}"
        finally:
            if self._stop_watch is not None:
                self._stop_watch.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._stop_watch
            await self.rt.tool_ctx.processes.stop_all()
        if final_status in (TaskStatus.BLOCKED, TaskStatus.INTERRUPTED, TaskStatus.CANCELLED, TaskStatus.FAILED) and state.stage not in ("report", "done", "understand"):
            with contextlib.suppress(Exception):
                await self._write_report(task, state, final_status, error)
        task = self._save(task, state, status=final_status, error=error, finished=_now())
        self.store.release_lease(task.id, owner)
        event = {
            TaskStatus.PENDING: EventType.STAGE_COMPLETED,
            TaskStatus.COMPLETED: EventType.TASK_COMPLETED,
            TaskStatus.COMPLETED_UNVERIFIED: EventType.TASK_COMPLETED,
            TaskStatus.BLOCKED: EventType.TASK_BLOCKED,
            TaskStatus.CANCELLED: EventType.TASK_CANCELLED,
            TaskStatus.INTERRUPTED: EventType.TASK_INTERRUPTED,
        }.get(final_status, EventType.TASK_FAILED)
        self._emit(event, f"Task {final_status}" + (f": {error}" if error else ""), status=str(final_status), report=state.report_path)
        self.store.save_metrics(task.id, self.rt.metrics.snapshot(task.id))
        self.rt.export_snapshots()
        return task

    # ------------------------------------------------------------------ stages

    async def _pipeline(self, task: Task, state: PipelineState) -> TaskStatus:
        if state.stage == "understand":
            await self._stage_understand(task, state)
        u = TaskUnderstanding.model_validate(state.understanding)
        if state.stage == "inspect":
            await self._stage_inspect(task, state, u)
        if u.task_type == "question":
            return await self._stage_answer(task, state, u)
        if state.stage == "plan":
            await self._stage_plan(task, state, u)
        if self._stop_after == "plan":
            raise _StopAfter
        if state.stage == "execute":
            await self._stage_execute(task, state, u)
        if state.stage == "final_qa":
            await self._stage_final(task, state, u)
        verdict = (state.gates or {}).get("verdict", "FAILED")
        status = {"COMPLETED": TaskStatus.COMPLETED, "COMPLETED_UNVERIFIED": TaskStatus.COMPLETED_UNVERIFIED}.get(verdict, TaskStatus.FAILED)
        if state.stage in ("report", "final_qa"):
            state.stage = "report"
            await self._write_report(task, state, status, None)
            self._retrospective(task, state)
            state.stage = "done"
            self._save(task, state)
        return status

    async def _stage_understand(self, task: Task, state: PipelineState) -> None:
        self._emit(EventType.STAGE_STARTED, "Understanding the request", stage="understand")
        self.bus.context["stage"] = "understand"
        if self.rt.profile is None:
            await self.rt.ensure_profile()
        brief = self.rt.context.project_brief(max_chars=3000)
        model = self.rt.router.for_role("classifier")
        u, source = await understand(model, task.description, brief, cancel=self.rt.tool_ctx.cancel)
        if u.blocking_questions:
            u = await self._clarify(task, state, u)
        state.understanding = u.model_dump()
        state.understanding_source = source
        state.stage = "inspect"
        self._save(task, state)
        self._emit(
            EventType.UNDERSTANDING_CREATED,
            f"{u.task_type} · {u.complexity}: {u.summary}",
            task_type=u.task_type, complexity=u.complexity, requirements=len(u.requirements),
            criteria=len(u.acceptance_criteria), source=source, needs=u.needs.model_dump(),
        )
        self._emit(EventType.STAGE_COMPLETED, "Understanding recorded", stage="understand")

    async def _clarify(self, task: Task, state: PipelineState, u: TaskUnderstanding) -> TaskUnderstanding:
        policy = self.settings.agent.on_questions
        interactive = self.rt.interactive and self.settings.permissions.mode != Mode.AUTONOMOUS
        self._emit(EventType.QUESTION_ASKED, f"{len(u.blocking_questions)} blocking question(s)", questions=u.blocking_questions)
        if policy == "block":
            raise BlockedError("clarification required: " + " | ".join(u.blocking_questions))
        if policy == "ask" and interactive:
            answers = await self.rt.questions.ask(u.blocking_questions, context=u.summary)
            if answers:
                for q, a in zip(u.blocking_questions, answers, strict=False):
                    state.clarifications.append({"question": q, "answer": a})
                extra = [f"Clarified: {c['question']} → {c['answer']}" for c in state.clarifications]
                return u.model_copy(update={"blocking_questions": [], "requirements": [*u.requirements, *extra]})
        assumed = [f"Unanswered question (proceeding with the most reasonable interpretation): {q}" for q in u.blocking_questions]
        return u.model_copy(update={"blocking_questions": [], "assumptions": [*u.assumptions, *assumed]})

    async def _stage_inspect(self, task: Task, state: PipelineState, u: TaskUnderstanding) -> None:
        self._emit(EventType.STAGE_STARTED, "Inspecting the repository and environment", stage="inspect")
        self.bus.context["stage"] = "inspect"
        await self.rt.ensure_profile(refresh=True)
        await self.rt.refresh_index()
        p = self.rt.profile
        if p is not None:
            self._emit(EventType.INFO, f"Repository: {p.file_count} files; primary language {p.primary_language or 'unknown'}; frameworks: {', '.join(p.frameworks[:6]) or 'none detected'}")
        if u.task_type == "change" and self.settings.validation.baseline and not state.baseline:
            self._emit(EventType.INFO, "Running baseline checks before any change")
            from ..tester.models import CheckKind

            kinds = [CheckKind.TEST, CheckKind.LINT, CheckKind.TYPECHECK]
            for kind, result in zip(kinds, await self._run_checks_parallel(task, None, kinds, baseline=True), strict=True):
                state.baseline[str(kind)] = result.model_dump(mode="json")
        state.stage = "plan" if u.task_type == "change" else "answer"
        self._save(task, state)
        self._emit(EventType.STAGE_COMPLETED, "Inspection complete", stage="inspect")

    async def _stage_answer(self, task: Task, state: PipelineState, u: TaskUnderstanding) -> TaskStatus:
        self._emit(EventType.STAGE_STARTED, "Investigating to answer the question", stage="answer")
        context = self.rt.context.task_context(task.description, u.relevant_paths, budget_chars=20000)
        message = f"# Question\n{task.description}\n\n# Context\n{context}"
        tools = self._tools(READ_TOOLS, ["web_fetch", "web_search"])
        result = await self._run_loop(role="coder", stage="answer", prompt="investigator", message=message, finish="submit_answer", tools=tools)
        answer = result.submission or {"answer": result.final_text or "(no answer produced)", "evidence": [], "confidence": "low"}
        state.answer = {**answer, "loop_status": result.status}
        ok = result.status == "finished" and bool(answer.get("answer"))
        verdict = "COMPLETED" if ok and answer.get("evidence") else "COMPLETED_UNVERIFIED"
        if not ok:
            verdict = "FAILED" if not result.final_text else "COMPLETED_UNVERIFIED"
        state.gates = {"verdict": verdict, "results": [], "note": "question task: no repository changes; answer evidence listed"}
        state.stage = "report"
        self._save(task, state)
        self._emit(EventType.STAGE_COMPLETED, "Answer ready", stage="answer")
        status = {"COMPLETED": TaskStatus.COMPLETED, "COMPLETED_UNVERIFIED": TaskStatus.COMPLETED_UNVERIFIED}.get(verdict, TaskStatus.FAILED)
        await self._write_report(task, state, status, None)
        state.stage = "done"
        self._save(task, state)
        return status

    async def _stage_plan(self, task: Task, state: PipelineState, u: TaskUnderstanding) -> None:
        self._emit(EventType.STAGE_STARTED, "Planning", stage="plan")
        self.bus.context["stage"] = "plan"
        brief = self.rt.context.project_brief(max_chars=4000)
        context = self.rt.context.task_context(task.description, u.relevant_paths, budget_chars=16000)
        plan, ordered, source = await make_plan(
            self.rt.router.for_role("planner"), task.description, u, brief, context, self.settings.agent.max_subtasks,
            cancel=self.rt.tool_ctx.cancel,
        )
        if self.settings.permissions.mode == Mode.ASSISTED and len(ordered) > 1:
            summary = "; ".join(f"{s.id}: {s.title}" for s in ordered)
            decision = await self.rt.approvals.request(ApprovalRequest(tool="plan", summary=f"Approve plan: {summary}"[:500], reason="assisted mode: plans need approval", details={"plan": plan.model_dump()}, task_id=task.id))
            if not decision.approved:
                raise BlockedError(f"plan not approved: {decision.reason}")
        state.plan = plan.model_dump()
        state.plan_source = source
        state.order = [s.id for s in ordered]
        state.subtasks = {s.id: SubtaskState(id=s.id, title=s.title) for s in ordered}
        for s in ordered:
            sub = Task(parent_id=task.id, title=s.title, description=s.description, depends_on=[f"{task.id}:{d}" for d in s.depends_on], priority=task.priority)
            sub.id = f"{task.id}:{s.id}"
            if self.store.get_task(sub.id) is None:
                self.store.create_task(sub)
        if self.rt.memory is not None:
            for decision_text in plan.decisions[:10]:
                with contextlib.suppress(Exception):
                    self.rt.memory.record_decision(decision_text[:120], decision_text, "recorded during planning", task_id=task.id)
        await self._prepare_git(task, state)
        state.stage = "execute"
        self._save(task, state)
        self._emit(EventType.PLAN_CREATED, f"Plan with {len(ordered)} subtask(s) ({source})", subtasks=[{"id": s.id, "title": s.title} for s in ordered], approach=plan.approach)
        self._emit(EventType.STAGE_COMPLETED, "Plan ready", stage="plan")

    async def _prepare_git(self, task: Task, state: PipelineState) -> None:
        git = self.rt.git
        if git is not None:
            status = await git.status()
            if status.in_progress:
                raise BlockedError(f"repository has a {status.in_progress} in progress; finish or abort it first")
            if status.conflicts:
                raise BlockedError(f"repository has unresolved conflicts: {', '.join(status.conflicts[:5])}")
            state.started_clean = status.clean
            state.original_branch = status.branch
            mode = self.settings.permissions.mode
            if self.settings.git.auto_commit == "if_clean" and status.clean and status.head and mode != Mode.SAFE:
                slug = re.sub(r"[^a-z0-9]+", "-", task.title.lower()).strip("-")[:40] or "task"
                branch = f"{self.settings.git.branch_prefix}{slug}-{task.id[-6:]}"
                await git.create_branch(branch)
                state.branch = branch
                state.auto_commit = True
                self._emit(EventType.INFO, f"Working on new branch {branch} (from {status.branch})")
            elif not status.clean:
                state.notes.append("working tree had uncommitted changes at start: agent changes are left uncommitted to avoid mixing them with yours")
        if state.start_checkpoint is None:
            cp = await self.rt.checkpoints.create("task start", task_id=task.id)
            state.start_checkpoint = cp.id
            self.store.update_task(task.id, checkpoint_id=cp.id)

    # ------------------------------------------------------------------ execution

    async def _stage_execute(self, task: Task, state: PipelineState, u: TaskUnderstanding) -> None:
        self._emit(EventType.STAGE_STARTED, "Executing the plan", stage="execute")
        plan = Plan.model_validate(state.plan)
        by_id = {s.id: s for s in plan.subtasks}
        for sid in state.order:
            st = state.subtasks[sid]
            if st.status in ("completed", "completed_unverified", "failed", "skipped"):
                continue
            self._check_budget()
            sub = by_id[sid]
            failed_deps = [d for d in sub.depends_on if state.subtasks.get(d) and state.subtasks[d].status in ("failed", "blocked", "skipped")]
            if failed_deps:
                st.status = "skipped"
                st.notes.append(f"skipped because dependencies failed: {', '.join(failed_deps)}")
                self._save(task, state)
                continue
            await self._execute_subtask(task, state, u, plan, sub, st)
            if st.status == "failed" and self.settings.agent.stop_on_subtask_failure:
                break
        state.stage = "final_qa"
        self._save(task, state)
        self._emit(EventType.STAGE_COMPLETED, "Plan executed", stage="execute")

    def _subtask_message(self, task: Task, state: PipelineState, u: TaskUnderstanding, plan: Plan, sub: Subtask) -> str:
        lines = [f"# Task\n{task.description}"]
        if state.clarifications:
            lines.append("# Clarifications\n" + "\n".join(f"- {c['question']} → {c['answer']}" for c in state.clarifications))
        lines.append("# Requirements\n" + "\n".join(f"- {r}" for r in u.requirements))
        if u.acceptance_criteria:
            lines.append("# Acceptance criteria (whole task)\n" + "\n".join(f"- {c}" for c in u.acceptance_criteria))
        if u.assumptions:
            lines.append("# Assumptions\n" + "\n".join(f"- {a}" for a in u.assumptions))
        if len(plan.subtasks) > 1:
            plan_lines = [f"- {s.id}: {s.title} [{state.subtasks[s.id].status}]" for s in plan.subtasks if s.id in state.subtasks]
            lines.append(f"# Plan\nApproach: {plan.approach}\n" + "\n".join(plan_lines))
        lines.append(f"# Your current subtask: {sub.id} — {sub.title}\n{sub.description}")
        if sub.acceptance_criteria:
            lines.append("Subtask acceptance criteria:\n" + "\n".join(f"- {c}" for c in sub.acceptance_criteria))
        if sub.files_hint:
            lines.append("Files likely involved: " + ", ".join(sub.files_hint))
        baseline = self._baseline_note(state)
        if baseline:
            lines.append(f"# Baseline (before any change)\n{baseline}")
        context = self.rt.context.task_context(f"{sub.title}\n{sub.description}\n{task.description}", sub.files_hint + u.relevant_paths, budget_chars=24000)
        if context:
            lines.append(f"# Context\n{context}")
        return "\n\n".join(lines)

    def _baseline_note(self, state: PipelineState) -> str:
        notes = []
        for kind, data in state.baseline.items():
            status = data.get("status")
            if status in ("unavailable", None):
                continue
            notes.append(f"- {kind}: {data.get('summary') or status}")
        return "\n".join(notes)

    async def _execute_subtask(self, task: Task, state: PipelineState, u: TaskUnderstanding, plan: Plan, sub: Subtask, st: SubtaskState) -> None:
        self.bus.context["subtask_id"] = sub.id
        self._emit(EventType.SUBTASK_STARTED, f"Subtask {sub.id}: {sub.title}", subtask=sub.id)
        child_id = f"{task.id}:{sub.id}"
        with contextlib.suppress(Exception):
            self.store.update_task(child_id, status=TaskStatus.RUNNING)
        resumed_with_changes = False
        if st.checkpoint_before is None:
            cp = await self.rt.checkpoints.create(f"before {sub.id}: {sub.title}", task_id=task.id, subtask_id=sub.id)
            st.checkpoint_before = cp.id
        elif st.status == "running":
            changed = await self.rt.checkpoints.changed_files_since(st.checkpoint_before)
            resumed_with_changes = bool(changed)
            if resumed_with_changes:
                self._emit(EventType.INFO, f"Resumed subtask {sub.id}: {len(changed)} changed file(s) found on disk — re-verifying instead of assuming completion")
        st.status = "running"
        st.attempts += 1
        self._save(task, state)

        if not resumed_with_changes:
            self._emit(EventType.STAGE_STARTED, f"Implementing {sub.id}", stage="implement")
            tools = self._tools(READ_TOOLS, WRITE_TOOLS, NETWORK_TOOLS)
            result = await self._run_loop(role="coder", stage="implement", prompt="implementer", message=self._subtask_message(task, state, u, plan, sub), finish="submit_work", tools=tools)
            st.submission = result.submission
            st.loop_status = result.status
            if result.status not in ("finished", "finished_no_submit"):
                st.notes.append(f"implementation loop ended with {result.status}: {result.error}")
            self._emit(EventType.STAGE_COMPLETED, f"Implementation loop: {result.status} ({result.steps} steps, {result.tool_calls} tool calls)", stage="implement")
            self._save(task, state)

        results = await self._validate_and_repair(task, state, sub, st, scope="subtask")
        review: ReviewResult | None = None
        findings: list[Any] = []
        if self.settings.gates.review != GateMode.DISABLED:
            review, findings, results = await self._review_loop(task, state, u, sub, st, results)
        st.files_changed = await self.rt.checkpoints.changed_files_since(st.checkpoint_before)
        report = self._subtask_gates(state, u, sub, st, results, review, findings)
        st.gates = report.model_dump(mode="json")
        verdict = report.verdict
        st.status = {"COMPLETED": "completed", "COMPLETED_UNVERIFIED": "completed_unverified"}.get(verdict, "failed")
        cp = await self.rt.checkpoints.create(f"after {sub.id} ({verdict})", task_id=task.id, subtask_id=sub.id, verified=verdict == "COMPLETED")
        st.checkpoint_after = cp.id
        if state.auto_commit and st.files_changed and verdict != "FAILED":
            st.commit = await self._commit(task, sub, st, verdict)
        self._save(task, state)
        child_status = {"completed": TaskStatus.COMPLETED, "completed_unverified": TaskStatus.COMPLETED_UNVERIFIED}.get(st.status, TaskStatus.FAILED)
        with contextlib.suppress(Exception):
            self.store.update_task(child_id, status=child_status, checkpoint_id=cp.id, verification={"verdict": verdict}, artifacts={"files_changed": st.files_changed}, attempts=st.attempts)
        self._emit(EventType.SUBTASK_COMPLETED, f"Subtask {sub.id} {verdict}", subtask=sub.id, verdict=verdict, files=st.files_changed[:30])
        self.bus.context["subtask_id"] = ""

    def _needs_repair(self, results: list[Any], state: PipelineState, in_scope: set[str] | None = None) -> list[Any]:
        failing = []
        for r in results:
            if r.ok() or r.status in ("unavailable", "skipped", "cancelled"):
                continue
            if r.classification in ("command_not_found",):
                continue
            if r.classification == "no_tests" and r.kind == "test":
                continue
            only_pre, _ = compare_with_baseline(r, _restore_check(state.baseline.get(str(r.kind))), in_scope if str(r.kind) == "test" else None)
            if not only_pre:
                failing.append(r)
        return failing

    async def _validate_and_repair(self, task: Task, state: PipelineState, sub: Subtask | None, st: SubtaskState | None, scope: str) -> list[Any]:
        since = st.checkpoint_before if st is not None else state.start_checkpoint
        assert since is not None
        results = await self._validate(task, state, sub, since, scope)
        tracker = FailureTracker(task.id, sub.id if sub else "final", max_same_signature=3)
        iterations = 0
        while True:
            in_scope = set(self._related_tests(await self.rt.checkpoints.changed_files_since(since)))
            failing = self._needs_repair(results, state, in_scope)
            if not failing:
                break
            if tracker.only_environmental(failing):
                self._emit(EventType.WARNING, "remaining failures are environmental (tools missing); not attempting code fixes")
                break
            if iterations >= self.settings.agent.max_repair_iterations:
                self._emit(EventType.WARNING, f"repair budget ({iterations} attempts) exhausted")
                break
            self._check_budget()
            files = await self.rt.checkpoints.changed_files_since(since)
            record = tracker.record_failure(failing, context=f"{'subtask ' + sub.id + ': ' + sub.title if sub else 'final validation'}; changed files: {', '.join(files[:20])}")
            self.store.add_failure(task.id, sub.id if sub else None, record.signature, record.model_dump())
            self._emit(EventType.FAILURE_RECORDED, f"Diagnosing: {record.error[:200]}", signature=record.signature, classification=record.classification)
            reason = tracker.loop_detected()
            if reason:
                record.result = "abandoned"
                self.store.add_failure(task.id, sub.id if sub else None, record.signature, record.model_dump())
                self._emit(EventType.LOOP_DETECTED, f"Stopping repair: {reason}")
                break
            iterations += 1
            message = self._repair_message(task, sub, record.render(), tracker.history_prompt(), files)
            tools = self._tools(READ_TOOLS, WRITE_TOOLS, ["web_fetch"])
            loop = await self._run_loop(role="debugger", stage="repair", prompt="debugger", message=message, finish="submit_work", tools=tools)
            submission = loop.submission or {}
            diff = await self.rt.checkpoints.diff_since(since)
            fresh = tracker.record_fix(record, hypothesis=str(submission.get("summary", loop.final_text))[:1500], change=str(submission.get("verification", ""))[:1500], diff_text=diff)
            self._emit(EventType.FIX_ATTEMPTED, f"Fix attempt {iterations}: {str(submission.get('summary', ''))[:160]}", attempt=iterations, duplicate=not fresh)
            if not fresh:
                self._emit(EventType.LOOP_DETECTED, "the attempted fix is identical to a previous one")
            results = await self._validate(task, state, sub, since, scope)
            in_scope = set(self._related_tests(await self.rt.checkpoints.changed_files_since(since)))
            outcome = tracker.record_outcome(record, self._needs_repair(results, state, in_scope))
            self.store.add_failure(task.id, sub.id if sub else None, record.signature, record.model_dump())
            if st is not None:
                st.repair_iterations = iterations
                st.failures.append(record.model_dump())
                self._save(task, state)
            if outcome == "fixed":
                self._emit(EventType.INFO, f"Fix attempt {iterations} resolved the failure")
        return results

    def _repair_message(self, task: Task, sub: Subtask | None, record: str, history: str, files: list[str]) -> str:
        parts = [f"# Task\n{task.description}"]
        if sub is not None:
            parts.append(f"# Subtask {sub.id}: {sub.title}\n{sub.description}")
        parts.append(f"# Failure record\n{record}")
        if history:
            parts.append(history)
        if files:
            parts.append("# Files changed so far\n" + "\n".join(f"- {f}" for f in files[:50]))
        return "\n\n".join(parts)

    async def _run_check(self, task: Task, sub_id: str | None, kind: Any, targeted: list[str] | None, baseline: bool = False) -> Any:
        engine = self.rt.validation
        label = "baseline " if baseline else ""
        is_test = str(kind) == "test"
        if is_test:
            self._emit(EventType.TEST_STARTED, f"Running {label}{'targeted ' if targeted else ''}tests", kind=str(kind), targeted=targeted[:20] if targeted else None)
        result = await engine.run_check(kind, targeted_files=targeted, cancel=self.rt.tool_ctx.cancel)
        result.output_tail = self.rt.redactor.redact_text(result.output_tail or "")
        if result.status != "unavailable":
            self.store.add_test_run(task.id, sub_id, result)
            if self.rt.memory is not None and result.status in ("passed", "failed") and result.classification not in ("command_not_found",):
                with contextlib.suppress(Exception):
                    self.rt.memory.record_command(result.command, kind=str(kind), ok=result.ok(), duration_s=result.duration_s)
        etype = EventType.TEST_PASSED if result.ok() else EventType.TEST_FAILED
        if not is_test:
            etype = EventType.VALIDATION_RESULT
        if result.status != "unavailable" or is_test:
            self._emit(etype, f"{label}{result.summary or (str(kind) + ': ' + result.status)}", kind=str(kind), status=result.status, passed=result.passed, failed=result.failed, errors=result.errors, command=result.command, baseline=baseline)
        return result

    async def _validate(self, task: Task, state: PipelineState, sub: Subtask | None, since: str, scope: str) -> list[Any]:
        from ..tester.models import CheckKind

        self.bus.context["stage"] = "validate"
        changed = await self.rt.checkpoints.changed_files_since(since)
        sub_id = sub.id if sub else None
        results: list[Any] = []
        if scope == "subtask":
            targeted_tests = self._related_tests(changed)
            if targeted_tests:
                targeted = await self._run_check(task, sub_id, CheckKind.TEST, targeted_tests)
                results.append(targeted)
                if targeted.ok() and self.settings.validation.full_suite_per_subtask:
                    results[-1] = await self._run_check(task, sub_id, CheckKind.TEST, None)
            else:
                results.append(await self._run_check(task, sub_id, CheckKind.TEST, None))
            lintable = [f for f in changed if (self.rt.workspace / f).exists()]
            results += await self._run_checks_parallel(
                task, sub_id, [CheckKind.LINT, CheckKind.TYPECHECK], targeted={CheckKind.LINT: lintable or None}
            )
        else:
            results += await self._run_checks_parallel(task, sub_id, [CheckKind.TEST, CheckKind.LINT, CheckKind.TYPECHECK, CheckKind.BUILD])
        summary = [r.model_dump(mode="json", exclude={"output_tail"}) for r in results]
        if sub is not None and sub.id in state.subtasks:
            state.subtasks[sub.id].validation = summary
        else:
            state.final_validation = summary
        self._save(task, state)
        return results

    async def _run_checks_parallel(
        self, task: Task, sub_id: str | None, kinds: list[Any], *, baseline: bool = False, targeted: dict[Any, list[str] | None] | None = None,
    ) -> list[Any]:
        """Run independent checks concurrently via the job graph.

        Lint and type checks only read the tree, so they run in parallel. Tests and
        builds can write artifacts, so they share an exclusive resource lock.
        Results are returned in the order of ``kinds``.
        """
        from ..parallel.graph import Job, run_jobs
        from ..tester.models import CheckKind

        targeted = targeted or {}
        writers = {CheckKind.TEST, CheckKind.BUILD}

        def make(kind: Any) -> Job:
            async def fn() -> Any:
                return await self._run_check(task, sub_id, kind, targeted.get(kind), baseline=baseline)

            return Job(id=str(kind), fn=fn, resources={"workspace-writers"} if kind in writers else set())

        outcome = await run_jobs([make(k) for k in kinds], max_concurrency=3, cancel=self.rt.tool_ctx.cancel)
        self.rt.tool_ctx.cancel.raise_if_cancelled()
        results = []
        for kind in kinds:
            job = outcome[str(kind)]
            if job.status != "ok":
                raise RuntimeError(f"{kind} check could not run: {job.status} {job.error}")
            results.append(job.value)
        return results

    def _related_tests(self, changed: list[str]) -> list[str]:
        tests: list[str] = []
        test_re = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+\.py$|_test\.(py|go)$|\.(test|spec)\.[jt]sx?$")
        for f in changed:
            if test_re.search(f) and (self.rt.workspace / f).exists():
                tests.append(f)
        index = self.rt.index
        if index is not None:
            for f in changed:
                if test_re.search(f):
                    continue
                with contextlib.suppress(Exception):
                    tests += [t for t in index.related_tests(f) if (self.rt.workspace / t).exists()]
        return list(dict.fromkeys(tests))[:50]

    def _validation_summary(self, results: list[Any]) -> str:
        return "\n".join(f"- {r.kind}: {r.status} — {r.summary} (command: {r.command or 'n/a'})" for r in results)

    async def _review_loop(self, task: Task, state: PipelineState, u: TaskUnderstanding, sub: Subtask, st: SubtaskState, results: list[Any]) -> tuple[ReviewResult | None, list[Any], list[Any]]:
        review: ReviewResult | None = None
        findings: list[Any] = []
        assert st.checkpoint_before is not None
        for iteration in range(self.settings.agent.max_review_iterations + 1):
            diff = await self.rt.checkpoints.diff_since(st.checkpoint_before)
            if not diff.strip():
                return None, [], results
            self._emit(EventType.REVIEW_STARTED, f"Independent review of {sub.id} (round {iteration + 1})", stage="review")
            self.bus.context["stage"] = "review"
            criteria = sub.acceptance_criteria or u.acceptance_criteria
            deleted = [f for f in st.files_changed if not (self.rt.workspace / f).exists()]
            review, findings = await review_change(
                self.rt.router.for_role("reviewer"), task=f"{task.description}\n\nSubtask {sub.id}: {sub.title}\n{sub.description}",
                criteria=criteria, diff=diff, validation_summary=self._validation_summary(results), deleted_files=deleted,
                repo_has_tests=bool(self.rt.profile and self.rt.profile.test_files), project_brief=self.rt.context.project_brief(max_chars=2000),
                cancel=self.rt.tool_ctx.cancel, focus=self._review_focus(u),
            )
            st.review = review.model_dump()
            st.review_iterations = iteration + 1
            self._save(task, state)
            blocking = review.blocking()
            unmet = [c for c in review.requirements if c.status == "unmet"]
            self._emit(EventType.REVIEW_COMPLETED, f"Review: {review.verdict} ({len(blocking)} blocking, {len(review.issues)} total)", verdict=review.verdict, blocking=len(blocking), source=review.source)
            if (not blocking and not unmet) or iteration >= self.settings.agent.max_review_iterations:
                break
            issues = "\n".join(f"- [{i.severity}/{i.category}] {i.file or ''}{':' + str(i.line) if i.line else ''} {i.description}" + (f" (suggestion: {i.suggestion})" if i.suggestion else "") for i in blocking)
            issues += "".join(f"\n- [unmet criterion] {c.criterion}: {c.evidence}" for c in unmet)
            message = f"# Task\n{task.description}\n\n# Subtask {sub.id}: {sub.title}\n{sub.description}\n\n# Review findings to address\n{issues}"
            tools = self._tools(READ_TOOLS, WRITE_TOOLS)
            await self._run_loop(role="coder", stage="review_fix", prompt="fixer", message=message, finish="submit_work", tools=tools)
            results = await self._validate_and_repair(task, state, sub, st, scope="subtask")
        return review, findings, results

    @staticmethod
    def _review_focus(u: TaskUnderstanding) -> list[str]:
        """Triage-driven routing: extra review depth only where the task calls for it."""
        focus = []
        if u.needs.security_review:
            focus.append("security")
        if u.needs.performance_review:
            focus.append("performance")
        return focus

    def _subtask_gates(self, state: PipelineState, u: TaskUnderstanding, sub: Subtask, st: SubtaskState, results: list[Any], review: ReviewResult | None, findings: list[Any]) -> GateReport:
        gates = self.settings.gates.model_copy(update={"docs": GateMode.DISABLED, "git_state": GateMode.DISABLED, "build": GateMode.DISABLED})
        submission = st.submission or {}
        blocked = bool(submission.get("blocked"))
        needs_changes = sub.kind not in ("research", "investigate")
        done = (st.loop_status in ("finished", "finished_no_submit", None)) and not blocked and (bool(st.files_changed) or not needs_changes)
        detail = "changes made: " + (", ".join(st.files_changed[:10]) or "none")
        if blocked:
            detail = f"agent reported it was blocked: {submission.get('unresolved', '')[:300]}"
        elif not st.files_changed and needs_changes:
            detail = "no files were changed"
        elif st.loop_status not in ("finished", "finished_no_submit", None):
            detail += f" (loop ended: {st.loop_status})"
        by_kind = {str(r.kind): r for r in results}
        return evaluate_gates(gates, GateInputs(
            in_scope_tests=set(self._related_tests(st.files_changed)),
            requirements_understood=bool(u.requirements),
            implementation_done=done,
            implementation_detail=detail,
            checks={"tests": by_kind.get("test"), "lint": by_kind.get("lint"), "typecheck": by_kind.get("typecheck")},
            baseline={k: _restore_check(state.baseline.get(v)) for k, v in (("tests", "test"), ("lint", "lint"), ("typecheck", "typecheck"))},
            security_findings=findings,
            security_ran=True,
            review=review,
            is_git=self.rt.git is not None,
        ))

    async def _commit(self, task: Task, sub: Subtask, st: SubtaskState, verdict: str) -> str | None:
        git = self.rt.git
        if git is None:
            return None
        paths = st.files_changed
        if not paths:
            return None
        try:
            try:
                staged_files = await stage_for_commit(git, self.rt.tool_ctx.guard, paths)
            except ToolError as exc:
                st.notes.append(f"not committed: {exc}")
                self._emit(EventType.WARNING, f"Not committing {sub.id}: {exc}")
                return None
            if not staged_files:
                return None
            message = f"{self.settings.git.commit_prefix}{sub.title}\n\nTask: {task.id}\nSubtask: {sub.id}\nVerification: {verdict}\n"
            sha = await git.commit(message, only=paths)
        except Exception as exc:
            st.notes.append(f"commit failed: {exc}")
            self._emit(EventType.WARNING, f"Commit for {sub.id} failed: {exc}")
            return None
        self._emit(EventType.COMMIT_CREATED, f"Committed {sub.id} as {sha[:10]}", sha=sha)
        return sha

    # ------------------------------------------------------------------ final QA

    async def _stage_final(self, task: Task, state: PipelineState, u: TaskUnderstanding) -> None:
        self._emit(EventType.STAGE_STARTED, "Final QA: full validation, security scan, final review", stage="final_qa")
        self.bus.context.update({"stage": "final_qa", "subtask_id": ""})
        assert state.start_checkpoint is not None
        results = await self._validate_and_repair(task, state, None, None, scope="final")
        diff = await self.rt.checkpoints.diff_since(state.start_checkpoint)
        changed = await self.rt.checkpoints.changed_files_since(state.start_checkpoint)
        review: ReviewResult | None = None
        findings: list[Any] = []
        plan = Plan.model_validate(state.plan) if state.plan else None
        single = plan is not None and len(plan.subtasks) == 1
        if self.settings.gates.review != GateMode.DISABLED and diff.strip():
            if single and state.subtasks:
                only = next(iter(state.subtasks.values()))
                if only.review and not await self._changed_after(only.checkpoint_after):
                    review = ReviewResult.model_validate(only.review)
            if review is None:
                self._emit(EventType.REVIEW_STARTED, "Final review against the original requirements", stage="final_qa")
                review, _ = await review_change(
                    self.rt.router.for_role("reviewer"), task=task.description, criteria=u.acceptance_criteria, diff=diff,
                    validation_summary=self._validation_summary(results), deleted_files=[f for f in changed if not (self.rt.workspace / f).exists()],
                    repo_has_tests=bool(self.rt.profile and self.rt.profile.test_files), project_brief=self.rt.context.project_brief(max_chars=2000),
                    cancel=self.rt.tool_ctx.cancel, focus=self._review_focus(u),
                )
                self._emit(EventType.REVIEW_COMPLETED, f"Final review: {review.verdict}", verdict=review.verdict, blocking=len(review.blocking()))
            state.final_review = review.model_dump()
        from ..reviewer.review import deterministic_review

        _, findings = deterministic_review(diff)
        state.security_findings = [f.to_dict() for f in findings]
        audit = None
        if self.settings.validation.dependency_audit and any(re.search(r"(package\.json|requirements.*\.txt|pyproject\.toml|poetry\.lock|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|go\.mod|Cargo\.toml)$", f) for f in changed):
            from ..tester.models import CheckKind

            audit = await self._run_check(task, None, CheckKind.AUDIT, None)
        docs = await self._docs_gate(task, state, u, changed)
        state.docs = {"status": docs[0], "detail": docs[1]} if docs else None
        git_state = await self._git_state(state, diff)
        state.git_state = {"ok": git_state[0], "detail": git_state[1]} if git_state else None
        by_kind = {str(r.kind): r for r in results}
        any_failed = any(s.status in ("failed", "skipped", "blocked") for s in state.subtasks.values())
        detail = f"{sum(1 for s in state.subtasks.values() if s.status.startswith('completed'))}/{len(state.subtasks)} subtask(s) completed; {len(changed)} file(s) changed"
        report = evaluate_gates(self.settings.gates, GateInputs(
            in_scope_tests=set(self._related_tests(changed)),
            requirements_understood=bool(u.requirements),
            implementation_done=bool(changed) and not any_failed,
            implementation_detail=detail,
            checks={"tests": by_kind.get("test"), "lint": by_kind.get("lint"), "typecheck": by_kind.get("typecheck"), "build": by_kind.get("build")},
            baseline={k: _restore_check(state.baseline.get(v)) for k, v in (("tests", "test"), ("lint", "lint"), ("typecheck", "typecheck"))},
            security_findings=findings,
            security_ran=True,
            audit=audit,
            review=review,
            docs=docs,
            git_state=git_state,
            is_git=self.rt.git is not None,
        ))
        state.gates = report.model_dump(mode="json")
        state.stage = "report"
        self._save(task, state, verification=report.model_dump(mode="json"), artifacts={"files_changed": changed, "branch": state.branch})
        self._emit(EventType.GATES_EVALUATED, f"Quality gates: {report.verdict}", verdict=report.verdict, failed=[g.name for g in report.failed()], unverified=[g.name for g in report.unverified()])
        self._emit(EventType.STAGE_COMPLETED, "Final QA complete", stage="final_qa")

    async def _changed_after(self, checkpoint_id: str | None) -> bool:
        if checkpoint_id is None:
            return True
        return bool(await self.rt.checkpoints.changed_files_since(checkpoint_id))

    async def _docs_gate(self, task: Task, state: PipelineState, u: TaskUnderstanding, changed: list[str]) -> tuple[str, str] | None:
        if not u.needs.docs_update:
            return None
        if any(_DOC_FILE.search(f) for f in changed):
            return "passed", "documentation updated: " + ", ".join(f for f in changed if _DOC_FILE.search(f))[:300]
        assert state.start_checkpoint is not None
        stat = await self.rt.checkpoints.diff_since(state.start_checkpoint, stat=True)
        message = f"# Change summary\n{task.description}\n\n# Files changed\n{stat}\n\nUpdate the documentation affected by this change."
        await self._run_loop(role="coder", stage="docs", prompt="docs", message=message, finish="submit_work", tools=self._tools(READ_TOOLS, ["write_file", "edit_file"]))
        changed = await self.rt.checkpoints.changed_files_since(state.start_checkpoint)
        if any(_DOC_FILE.search(f) for f in changed):
            return "passed", "documentation updated by the docs stage"
        return "failed", "the change affects documented behaviour but no documentation was updated"

    async def _git_state(self, state: PipelineState, diff: str) -> tuple[bool, str] | None:
        git = self.rt.git
        if git is None:
            return None
        status = await git.status()
        problems = []
        if status.conflicts:
            problems.append(f"unresolved conflicts: {', '.join(status.conflicts[:5])}")
        if status.in_progress:
            problems.append(f"{status.in_progress} in progress")
        added = added_lines_by_file(diff)
        markers = [p for p, lines in added.items() if any(t.startswith(("<<<<<<< ", ">>>>>>> ")) for _, t in lines)]
        if markers:
            problems.append(f"conflict markers in {', '.join(markers[:5])}")
        secrets = scan_text("\n".join(t for lines in added.values() for _, t in lines))
        if secrets:
            problems.append(f"possible secrets in the change ({', '.join(sorted({s.kind for s in secrets}))})")
        big = [p for p in added if (self.rt.workspace / p).exists() and (self.rt.workspace / p).stat().st_size > 5_000_000]
        if big:
            problems.append(f"very large files added: {', '.join(big[:3])}")
        if problems:
            return False, "; ".join(problems)
        note = f"branch {status.branch or '(detached)'}"
        if state.auto_commit:
            uncommitted = [e.path for e in status.entries]
            note += f"; agent changes committed on {state.branch}" if not uncommitted else f"; {len(uncommitted)} uncommitted path(s) remain"
        else:
            note += "; changes left uncommitted for your review" if not status.clean else "; working tree clean"
        return True, note

    # ------------------------------------------------------------------ report & retrospective

    async def _write_report(self, task: Task, state: PipelineState, status: TaskStatus, error: str | None) -> None:
        from .report import build_report

        task = self.store.require_task(task.id)
        metrics = self.rt.metrics.snapshot(task.id)
        diffstat = ""
        if state.start_checkpoint:
            with contextlib.suppress(Exception):
                diffstat = await self.rt.checkpoints.diff_since(state.start_checkpoint, stat=True)
        md, data = build_report(task, state, status, error, metrics, diffstat, self.store.failures(task.id), self.store.list_checkpoints(task.id), self.rt.router.configured_roles())
        reports = self.rt.state_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        path = reports / f"{task.id}.md"
        path.write_text(self.rt.redactor.redact_text(md), encoding="utf-8")
        from ..core.util import atomic_write_json

        atomic_write_json(reports / f"{task.id}.json", self.rt.redactor.redact(data))
        state.report_path = str(path)
        self._save(task, state)
        self._emit(EventType.REPORT_CREATED, f"Report written to {path.relative_to(self.rt.workspace).as_posix()}", path=str(path))

    def _retrospective(self, task: Task, state: PipelineState) -> None:
        from ..improvement.retrospective import analyze

        with contextlib.suppress(Exception):
            suggestions = analyze(self.rt.metrics.snapshot(task.id), state, self.store.failures(task.id))
            if suggestions:
                from ..core.util import atomic_write_json

                atomic_write_json(self.rt.state_dir / "reports" / f"{task.id}.improvements.json", suggestions)


def _restore_check(data: dict[str, Any] | None) -> Any:
    if not data:
        return None
    from ..tester.models import CheckResult

    try:
        return CheckResult.model_validate(data)
    except Exception:
        return None


def _now() -> str:
    from ..core.util import utcnow_iso

    return utcnow_iso()
