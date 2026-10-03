"""Local web dashboard: Starlette JSON API + Server-Sent Events over the same Runtime the CLI uses.

Security model (the dashboard can drive an agent that edits files and runs commands):

* binds to loopback by default (``settings.ui.host``);
* every ``/api`` route and the event stream require an access token (``AIE_WEB_TOKEN`` or a
  random one printed once at start-up), sent as ``Authorization: Bearer``, as ``?token=``, or
  as the HttpOnly ``SameSite=Strict`` cookie set when ``/`` is opened with a valid ``?token=``;
* state-changing requests from another origin are rejected (CSRF guard) and JSON bodies must
  be sent as ``application/json``;
* everything returned passes through the runtime's secret redactor; events carry action
  summaries only, never hidden model reasoning;
* static files are served from a fixed allow-list (no path handling at all).

Only one task runs at a time: concurrent agents writing to one working tree are unsafe.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import socket
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from types import FrameType
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ... import __version__
from ...config.settings import Mode
from ...core.cancel import CancellationToken
from ...core.errors import StateError
from ...core.events import Event, EventBus, EventType
from ...core.util import utcnow_iso
from ...orchestrator.questions import QueueQuestionBroker
from ...runtime import Runtime
from ...tasks.store import TERMINAL_STATUSES, Task, TaskStatus
from ...tools.approval import ApprovalRequest, QueueBroker

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
# The only files ever served from disk. Requests never touch the filesystem by name.
STATIC_FILES: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}
COOKIE_NAME = "aie_token"
TOKEN_ENV = "AIE_WEB_TOKEN"  # noqa: S105 - the variable name, not a secret
MAX_BODY_BYTES = 1_000_000
MAX_DESCRIPTION_CHARS = 50_000
DIFF_LIMIT_CHARS = 200_000
HEARTBEAT_S = 15.0
RESUMABLE = {TaskStatus.INTERRUPTED, TaskStatus.BLOCKED, TaskStatus.PENDING, TaskStatus.FAILED, TaskStatus.QUEUED}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'self'"
    ),
}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _json(data: Any, status: int = 200, headers: dict[str, str] | None = None) -> Response:
    body = json.dumps(data, default=str, ensure_ascii=False, separators=(",", ":"))
    return Response(body, status_code=status, media_type="application/json",
                    headers={"Cache-Control": "no-store", **(headers or {})})


def format_sse(event: Event) -> str:
    """One Server-Sent Events frame. ``json.dumps`` escapes newlines, so ``data`` is a single line."""
    payload = json.dumps(event.model_dump(mode="json"), default=str, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.id}\nevent: {event.type}\ndata: {payload}\n\n"


async def sse_events(
    bus: EventBus,
    *,
    task_id: str | None = None,
    heartbeat_s: float = HEARTBEAT_S,
    poll_s: float = 1.0,
    should_stop: Callable[[], bool] = lambda: False,
) -> AsyncIterator[str]:
    """Live bus events as SSE frames, with heartbeat comments. Unsubscribes when closed.

    The subscription is made on first iteration so that a response that is never
    streamed leaves no dangling subscriber.
    """
    queue, unsubscribe = bus.subscribe_queue(maxsize=5000)
    try:
        yield "retry: 3000\n: connected\n\n"
        last = time.monotonic()
        while not should_stop():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=poll_s)
            except TimeoutError:
                if time.monotonic() - last >= heartbeat_s:
                    last = time.monotonic()
                    yield ": keep-alive\n\n"
                continue
            if task_id and event.task_id != task_id:
                continue
            last = time.monotonic()
            yield format_sse(event)
    finally:
        unsubscribe()


class _SecurityHeaders:
    """Pure ASGI middleware (safe for streaming responses) adding security headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for key, value in SECURITY_HEADERS.items():
                    headers.setdefault(key, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


Handler = Callable[[Request], Awaitable[Response]]


class Dashboard:
    """State and request handlers for one Runtime."""

    def __init__(self, rt: Runtime, token: str) -> None:
        if not token:
            raise ValueError("an access token is required")
        self.rt = rt
        self.token = token
        self.lock = asyncio.Lock()
        self.default_mode: Mode = rt.settings.permissions.mode
        self.running_task_id: str | None = None
        self.current: asyncio.Task[None] | None = None
        self.results: dict[str, dict[str, Any]] = {}
        self.closing = False
        rt.redactor.add_value(token)  # the token must never appear in events or responses

    # ---------------------------------------------------------------- security

    def token_ok(self, value: str | None) -> bool:
        if not value:
            return False
        return secrets.compare_digest(value.encode("utf-8"), self.token.encode("utf-8"))

    def authorized(self, request: Request) -> bool:
        scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
        candidates = [
            credentials.strip() if scheme.lower() == "bearer" else "",
            request.query_params.get("token", ""),
            request.cookies.get(COOKIE_NAME, ""),
        ]
        return any(self.token_ok(c) for c in candidates if c)

    @staticmethod
    def same_origin(request: Request) -> bool:
        origin = request.headers.get("origin")
        if origin is None:
            return request.headers.get("sec-fetch-site", "same-origin") in ("same-origin", "none")
        host = request.headers.get("host", "")
        return bool(host) and origin.lower() == f"{request.url.scheme}://{host}".lower()

    def api(self, handler: Handler) -> Handler:
        async def endpoint(request: Request) -> Response:
            if not self.authorized(request):
                return self.error(401, "unauthorized: an access token is required", {"WWW-Authenticate": "Bearer"})
            if request.method not in ("GET", "HEAD", "OPTIONS") and not self.same_origin(request):
                return self.error(403, "cross-origin request rejected")
            try:
                return await handler(request)
            except ApiError as exc:
                return self.error(exc.status, exc.message)
            except Exception:
                log.exception("web API error on %s %s", request.method, request.url.path)
                return self.error(500, "internal error (see .agent/logs/agent.log)")

        return endpoint

    def error(self, status: int, message: str, headers: dict[str, str] | None = None) -> Response:
        return _json({"error": self.rt.redactor.redact_text(message)}, status, headers)

    def out(self, data: Any, status: int = 200) -> Response:
        return _json(self.rt.redactor.redact(data), status)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    async def body(request: Request) -> dict[str, Any]:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            raise ApiError(413, "request body too large")
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise ApiError(413, "request body too large")
        if not raw.strip():
            return {}
        if "application/json" not in request.headers.get("content-type", "").lower():
            raise ApiError(415, "request body must be application/json")
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ApiError(400, "invalid JSON body") from exc
        if not isinstance(data, dict):
            raise ApiError(400, "JSON body must be an object")
        return data

    @staticmethod
    def int_param(request: Request, name: str, default: int, lo: int, hi: int) -> int:
        raw = request.query_params.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ApiError(400, f"'{name}' must be an integer") from exc
        return max(lo, min(hi, value))

    def get_task(self, task_id: str) -> Task:
        try:
            return self.rt.store.require_task(task_id)
        except StateError as exc:
            raise ApiError(404, f"no such task: {task_id}") from exc

    @staticmethod
    def parse_mode(value: Any) -> Mode | None:
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ApiError(400, "'mode' must be a string")
        try:
            return Mode(value.strip().lower())
        except ValueError as exc:
            raise ApiError(400, f"unknown mode '{value}'; expected one of {[m.value for m in Mode]}") from exc

    def mode_for(self, task: Task) -> Mode:
        try:
            return Mode(task.mode) if task.mode else self.default_mode
        except ValueError:
            return self.default_mode

    def approvals_pending(self) -> list[ApprovalRequest]:
        broker = self.rt.approvals
        return list(broker.pending()) if isinstance(broker, QueueBroker) else []

    def questions_pending(self) -> list[Any]:
        broker = self.rt.questions
        return list(broker.pending()) if isinstance(broker, QueueQuestionBroker) else []

    def summary(self, task: Task) -> dict[str, Any]:
        return {
            "id": task.id, "title": task.title, "status": str(task.status), "stage": task.stage,
            "priority": task.priority, "mode": task.mode, "created": task.created, "updated": task.updated,
            "started": task.started, "finished": task.finished, "error": task.error, "attempts": task.attempts,
            "running": task.id == self.running_task_id,
        }

    def relative(self, path: str | None) -> str | None:
        if not path:
            return None
        try:
            return Path(path).resolve().relative_to(self.rt.workspace).as_posix()
        except (ValueError, OSError):
            return Path(path).name

    def report_file(self, task: Task) -> Path | None:
        reports = (self.rt.state_dir / "reports").resolve()
        candidates: list[Path] = []
        declared = (task.state or {}).get("report_path")
        if isinstance(declared, str) and declared:
            candidates.append(Path(declared))
        candidates.append(reports / f"{task.id}.md")
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                continue
            if resolved.is_relative_to(reports) and resolved.suffix == ".md" and resolved.is_file():
                return resolved
        return None

    def state_summary(self, task: Task) -> dict[str, Any]:
        state: dict[str, Any] = task.state or {}
        u = state.get("understanding") or {}
        plan = state.get("plan") or {}
        plan_subs = {s.get("id"): s for s in plan.get("subtasks", []) if isinstance(s, dict)}
        subs: dict[str, Any] = state.get("subtasks") or {}
        order = [sid for sid in (state.get("order") or []) if sid in subs] or list(subs)
        subtasks = []
        for sid in order:
            st = subs.get(sid) or {}
            ps = plan_subs.get(sid) or {}
            review = st.get("review") or {}
            gates = st.get("gates") or {}
            subtasks.append({
                "id": sid,
                "title": st.get("title") or ps.get("title") or sid,
                "description": ps.get("description", ""),
                "kind": ps.get("kind"),
                "depends_on": ps.get("depends_on", []),
                "acceptance_criteria": ps.get("acceptance_criteria", []),
                "status": st.get("status", "pending"),
                "attempts": st.get("attempts", 0),
                "files_changed": st.get("files_changed", []),
                "repair_iterations": st.get("repair_iterations", 0),
                "review_iterations": st.get("review_iterations", 0),
                "review": {
                    "verdict": review.get("verdict"), "summary": review.get("summary", ""),
                    "issues": len(review.get("issues") or []), "source": review.get("source", ""),
                } if review else None,
                "gates_verdict": gates.get("verdict"),
                "failures": len(st.get("failures") or []),
                "loop_status": st.get("loop_status"),
                "commit": st.get("commit"),
                "notes": (st.get("notes") or [])[-5:],
            })
        gates = state.get("gates") or None
        final_review = state.get("final_review") or None
        report = self.report_file(task)
        return {
            "stage": state.get("stage") or task.stage,
            "understanding": {
                "summary": u.get("summary", ""), "task_type": u.get("task_type"), "complexity": u.get("complexity"),
                "requirements": u.get("requirements", []), "acceptance_criteria": u.get("acceptance_criteria", []),
                "assumptions": u.get("assumptions", []), "risk_areas": u.get("risk_areas", []),
                "source": state.get("understanding_source", ""),
            } if u else None,
            "clarifications": state.get("clarifications", []),
            "plan": {
                "goal": plan.get("goal", ""), "approach": plan.get("approach", ""), "risks": plan.get("risks", []),
                "decisions": plan.get("decisions", []), "source": state.get("plan_source", ""),
            } if plan else None,
            "subtasks": subtasks,
            "gates": {
                "verdict": gates.get("verdict"), "note": gates.get("note", ""),
                "results": [
                    {k: r.get(k) for k in ("name", "mode", "status", "detail")}
                    for r in gates.get("results", []) if isinstance(r, dict)
                ],
            } if gates else None,
            "final_review": {
                "verdict": final_review.get("verdict"), "summary": final_review.get("summary", ""),
                "issues": (final_review.get("issues") or [])[:50], "source": final_review.get("source", ""),
            } if final_review else None,
            "security_findings": (state.get("security_findings") or [])[:50],
            "answer": state.get("answer"),
            "branch": state.get("branch"),
            "original_branch": state.get("original_branch"),
            "start_checkpoint": state.get("start_checkpoint"),
            "report_path": self.relative(str(report)) if report else None,
            "has_report": report is not None,
            "resumed": state.get("resumed", 0),
            "notes": (state.get("notes") or [])[-20:],
        }

    # ---------------------------------------------------------------- running tasks

    async def start(self, task_id: str) -> None:
        """Start ``task_id`` in the background; raises 409 if a task is already running."""
        if self.lock.locked():
            raise ApiError(409, f"busy: task {self.running_task_id or '?'} is already running; one task runs at a time")
        await self.lock.acquire()
        try:
            task = self.get_task(task_id)
            self.rt.settings.permissions.mode = self.mode_for(task)
            # A stopped token stays cancelled: every run gets a fresh one.
            self.rt.tool_ctx.cancel = CancellationToken()
            self.running_task_id = task.id
            self.results.pop(task.id, None)
            self.current = asyncio.create_task(self._run(task.id), name=f"aie-web-run-{task.id}")
        except BaseException:
            self._finish_run()
            raise

    async def _run(self, task_id: str) -> None:
        try:
            result = await self.rt.orchestrator.run(task_id)
            self.results[task_id] = {"status": str(result.status), "error": result.error, "finished": result.finished}
        except Exception as exc:
            log.exception("web task run failed: %s", task_id)
            self.results[task_id] = {"status": "ERROR", "error": str(exc), "finished": utcnow_iso()}
            self.rt.bus.emit(EventType.ERROR, f"task run failed: {exc}", task_id=task_id, level="error")
        finally:
            self._finish_run()

    def _finish_run(self) -> None:
        self.rt.settings.permissions.mode = self.default_mode
        self.running_task_id = None
        for key in ("task_id", "subtask_id", "stage"):
            self.rt.bus.context.pop(key, None)
        if self.lock.locked():
            self.lock.release()

    def release_waiters(self, reason: str) -> None:
        """Unblock a run waiting on a human so cancellation takes effect promptly."""
        broker = self.rt.approvals
        if isinstance(broker, QueueBroker):
            broker.deny_all(reason)
        questions = self.rt.questions
        if isinstance(questions, QueueQuestionBroker):
            for item in questions.pending():
                questions.answer(item.id, [])

    async def shutdown(self, timeout_s: float = 20.0) -> None:
        self.closing = True
        current = self.current
        if current is None or current.done():
            return
        self.rt.tool_ctx.cancel.cancel("web server shutting down")
        self.release_waiters("web server shutting down")
        try:
            await asyncio.wait_for(asyncio.shield(current), timeout=timeout_s)
        except Exception:  # timed out or failed while shutting down: force-cancel
            current.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await current

    # ---------------------------------------------------------------- pages

    async def index(self, request: Request) -> Response:
        response = FileResponse(STATIC_DIR / "index.html", media_type=STATIC_FILES["index.html"],
                                headers={"Cache-Control": "no-store"})
        supplied = request.query_params.get("token")
        if supplied and self.token_ok(supplied):
            response.set_cookie(COOKIE_NAME, self.token, httponly=True, samesite="strict", path="/",
                                secure=request.url.scheme == "https")
        return response

    async def static(self, request: Request) -> Response:
        name = str(request.path_params.get("name", ""))
        media_type = STATIC_FILES.get(name)
        if media_type is None:
            return self.error(404, "not found")
        return FileResponse(STATIC_DIR / name, media_type=media_type, headers={"Cache-Control": "no-cache"})

    # ---------------------------------------------------------------- API: status & tasks

    async def status(self, request: Request) -> Response:
        rt = self.rt
        data = self.rt.redactor.redact({
            "version": __version__,
            "mode": str(self.default_mode),
            "active_mode": str(rt.settings.permissions.mode),
            "modes": [m.value for m in Mode],
            "ceiling": rt.policy.ceiling.name,
            "max_level": rt.settings.permissions.max_level.name,
            "roles": rt.router.configured_roles(),
            "git": rt.git is not None,
            "profile": getattr(rt.profile, "summary", None) or None,
            "running_task_id": self.running_task_id,
            "pending_approvals": len(self.approvals_pending()),
            "pending_questions": len(self.questions_pending()),
            "event_types": [t.value for t in EventType],
        })
        # The workspace path is the operator's own configuration, not model/tool output; the
        # redactor's entropy heuristics would otherwise mangle UUID-like directory names.
        data.update(workspace=str(rt.workspace), project=rt.workspace.name)
        return _json(data)

    async def list_tasks(self, request: Request) -> Response:
        limit = self.int_param(request, "limit", 100, 1, 500)
        tasks = self.rt.store.list_tasks(top_level=True, limit=limit)
        return self.out({"tasks": [self.summary(t) for t in tasks], "running_task_id": self.running_task_id})

    async def create_task(self, request: Request) -> Response:
        body = await self.body(request)
        description = body.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ApiError(400, "'description' is required")
        if len(description) > MAX_DESCRIPTION_CHARS:
            raise ApiError(400, f"'description' is longer than {MAX_DESCRIPTION_CHARS} characters")
        title = body.get("title")
        if title is not None and not isinstance(title, str):
            raise ApiError(400, "'title' must be a string")
        mode = self.parse_mode(body.get("mode"))
        start = body.get("start", True)
        if not isinstance(start, bool):
            raise ApiError(400, "'start' must be a boolean")
        priority = body.get("priority", 50)
        if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 1000:
            raise ApiError(400, "'priority' must be an integer between 0 and 1000")
        if start and self.lock.locked():
            raise ApiError(409, f"busy: task {self.running_task_id or '?'} is already running; create it with start=false to run it later")
        task = self.rt.create_task(description, title=title or None, priority=priority)
        task = self.rt.store.update_task(task.id, mode=(mode or self.default_mode).value)
        if start:
            await self.start(task.id)
            task = self.get_task(task.id)
        return self.out({"task": self.summary(task), "started": start}, 201)

    async def get_task_detail(self, request: Request) -> Response:
        task = self.get_task(request.path_params["task_id"])
        state = self.state_summary(task)
        metrics = self.rt.metrics.snapshot(task.id) or self.rt.store.get_metrics(task.id)
        files: list[str] = []
        for sub in state["subtasks"]:
            files += [f for f in sub["files_changed"] if isinstance(f, str)]
        files += [f for f in (metrics.get("files_changed") or []) if isinstance(f, str)]
        data = task.model_dump(mode="json", exclude={"state", "lease_owner", "lease_expires"})
        return self.out({
            "task": {**data, "running": task.id == self.running_task_id},
            "result": self.results.get(task.id),
            "state": state,
            "files_changed": list(dict.fromkeys(files)),
            "checkpoints": [cp.model_dump(mode="json") for cp in self.rt.store.list_checkpoints(task.id)],
            "test_runs": self.rt.store.test_runs(task.id),
            "failures": self.rt.store.failures(task.id),
            "metrics": metrics,
            "running_task_id": self.running_task_id,
        })

    async def task_events(self, request: Request) -> Response:
        task = self.get_task(request.path_params["task_id"])
        after = request.query_params.get("after") or None
        limit = self.int_param(request, "limit", 500, 1, 5000)
        selected, has_more, reset = self.rt.store.events_page(task.id, after, limit)
        return self.out({
            "events": selected,
            "last_id": selected[-1]["id"] if selected else after,
            "has_more": has_more,
            "reset": reset,
        })

    async def task_report(self, request: Request) -> Response:
        task = self.get_task(request.path_params["task_id"])
        path = self.report_file(task)
        if path is None:
            raise ApiError(404, "no report for this task yet")
        text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        return self.out({"markdown": text, "path": self.relative(str(path))})

    async def stop_task(self, request: Request) -> Response:
        return await self._halt(request, cancel=False)

    async def cancel_task(self, request: Request) -> Response:
        return await self._halt(request, cancel=True)

    async def _halt(self, request: Request, *, cancel: bool) -> Response:
        task = self.get_task(request.path_params["task_id"])
        action = "cancel" if cancel else "stop"
        if task.id == self.running_task_id:
            # "cancel" in the reason makes the orchestrator record CANCELLED, otherwise INTERRUPTED.
            reason = "cancel requested (web)" if cancel else "stop requested (web)"
            self.rt.tool_ctx.cancel.cancel(reason)
            self.release_waiters(f"task {action} requested")
            return self.out({"ok": True, "task_id": task.id, "action": action, "via": "this server"}, 202)
        if task.status == TaskStatus.RUNNING:
            self.rt.request_stop(task.id, cancel=cancel)
            return self.out({"ok": True, "task_id": task.id, "action": action, "via": "control request"}, 202)
        if cancel and task.status not in TERMINAL_STATUSES:
            task = self.rt.store.update_task(task.id, status=TaskStatus.CANCELLED, finished=utcnow_iso(),
                                             error="cancelled from the web dashboard before it ran")
            self.rt.bus.emit(EventType.TASK_CANCELLED, f"Task {task.id} cancelled", task_id=task.id, data={"status": "CANCELLED"})
            return self.out({"ok": True, "task_id": task.id, "action": action, "via": "store", "task": self.summary(task)})
        raise ApiError(409, f"task is not running (status {task.status})")

    async def resume_task(self, request: Request) -> Response:
        task = self.get_task(request.path_params["task_id"])
        if task.parent_id:
            raise ApiError(400, "subtasks run as part of their parent task; resume the parent instead")
        if task.status not in RESUMABLE:
            raise ApiError(409, f"task cannot be resumed from status {task.status}")
        if self.lock.locked():
            raise ApiError(409, f"busy: task {self.running_task_id or '?'} is already running; one task runs at a time")
        if (task.lease_owner and task.lease_owner != self.rt.lease_owner
                and task.lease_expires is not None and task.lease_expires > time.time()):
            raise ApiError(409, "task is held by another agent process")
        if task.status == TaskStatus.FAILED:
            self.rt.store.update_task(task.id, status=TaskStatus.INTERRUPTED)
        await self.start(task.id)
        return self.out({"ok": True, "task": self.summary(self.get_task(task.id))}, 202)

    # ---------------------------------------------------------------- API: human-in-the-loop

    async def list_approvals(self, request: Request) -> Response:
        return self.out({"approvals": [r.model_dump(mode="json") for r in self.approvals_pending()]})

    async def resolve_approval(self, request: Request) -> Response:
        approval_id = request.path_params["approval_id"]
        body = await self.body(request)
        approved = body.get("approved")
        if not isinstance(approved, bool):
            raise ApiError(400, "'approved' must be a boolean")
        reason = body.get("reason") or ""
        if not isinstance(reason, str):
            raise ApiError(400, "'reason' must be a string")
        remember = body.get("remember", False)
        if not isinstance(remember, bool):
            raise ApiError(400, "'remember' must be a boolean")
        broker = self.rt.approvals
        if not isinstance(broker, QueueBroker):
            raise ApiError(409, "this runtime does not accept approvals from the web")
        if not broker.resolve(approval_id, approved, reason[:2000], remember=remember and approved, by="web"):
            raise ApiError(404, "no such pending approval (it may have timed out or been resolved)")
        return self.out({"ok": True, "id": approval_id, "approved": approved})

    async def list_questions(self, request: Request) -> Response:
        return self.out({"questions": [q.model_dump(mode="json") for q in self.questions_pending()]})

    async def answer_questions(self, request: Request) -> Response:
        question_id = request.path_params["question_id"]
        body = await self.body(request)
        answers = body.get("answers")
        if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers) or len(answers) > 50:
            raise ApiError(400, "'answers' must be a list of strings")
        broker = self.rt.questions
        if not isinstance(broker, QueueQuestionBroker):
            raise ApiError(409, "this runtime does not accept answers from the web")
        if not broker.answer(question_id, [a.strip()[:5000] for a in answers]):
            raise ApiError(404, "no such pending question set (it may have timed out or been answered)")
        return self.out({"ok": True, "id": question_id})

    # ---------------------------------------------------------------- API: checkpoints & stream

    async def checkpoint_diff(self, request: Request) -> Response:
        record = self.rt.store.get_checkpoint(request.path_params["cp_id"])
        if record is None:
            raise ApiError(404, "no such checkpoint")
        full = request.query_params.get("full", "").lower() in ("1", "true", "yes")
        try:
            diff = await self.rt.checkpoints.diff_since(record.id, stat=not full)
        except KeyError as exc:
            raise ApiError(404, "no such checkpoint") from exc
        except Exception as exc:
            raise ApiError(500, f"could not compute the diff: {exc}") from exc
        truncated = len(diff) > DIFF_LIMIT_CHARS
        return self.out({
            "checkpoint": record.model_dump(mode="json"),
            "full": full,
            "diff": diff[:DIFF_LIMIT_CHARS],
            "truncated": truncated,
        })

    async def stream(self, request: Request) -> Response:
        task_id = request.query_params.get("task_id") or None
        frames = sse_events(self.rt.bus, task_id=task_id, should_stop=lambda: self.closing)
        return StreamingResponse(frames, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"})


def create_app(rt: Runtime, token: str) -> Starlette:
    """Build the ASGI app for ``rt``. The caller owns (and closes) the runtime."""
    d = Dashboard(rt, token)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        await d.shutdown()

    async def not_found(request: Request, exc: Exception) -> Response:
        status = exc.status_code if isinstance(exc, HTTPException) else 404
        return d.error(status, "not found" if status == 404 else "method not allowed")

    routes = [
        Route("/", d.index, methods=["GET"]),
        Route("/static/{name}", d.static, methods=["GET"]),
        Route("/api/status", d.api(d.status), methods=["GET"]),
        Route("/api/tasks", d.api(d.list_tasks), methods=["GET"]),
        Route("/api/tasks", d.api(d.create_task), methods=["POST"]),
        Route("/api/tasks/{task_id}", d.api(d.get_task_detail), methods=["GET"]),
        Route("/api/tasks/{task_id}/events", d.api(d.task_events), methods=["GET"]),
        Route("/api/tasks/{task_id}/report", d.api(d.task_report), methods=["GET"]),
        Route("/api/tasks/{task_id}/stop", d.api(d.stop_task), methods=["POST"]),
        Route("/api/tasks/{task_id}/cancel", d.api(d.cancel_task), methods=["POST"]),
        Route("/api/tasks/{task_id}/resume", d.api(d.resume_task), methods=["POST"]),
        Route("/api/approvals", d.api(d.list_approvals), methods=["GET"]),
        Route("/api/approvals/{approval_id}", d.api(d.resolve_approval), methods=["POST"]),
        Route("/api/questions", d.api(d.list_questions), methods=["GET"]),
        Route("/api/questions/{question_id}", d.api(d.answer_questions), methods=["POST"]),
        Route("/api/checkpoints/{cp_id}/diff", d.api(d.checkpoint_diff), methods=["GET"]),
        Route("/api/stream", d.api(d.stream), methods=["GET"]),
    ]
    app = Starlette(
        routes=routes,
        middleware=[Middleware(_SecurityHeaders)],
        exception_handlers={404: not_found, 405: not_found},
        lifespan=lifespan,
    )
    app.state.dashboard = d
    return app


# -------------------------------------------------------------------- serving


class _Server(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, dashboard: Dashboard) -> None:
        super().__init__(config)
        self._dashboard = dashboard

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        # Only flip a flag here (signal context); open event streams end within a second.
        self._dashboard.closing = True
        super().handle_exit(sig, frame)


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _url_host(host: str) -> str:
    if host in ("0.0.0.0", "::", ""):  # noqa: S104 - display only
        return "127.0.0.1"
    return f"[{host}]" if ":" in host else host


def serve(workspace: Path, overrides: dict[str, Any] | None, host: str | None = None, port: int | None = None) -> int:
    """Run the dashboard until interrupted. Returns a process exit code."""
    approvals = QueueBroker()
    questions = QueueQuestionBroker()
    rt = Runtime.open(workspace, overrides, interactive=True, approvals=approvals, questions=questions)
    try:
        approvals.timeout_s = rt.settings.permissions.approval_timeout_s
        bind_host = host or rt.settings.ui.host
        bind_port = rt.settings.ui.port if port is None else port
        if bind_port == 0:
            bind_port = _free_port(bind_host)
        env_token = os.environ.get(TOKEN_ENV, "").strip()
        token = env_token or secrets.token_urlsafe(24)
        app = create_app(rt, token)
        dashboard: Dashboard = app.state.dashboard
    except BaseException:
        asyncio.run(rt.aclose())
        raise

    base = f"http://{_url_host(bind_host)}:{bind_port}/"
    print(f"AI Engineer dashboard for {rt.workspace}", file=sys.stderr)
    if env_token:
        print(f"  Open {base}?token=<value of {TOKEN_ENV}>", file=sys.stderr)
    else:
        print(f"  Open {base}?token={token}", file=sys.stderr)
        print("  (the token grants control of the agent; keep this URL private)", file=sys.stderr)
    if bind_host not in LOOPBACK_HOSTS:
        print(f"  WARNING: listening on {bind_host}, not loopback; traffic is unencrypted.", file=sys.stderr)
    print("  Press Ctrl+C to stop.", file=sys.stderr, flush=True)

    config = uvicorn.Config(
        app, host=bind_host, port=bind_port, log_level="warning", lifespan="on", access_log=False,
        server_header=False, proxy_headers=False, timeout_graceful_shutdown=5,
    )
    server = _Server(config, dashboard)
    try:
        # uvicorn re-raises SIGINT after its graceful shutdown (which also stops a running task
        # through the app lifespan); asyncio then cancels the main task. The runtime is therefore
        # closed afterwards, outside that loop, so the close itself is never cancelled.
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        return 0 if server.started else 130
    except SystemExit as exc:  # uvicorn exits when it cannot bind
        return exc.code if isinstance(exc.code, int) else 1
    finally:
        asyncio.run(rt.aclose())
    return 0 if server.started else 1
