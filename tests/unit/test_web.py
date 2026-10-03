"""Web dashboard: auth, CSRF, static files, task/approval/question APIs, runs and the event stream."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient

from ai_engineer.config.settings import ModelsSettings
from ai_engineer.core.events import EventType
from ai_engineer.models.registry import ProviderRegistry
from ai_engineer.orchestrator.questions import QueueQuestionBroker
from ai_engineer.providers.scripted import ScriptedProvider
from ai_engineer.runtime import Runtime
from ai_engineer.tasks.store import TaskStatus
from ai_engineer.tools.approval import ApprovalRequest, QueueBroker
from ai_engineer.ui.web.server import COOKIE_NAME, create_app, sse_events

TOKEN = "unit-test-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TERMINAL = {"COMPLETED", "COMPLETED_UNVERIFIED", "FAILED", "CANCELLED", "BLOCKED", "INTERRUPTED"}


def _config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "models": {
            "providers": {"s": {"type": "scripted"}},
            "roles": {"default": ["s:m"]},
            "retry": {"max_attempts": 1, "base_delay_s": 0.0, "max_delay_s": 0.0},
        },
        "agent": {"max_steps": 8},
        "validation": {"dependency_audit": False},
    }
    for key, value in extra.items():
        config.setdefault(key, {}).update(value)
    return config


def open_rt(workspace: Path, provider: ScriptedProvider | None = None, **extra: Any) -> Runtime:
    registry = ProviderRegistry(ModelsSettings())
    registry.register_instance("s", provider or ScriptedProvider("s"))
    return Runtime.open(
        workspace, _config(**extra), use_global_config=False, registry=registry,
        approvals=QueueBroker(timeout_s=5), questions=QueueQuestionBroker(timeout_s=5),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "calc.py").write_text("def sub(a, b):\n    return a - b\n")
    return ws


@pytest.fixture
def rt(workspace: Path) -> Iterator[Runtime]:
    runtime = open_rt(workspace)
    yield runtime
    asyncio.run(runtime.aclose())


@pytest.fixture
def client(rt: Runtime) -> Iterator[TestClient]:
    with TestClient(create_app(rt, TOKEN)) as c:
        yield c


def _create(client: TestClient, description: str = "Explain how sub works", **body: Any) -> dict[str, Any]:
    resp = client.post("/api/tasks", json={"description": description, "start": False, **body}, headers=AUTH)
    assert resp.status_code == 201, resp.text
    return resp.json()["task"]


def _wait_terminal(client: TestClient, task_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"/api/tasks/{task_id}", headers=AUTH).json()
        status = client.get("/api/status", headers=AUTH).json()
        if detail["task"]["status"] in TERMINAL and status["running_task_id"] is None:
            return detail
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish: {detail['task']['status']}")


# ---------------------------------------------------------------- authentication


def test_api_requires_token(client: TestClient) -> None:
    for path in ("/api/status", "/api/tasks", "/api/approvals", "/api/questions", "/api/stream"):
        resp = client.get(path)
        assert resp.status_code == 401, path
        assert resp.json()["error"].startswith("unauthorized")
        assert resp.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/status", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert client.get("/api/status?token=wrong").status_code == 401
    assert client.post("/api/tasks", json={"description": "x"}).status_code == 401
    assert client.get("/api/status", headers=AUTH).status_code == 200
    assert client.get(f"/api/status?token={TOKEN}").status_code == 200


def test_cookie_is_set_from_query_token_and_then_accepted(client: TestClient) -> None:
    page = client.get("/?token=wrong")
    assert page.status_code == 200 and "set-cookie" not in page.headers
    assert client.get("/api/status").status_code == 401

    page = client.get(f"/?token={TOKEN}")
    assert page.status_code == 200
    assert "<title>AI Engineer</title>" in page.text
    cookie = page.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE_NAME}=")
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie.replace("Strict", "strict")
    # the client jar now carries the cookie: no header, no query
    assert client.get("/api/status").status_code == 200
    client.cookies.clear()
    assert client.get("/api/status").status_code == 401


def test_security_headers_and_no_token_leak(client: TestClient) -> None:
    resp = client.get("/", params={"token": TOKEN})
    csp = resp.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"] == "no-referrer"
    status = client.get("/api/status", headers=AUTH)
    assert status.headers["cache-control"] == "no-store"
    assert TOKEN not in status.text


def test_cross_origin_post_is_rejected(client: TestClient) -> None:
    body = {"description": "do something", "start": False}
    evil = client.post("/api/tasks", json=body, headers={**AUTH, "Origin": "http://evil.example"})
    assert evil.status_code == 403
    assert client.post("/api/tasks", json=body, headers={**AUTH, "Origin": "null"}).status_code == 403
    assert client.post("/api/tasks", json=body, headers={**AUTH, "Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/tasks", headers=AUTH).json()["tasks"] == []
    same = client.post("/api/tasks", json=body, headers={**AUTH, "Origin": "http://testserver"})
    assert same.status_code == 201


def test_static_files_allow_list(client: TestClient) -> None:
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert client.get("/static/app.css").headers["content-type"].startswith("text/css")
    assert client.get("/static/index.html").status_code == 200
    for path in ("/static/server.py", "/static/..%2Fserver.py", "/static/%2e%2e%2fserver.py",
                 "/static/..%5Cserver.py", "/static/../server.py", "/static/", "/static/app.js%00"):
        resp = client.get(path)
        assert resp.status_code == 404, path
        assert "def create_app" not in resp.text


def test_unknown_api_route_is_json_404(client: TestClient) -> None:
    resp = client.get("/api/nope", headers=AUTH)
    assert resp.status_code == 404
    assert resp.json() == {"error": "not found"}


# ---------------------------------------------------------------- status & tasks


def test_status(client: TestClient, rt: Runtime) -> None:
    data = client.get("/api/status", headers=AUTH).json()
    assert data["workspace"] == str(rt.workspace)
    assert data["mode"] == "developer"
    assert data["ceiling"] == rt.policy.ceiling.name
    assert data["roles"]["default"] == ["s:m"]
    assert data["git"] is False
    assert data["running_task_id"] is None
    assert "TASK_COMPLETED" in data["event_types"]
    assert set(data["modes"]) == {"safe", "assisted", "developer", "autonomous"}


def test_create_list_detail_events_report(client: TestClient, rt: Runtime) -> None:
    task = _create(client, "Explain how sub works\nwith details", mode="safe", priority=10)
    assert task["status"] == "PENDING" and task["mode"] == "safe" and task["priority"] == 10
    assert task["title"] == "Explain how sub works"
    assert rt.settings.permissions.mode == "developer"  # creating never changes the server default

    listing = client.get("/api/tasks?limit=5", headers=AUTH).json()["tasks"]
    assert [t["id"] for t in listing] == [task["id"]]

    detail = client.get(f"/api/tasks/{task['id']}", headers=AUTH).json()
    assert detail["task"]["description"].startswith("Explain how sub works")
    assert detail["state"]["subtasks"] == [] and detail["state"]["has_report"] is False
    for key in ("checkpoints", "test_runs", "failures", "metrics", "files_changed"):
        assert key in detail

    events = client.get(f"/api/tasks/{task['id']}/events", headers=AUTH).json()
    assert [e["type"] for e in events["events"]] == ["TASK_CREATED"]
    first = events["events"][0]["id"]
    after = client.get(f"/api/tasks/{task['id']}/events?after={first}", headers=AUTH).json()
    assert after["events"] == [] and after["reset"] is False and after["last_id"] == first
    unknown = client.get(f"/api/tasks/{task['id']}/events?after=evt_missing", headers=AUTH).json()
    assert unknown["reset"] is True and len(unknown["events"]) == 1

    assert client.get(f"/api/tasks/{task['id']}/report", headers=AUTH).status_code == 404
    assert client.get("/api/tasks/task_nope", headers=AUTH).status_code == 404
    assert client.get("/api/tasks?limit=abc", headers=AUTH).status_code == 400


def test_events_keep_emission_order_and_paginate(client: TestClient, rt: Runtime) -> None:
    task = _create(client)
    for i in range(30):  # many events share a millisecond: ids/timestamps alone cannot order them
        rt.bus.emit(EventType.INFO, f"step {i}", task_id=task["id"])
    url = f"/api/tasks/{task['id']}/events"
    tail = client.get(f"{url}?limit=10", headers=AUTH).json()
    assert [e["message"] for e in tail["events"]] == [f"step {i}" for i in range(20, 30)]
    assert tail["has_more"] is True and tail["last_id"] == tail["events"][-1]["id"]
    everything = client.get(f"{url}?limit=100", headers=AUTH).json()["events"]
    assert everything[0]["type"] == "TASK_CREATED"
    assert [e["message"] for e in everything[1:]] == [f"step {i}" for i in range(30)]
    page = client.get(f"{url}?after={everything[0]['id']}&limit=5", headers=AUTH).json()
    assert [e["message"] for e in page["events"]] == [f"step {i}" for i in range(5)]
    assert page["has_more"] is True and page["reset"] is False
    last = client.get(f"{url}?after={everything[-1]['id']}", headers=AUTH).json()
    assert last["events"] == [] and last["has_more"] is False
    unknown = client.get(f"{url}?after=ev_unknown&limit=3", headers=AUTH).json()
    assert unknown["reset"] is True and [e["message"] for e in unknown["events"]] == [f"step {i}" for i in range(27, 30)]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"description": ""}, 400),
        ({"description": "   "}, 400),
        ({"description": 5}, 400),
        ({"description": "x", "mode": "godmode"}, 400),
        ({"description": "x", "start": "yes"}, 400),
        ({"description": "x", "priority": "high"}, 400),
        ({"description": "x", "priority": True}, 400),
        ({"description": "x" * 50_001}, 400),
    ],
)
def test_create_task_validation(client: TestClient, body: dict[str, Any], status: int) -> None:
    assert client.post("/api/tasks", json=body, headers=AUTH).status_code == status


def test_create_task_body_format(client: TestClient) -> None:
    bad_json = client.post("/api/tasks", content=b"{not json", headers={**AUTH, "Content-Type": "application/json"})
    assert bad_json.status_code == 400
    form = client.post("/api/tasks", content=b"description=x", headers={**AUTH, "Content-Type": "application/x-www-form-urlencoded"})
    assert form.status_code == 415
    array = client.post("/api/tasks", json=["x"], headers=AUTH)
    assert array.status_code == 400


def test_detail_is_redacted(client: TestClient) -> None:
    secret = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    task = _create(client, f"Rotate the leaked token {secret} in config")
    detail = client.get(f"/api/tasks/{task['id']}", headers=AUTH)
    assert secret not in detail.text and "[REDACTED]" in detail.text
    assert secret not in client.get("/api/tasks", headers=AUTH).text


def test_report_returns_raw_markdown_from_reports_dir_only(client: TestClient, rt: Runtime) -> None:
    task = _create(client)
    reports = rt.state_dir / "reports"
    markdown = "# Report\n\n<script>alert(1)</script> **bold**\n"
    (reports / f"{task['id']}.md").write_text(markdown, encoding="utf-8")
    rt.store.update_task(task["id"], state={"report_path": str(reports / f"{task['id']}.md")})
    resp = client.get(f"/api/tasks/{task['id']}/report", headers=AUTH)
    assert resp.status_code == 200
    # escaping is the client's job: the API returns the Markdown source unchanged
    assert resp.json()["markdown"] == markdown
    assert resp.json()["path"] == f".agent/reports/{task['id']}.md"
    detail = client.get(f"/api/tasks/{task['id']}", headers=AUTH).json()
    assert detail["state"]["has_report"] is True

    other = _create(client, "another task")
    outside = rt.workspace / "notes.md"
    outside.write_text("private notes")
    rt.store.update_task(other["id"], state={"report_path": str(outside)})
    assert client.get(f"/api/tasks/{other['id']}/report", headers=AUTH).status_code == 404
    rt.store.update_task(other["id"], state={"report_path": str(reports / ".." / ".." / "calc.py")})
    assert client.get(f"/api/tasks/{other['id']}/report", headers=AUTH).status_code == 404


# ---------------------------------------------------------------- control


def test_stop_cancel_resume_without_a_run(client: TestClient) -> None:
    task = _create(client)
    assert client.post(f"/api/tasks/{task['id']}/stop", json={}, headers=AUTH).status_code == 409
    cancelled = client.post(f"/api/tasks/{task['id']}/cancel", headers=AUTH)
    assert cancelled.status_code == 200 and cancelled.json()["task"]["status"] == "CANCELLED"
    assert client.post(f"/api/tasks/{task['id']}/resume", headers=AUTH).status_code == 409
    assert client.post(f"/api/tasks/{task['id']}/cancel", headers=AUTH).status_code == 409
    assert client.post("/api/tasks/task_nope/stop", headers=AUTH).status_code == 404


def test_stop_for_task_running_elsewhere_uses_control_table(client: TestClient, rt: Runtime) -> None:
    task = _create(client)
    rt.store.update_task(task["id"], status=TaskStatus.RUNNING)
    resp = client.post(f"/api/tasks/{task['id']}/stop", headers=AUTH)
    assert resp.status_code == 202 and resp.json()["via"] == "control request"
    assert rt.store.pop_control(task["id"]) == "stop"
    client.post(f"/api/tasks/{task['id']}/cancel", headers=AUTH)
    assert rt.store.pop_control(task["id"]) == "cancel"


def test_busy_returns_409(client: TestClient) -> None:
    task = _create(client)
    dashboard = client.app.state.dashboard
    client.portal.call(dashboard.lock.acquire)
    try:
        busy = client.post("/api/tasks", json={"description": "second", "start": True}, headers=AUTH)
        assert busy.status_code == 409 and "busy" in busy.json()["error"]
        assert client.post(f"/api/tasks/{task['id']}/resume", headers=AUTH).status_code == 409
        # creating without starting is still allowed while busy
        assert client.post("/api/tasks", json={"description": "later", "start": False}, headers=AUTH).status_code == 201
    finally:
        dashboard.lock.release()


# ---------------------------------------------------------------- approvals & questions


def test_approval_roundtrip(client: TestClient, rt: Runtime) -> None:
    req = ApprovalRequest(tool="run_command", summary="rm -rf build/", reason="destructive command", risk="high", task_id="task_x")
    future = client.portal.start_task_soon(rt.approvals.request, req)
    pending: list[dict[str, Any]] = []
    for _ in range(100):
        pending = client.get("/api/approvals", headers=AUTH).json()["approvals"]
        if pending:
            break
        time.sleep(0.02)
    assert [p["id"] for p in pending] == [req.id] and pending[0]["tool"] == "run_command"
    assert client.get("/api/status", headers=AUTH).json()["pending_approvals"] == 1

    assert client.post(f"/api/approvals/{req.id}", json={"approved": "yes"}, headers=AUTH).status_code == 400
    assert client.post("/api/approvals/apr_nope", json={"approved": True}, headers=AUTH).status_code == 404
    ok = client.post(f"/api/approvals/{req.id}", json={"approved": True, "reason": "fine", "remember": True}, headers=AUTH)
    assert ok.status_code == 200
    decision = future.result(timeout=5)
    assert decision.approved and decision.remember and decision.by == "web" and decision.reason == "fine"
    assert client.post(f"/api/approvals/{req.id}", json={"approved": True}, headers=AUTH).status_code == 404
    assert client.get("/api/approvals", headers=AUTH).json()["approvals"] == []


def test_question_roundtrip(client: TestClient, rt: Runtime) -> None:
    future = client.portal.start_task_soon(rt.questions.ask, ["Which database?", "Keep the old API?"], "migrating storage")
    pending: list[dict[str, Any]] = []
    for _ in range(100):
        pending = client.get("/api/questions", headers=AUTH).json()["questions"]
        if pending:
            break
        time.sleep(0.02)
    assert pending and pending[0]["questions"] == ["Which database?", "Keep the old API?"]
    qid = pending[0]["id"]
    assert client.post(f"/api/questions/{qid}", json={"answers": "postgres"}, headers=AUTH).status_code == 400
    assert client.post(f"/api/questions/{qid}", json={"answers": [1]}, headers=AUTH).status_code == 400
    assert client.post(f"/api/questions/{qid}", json={"answers": ["Postgres", " yes "]}, headers=AUTH).status_code == 200
    assert future.result(timeout=5) == ["Postgres", "yes"]
    assert client.post(f"/api/questions/{qid}", json={"answers": []}, headers=AUTH).status_code == 404


# ---------------------------------------------------------------- running tasks


def _question_script() -> dict[str, list[Any]]:
    understanding = {
        "summary": "Where is subtraction implemented?", "task_type": "question", "complexity": "trivial",
        "requirements": [], "acceptance_criteria": [],
    }
    return {
        "classifier": [json.dumps(understanding)],
        "coder": [{"tool_calls": [{"name": "submit_answer", "input": {
            "answer": "`sub` is in calc.py.", "evidence": ["calc.py:1"], "confidence": "high"}}]}],
    }


def test_run_question_task_end_to_end(workspace: Path) -> None:
    runtime = open_rt(workspace, ScriptedProvider("s", by_role=_question_script()))
    try:
        with TestClient(create_app(runtime, TOKEN)) as client:
            resp = client.post("/api/tasks", json={"description": "Where is subtraction implemented?", "mode": "safe"}, headers=AUTH)
            assert resp.status_code == 201, resp.text
            task_id = resp.json()["task"]["id"]
            detail = _wait_terminal(client, task_id)
            assert detail["task"]["status"] == "COMPLETED", detail["task"]["error"]
            assert detail["task"]["mode"] == "safe"
            assert detail["state"]["answer"]["evidence"] == ["calc.py:1"]
            assert detail["result"]["status"] == "COMPLETED"
            assert detail["metrics"]["model_calls"] >= 1
            assert runtime.settings.permissions.mode == "developer"  # restored after the run
            types = [e["type"] for e in client.get(f"/api/tasks/{task_id}/events", headers=AUTH).json()["events"]]
            assert "TASK_STARTED" in types and "TASK_COMPLETED" in types
            report = client.get(f"/api/tasks/{task_id}/report", headers=AUTH)
            assert report.status_code == 200 and "sub" in report.json()["markdown"]
            # a completed task cannot be resumed
            assert client.post(f"/api/tasks/{task_id}/resume", headers=AUTH).status_code == 409
            assert not client.app.state.dashboard.lock.locked()
    finally:
        asyncio.run(runtime.aclose())


def test_stop_running_task_then_resume_with_fresh_token(workspace: Path) -> None:
    started = threading.Event()
    understanding = {"summary": "Fix the bug", "task_type": "change", "complexity": "small"}

    async def slow_understanding(req: Any) -> str:
        started.set()
        await asyncio.sleep(0.3)  # an in-flight model call finishes; the stop applies at the next check
        return json.dumps(understanding)

    runtime = open_rt(workspace, ScriptedProvider("s", by_role={"classifier": [slow_understanding]}))
    try:
        with TestClient(create_app(runtime, TOKEN)) as client:
            task_id = client.post("/api/tasks", json={"description": "Fix the bug"}, headers=AUTH).json()["task"]["id"]
            assert started.wait(10)
            assert client.get("/api/status", headers=AUTH).json()["running_task_id"] == task_id
            stop = client.post(f"/api/tasks/{task_id}/stop", headers=AUTH)
            assert stop.status_code == 202 and stop.json()["via"] == "this server"
            detail = _wait_terminal(client, task_id, timeout=20)
            assert detail["task"]["status"] == "INTERRUPTED", detail["task"]
            assert "stop requested" in (detail["task"]["error"] or "")
            assert runtime.tool_ctx.cancel.cancelled

            resumed = client.post(f"/api/tasks/{task_id}/resume", headers=AUTH)
            assert resumed.status_code == 202, resumed.text
            assert not runtime.tool_ctx.cancel.cancelled  # each run gets a fresh token
            detail = _wait_terminal(client, task_id, timeout=20)
            # no coder responses are scripted, so the resumed run blocks, but it is not "stopped" again
            assert detail["task"]["status"] == "BLOCKED", detail["task"]
            assert "stop requested" not in (detail["task"]["error"] or "")
            assert detail["state"]["resumed"] == 1
    finally:
        asyncio.run(runtime.aclose())


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_diff(git_repo: Path) -> None:
    runtime = open_rt(git_repo)
    try:
        cp = asyncio.run(runtime.checkpoints.create("before edit", task_id="task_x"))
        (git_repo / "README.md").write_text("# demo\nnew line\n")
        with TestClient(create_app(runtime, TOKEN)) as client:
            stat = client.get(f"/api/checkpoints/{cp.id}/diff", headers=AUTH).json()
            assert stat["full"] is False and "README.md" in stat["diff"] and stat["truncated"] is False
            full = client.get(f"/api/checkpoints/{cp.id}/diff?full=1", headers=AUTH).json()
            assert "+new line" in full["diff"]
            assert client.get("/api/checkpoints/cp_nope/diff", headers=AUTH).status_code == 404
            assert client.get(f"/api/checkpoints/{cp.id}/diff").status_code == 401
    finally:
        asyncio.run(runtime.aclose())


# ---------------------------------------------------------------- event stream


async def test_sse_generator_formats_filters_and_unsubscribes(rt: Runtime) -> None:
    handlers_before = len(rt.bus._handlers)
    stream = sse_events(rt.bus, task_id="task_a", heartbeat_s=0.05, poll_s=0.02)
    first = await asyncio.wait_for(stream.__anext__(), 2)
    assert first.startswith("retry:")
    assert len(rt.bus._handlers) == handlers_before + 1
    rt.bus.emit(EventType.INFO, "other task", task_id="task_b")
    rt.bus.emit(EventType.TOOL_CALLED, "read calc.py", task_id="task_a", data={"tool": "read_file"})
    frame = await asyncio.wait_for(stream.__anext__(), 2)
    while frame.startswith(":"):
        frame = await asyncio.wait_for(stream.__anext__(), 2)
    lines = frame.strip().split("\n")
    assert lines[1] == "event: TOOL_CALLED"
    payload = json.loads(lines[2].removeprefix("data: "))
    assert payload["message"] == "read calc.py" and payload["task_id"] == "task_a"
    heartbeat = await asyncio.wait_for(stream.__anext__(), 2)
    assert heartbeat == ": keep-alive\n\n"
    await stream.aclose()
    assert len(rt.bus._handlers) == handlers_before


async def test_sse_endpoint_streams_and_stops_on_disconnect(rt: Runtime) -> None:
    app = create_app(rt, TOKEN)
    sent: list[dict[str, Any]] = []
    got_event = asyncio.Event()
    started = asyncio.Event()
    disconnect = asyncio.Event()
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.start":
            started.set()
        if message["type"] == "http.response.body" and b"event: WARNING" in message.get("body", b""):
            got_event.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": "/api/stream", "raw_path": b"/api/stream",
        "root_path": "", "query_string": b"", "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
        "headers": [(b"host", b"testserver"), (b"authorization", f"Bearer {TOKEN}".encode())],
    }
    handlers_before = len(rt.bus._handlers)
    call = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(started.wait(), 5)
    start = next(m for m in sent if m["type"] == "http.response.start")
    headers = dict(start["headers"])
    assert start["status"] == 200
    assert headers[b"content-type"].startswith(b"text/event-stream")
    for _ in range(100):  # wait for the generator to subscribe
        if len(rt.bus._handlers) > handlers_before:
            break
        await asyncio.sleep(0.01)
    rt.bus.emit(EventType.WARNING, "disk almost full", task_id="task_a")
    await asyncio.wait_for(got_event.wait(), 5)
    disconnect.set()
    await asyncio.wait_for(call, 5)
    assert len(rt.bus._handlers) == handlers_before
