"""`aie` command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..config import find_workspace_root, global_config_dir
from ..core.errors import AIEngineerError, ConfigError, StateError
from ..tasks.store import RESUMABLE_STATUSES, Schedule, TaskStatus

if TYPE_CHECKING:
    from ..runtime import Runtime

EXIT = {
    TaskStatus.COMPLETED: 0,
    TaskStatus.FAILED: 1,
    TaskStatus.COMPLETED_UNVERIFIED: 2,
    TaskStatus.BLOCKED: 3,
    TaskStatus.INTERRUPTED: 3,
    TaskStatus.PENDING: 0,
    TaskStatus.CANCELLED: 130,
}


def _out(text: str = "") -> None:
    print(text)


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}
    if getattr(args, "mode", None):
        o.setdefault("permissions", {})["mode"] = args.mode
    if getattr(args, "model", None):
        o.setdefault("models", {}).setdefault("roles", {})["default"] = [m.strip() for m in args.model.split(",")]
    if getattr(args, "review_model", None):
        o.setdefault("models", {}).setdefault("roles", {})["reviewer"] = [m.strip() for m in args.review_model.split(",")]
    if getattr(args, "debug", False):
        o.setdefault("observability", {})["debug"] = True
    if getattr(args, "max_steps", None):
        o.setdefault("agent", {})["max_steps"] = args.max_steps
    return o


def _workspace(args: argparse.Namespace) -> Path:
    if getattr(args, "workspace", None):
        return Path(args.workspace).resolve()
    return find_workspace_root(Path.cwd())


def _open(args: argparse.Namespace, interactive: bool = False) -> Runtime:
    from ..runtime import Runtime

    approvals = questions = None
    if interactive:
        from .interactive import terminal_approvals, terminal_questions

        approvals, questions = terminal_approvals(), terminal_questions()
    return Runtime.open(_workspace(args), _overrides(args), interactive=interactive, approvals=approvals, questions=questions)


def _attach_renderer(rt: Any, args: argparse.Namespace) -> None:
    if getattr(args, "quiet", False):
        return
    from .render import TerminalRenderer

    rt.bus.subscribe(TerminalRenderer(verbose=getattr(args, "verbose", False)))


def _install_stop_handler(rt: Any, daemon_stop: asyncio.Event | None = None) -> None:
    loop = asyncio.get_running_loop()
    presses = {"n": 0}

    def handler() -> None:
        presses["n"] += 1
        if daemon_stop is not None:
            daemon_stop.set()  # the daemon exits once the current task has stopped
        if presses["n"] == 1:
            _err("\n  ■ Stopping after the current step (state is saved; resume with `aie resume`). Press Ctrl+C again to force quit.")
            rt.tool_ctx.cancel.cancel("stop requested (Ctrl+C)")
            if hasattr(rt.approvals, "deny_all"):
                rt.approvals.deny_all("stopped")
        else:
            _err("  ■ Forced exit.")
            os._exit(130)

    with contextlib.suppress(NotImplementedError, RuntimeError):
        loop.add_signal_handler(signal.SIGINT, handler)


async def _execute(rt: Any, task_id: str, args: argparse.Namespace, stop_after: str | None = None, daemon_stop: asyncio.Event | None = None) -> int:
    _install_stop_handler(rt, daemon_stop)
    try:
        task = await rt.orchestrator.run(task_id, stop_after=stop_after)
    finally:
        await rt.aclose()
    state = task.state or {}
    _out("")
    _out(f"Task {task.id}: {task.status}" + (f" — {task.error}" if task.error else ""))
    if state.get("answer"):
        _out("")
        _out(state["answer"].get("answer", ""))
        if state["answer"].get("evidence"):
            _out("\nEvidence: " + ", ".join(state["answer"]["evidence"]))
    gates = state.get("gates") or {}
    for g in gates.get("results", []):
        _out(f"  {g['status']:<13} {g['name']:<15} {g['detail']}")
    if state.get("report_path"):
        _out(f"\nReport: {state['report_path']}")
    if task.status in RESUMABLE_STATUSES:
        _out(f"Resume with: aie resume {task.id}")
    return EXIT.get(task.status, 1)


# ---------------------------------------------------------------- commands


def cmd_init(args: argparse.Namespace) -> int:
    rt = _open(args)

    async def go() -> None:
        profile = await rt.ensure_profile(refresh=True)
        stats = await rt.refresh_index()
        rt.export_snapshots()
        _out(f"Initialized {rt.state_dir}")
        _out(profile.summary)
        if stats:
            _out(f"Indexed {stats.get('added', 0) + stats.get('updated', 0)} file(s).")
        if not rt.router.configured_roles():
            _out("\nNo model configured yet. Set AIE_MODEL=provider:model (see `aie providers models`) or edit .agent/config.toml.")
        await rt.aclose()

    asyncio.run(go())
    return 0


def cmd_run(args: argparse.Namespace, stop_after: str | None = None) -> int:
    description = " ".join(args.task) if isinstance(args.task, list) else args.task
    if description == "-":
        description = sys.stdin.read()
    interactive = sys.stdin.isatty() and not args.no_interactive

    async def go() -> int:
        rt = _open(args, interactive=interactive)
        rt.router.for_role("default")  # fail fast with a helpful message if no model is configured
        task = rt.create_task(description, priority=args.priority, queue=args.queue)
        if args.queue:
            _out(f"Queued {task.id}")
            await rt.aclose()
            return 0
        _attach_renderer(rt, args)
        _err(f"Task {task.id} · mode {rt.settings.permissions.mode} · models: {', '.join(f'{k}={v[0]}' for k, v in rt.router.configured_roles().items() if k in ('default', 'coder', 'reviewer'))}")
        return await _execute(rt, task.id, args, stop_after)

    return asyncio.run(go())


def cmd_ask(args: argparse.Namespace) -> int:
    args.mode = args.mode or "safe"
    args.queue = False
    return cmd_run(args)


def cmd_plan(args: argparse.Namespace) -> int:
    args.queue = False
    code = cmd_run(args, stop_after="plan")
    return code


def cmd_resume(args: argparse.Namespace) -> int:
    interactive = sys.stdin.isatty() and not args.no_interactive

    async def go() -> int:
        rt = _open(args, interactive=interactive)
        task_id = args.task_id
        if not task_id:
            candidates = rt.store.list_tasks(status=[TaskStatus.INTERRUPTED, TaskStatus.BLOCKED, TaskStatus.PENDING], top_level=True, limit=1)
            if not candidates:
                _err("No resumable task found.")
                await rt.aclose()
                return 64
            task_id = candidates[0].id
        task = rt.store.require_task(task_id)
        if task.status == TaskStatus.FAILED:
            rt.store.update_task(task.id, status=TaskStatus.INTERRUPTED)
        if getattr(args, "answer", None):
            # answers to the questions a blocked task is waiting on, in order
            state = dict(task.state or {})
            state["supplied_answers"] = list(args.answer)
            rt.store.update_task(task.id, state=state)
        _attach_renderer(rt, args)
        return await _execute(rt, task.id, args)

    return asyncio.run(go())


def cmd_status(args: argparse.Namespace) -> int:
    rt = _open(args)
    tasks = rt.store.list_tasks(top_level=True, limit=args.limit)
    if not tasks:
        _out("No tasks yet. Start one with: aie run \"<task>\"")
    for t in tasks:
        stage = f" [{t.stage}]" if t.status in (TaskStatus.RUNNING, TaskStatus.INTERRUPTED, TaskStatus.BLOCKED) and t.stage else ""
        _out(f"{t.id}  {t.status:<20}{stage:<12} {t.updated}  {t.title[:70]}")
    asyncio.run(rt.aclose())
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    rt = _open(args)
    try:
        if args.tasks_cmd == "show":
            task = rt.store.require_task(args.task_id)
            data = task.model_dump(exclude={"state"} if not args.full else set())
            data["subtasks"] = [{"id": s.id, "status": s.status, "title": s.title} for s in rt.store.subtasks(task.id)]
            _out(json.dumps(data, indent=2, default=str))
        elif args.tasks_cmd in ("stop", "cancel"):
            task = rt.store.require_task(args.task_id)
            if task.status == TaskStatus.RUNNING:
                rt.request_stop(task.id, cancel=args.tasks_cmd == "cancel")
                _out(f"{args.tasks_cmd} requested for {task.id}; the agent stops after its current step.")
            elif args.tasks_cmd == "cancel" and task.status not in (TaskStatus.COMPLETED, TaskStatus.COMPLETED_UNVERIFIED):
                rt.store.update_task(task.id, status=TaskStatus.CANCELLED)
                _out(f"cancelled {task.id}")
            else:
                _out(f"{task.id} is {task.status}; nothing to stop")
        elif args.tasks_cmd == "add":
            task = rt.create_task(" ".join(args.description), priority=args.priority, queue=True)
            _out(f"Queued {task.id}")
        elif args.tasks_cmd == "events":
            for e in rt.store.events(rt.store.require_task(args.task_id).id, limit=args.limit):
                _out(f"{e['ts']} {e['type']:<20} {e['message']}")
        else:
            for t in rt.store.list_tasks(top_level=not args.all, limit=args.limit):
                _out(f"{t.id}  {t.status:<20} p{t.priority:<3} {t.title[:80]}")
    finally:
        asyncio.run(rt.aclose())
    return 0


async def _run_queue(args: argparse.Namespace, once: bool) -> int:
    from ..runtime import Runtime

    code = 0
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def idle_handler() -> None:
        _err("\n  ■ Daemon stopping.")
        stop.set()

    while not stop.is_set():
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGINT, idle_handler)  # replaced by the per-task handler while a task runs
        rt = Runtime.open(_workspace(args), _overrides(args))
        for sched in rt.store.due_schedules():
            task = rt.create_task(sched.description, queue=True)
            rt.store.mark_schedule_run(sched.id, task.id)
            _err(f"Schedule {sched.id}: queued {task.id}")
        queued = rt.store.next_queued()
        if queued is None:
            await rt.aclose()
            if once:
                return code
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=args.interval)
            continue
        _attach_renderer(rt, args)
        try:
            code = max(code, await _execute(rt, queued.id, args, daemon_stop=None if once else stop))
        except ConfigError as exc:  # e.g. another daemon picked the same task first
            _err(f"Skipping {queued.id}: {exc}")
    return code


def cmd_queue(args: argparse.Namespace) -> int:
    return asyncio.run(_run_queue(args, once=True))


def cmd_daemon(args: argparse.Namespace) -> int:
    _err(f"Daemon started (poll every {args.interval}s). Ctrl+C to stop.")
    try:
        return asyncio.run(_run_queue(args, once=False))
    except KeyboardInterrupt:
        return 0


def cmd_report(args: argparse.Namespace) -> int:
    rt = _open(args)
    try:
        task = rt.store.require_task(args.task_id)
        path = (task.state or {}).get("report_path")
        if not path or not Path(path).exists():
            _err("No report for this task yet.")
            return 1
        _out(Path(path).read_text(encoding="utf-8"))
        return 0
    finally:
        asyncio.run(rt.aclose())


def cmd_checkpoints(args: argparse.Namespace) -> int:
    rt = _open(args)

    async def go() -> int:
        try:
            if args.cp_cmd == "restore":
                safety, paths = await rt.checkpoints.restore(args.checkpoint_id)
                _out(f"Restored {args.checkpoint_id} ({len(paths)} path(s) changed). Undo with: aie checkpoints restore {safety.id}")
            elif args.cp_cmd == "diff":
                _out(await rt.checkpoints.diff_since(args.checkpoint_id, stat=args.stat))
            else:
                for cp in rt.checkpoints.list_checkpoints(args.task):
                    _out(f"{cp.id}  {cp.created}  {'verified ' if cp.verified else ''}{cp.kind:<5} {cp.label}")
            return 0
        finally:
            await rt.aclose()

    return asyncio.run(go())


def cmd_memory(args: argparse.Namespace) -> int:
    rt = _open(args)
    try:
        mem = rt.memory
        if mem is None:
            _err("memory is unavailable")
            return 1
        if args.mem_cmd == "search":
            for item in mem.search(" ".join(args.query), limit=args.limit):
                _out(f"{item.id}  {item.render()}")
        elif args.mem_cmd == "add":
            item = mem.add(layer=args.layer, kind=args.kind, content=" ".join(args.content), source="user", confidence=args.confidence)
            _out(f"added {item.id}")
        elif args.mem_cmd == "forget":
            ok = mem.project.forget(args.item_id) or (mem.global_store.forget(args.item_id) if mem.global_store else False)
            _out("forgotten" if ok else "not found")
        elif args.mem_cmd == "update":
            store = mem.project if mem.project.get(args.item_id) else mem.global_store
            if store is None or store.get(args.item_id) is None:
                _err("not found")
                return 1
            item = store.update(args.item_id, content=" ".join(args.content), confidence=args.confidence)
            _out(f"updated {args.item_id} → {item.id} (v{item.version})")
        elif args.mem_cmd == "invalidate":
            if not (args.layer or args.kind or args.id or args.all):
                _err("say what to invalidate: --layer, --kind, --id or --all")
                return 64
            count = mem.invalidate(layer=args.layer, kind=args.kind, item_id=args.id)
            _out(f"invalidated {count} item(s); they stay in history and are flagged STALE until re-confirmed")
        elif args.mem_cmd == "history":
            for item in mem.project.history(args.item_id):
                _out(f"v{item.version} {item.updated} active={item.active}: {item.content}")
        else:
            for store in [mem.project, *( [mem.global_store] if mem.global_store else [])]:
                for item in store.list(layer=args.layer, limit=args.limit):
                    _out(f"{item.id}  {item.render()}")
        return 0
    finally:
        asyncio.run(rt.aclose())


def cmd_providers(args: argparse.Namespace) -> int:
    rt = _open(args)

    async def go() -> int:
        try:
            if args.prov_cmd == "models":
                names = [args.provider] if args.provider else rt.router.registry.names()
                for name in names:
                    try:
                        provider = rt.router.registry.get(name)
                        models = await asyncio.wait_for(provider.list_models(), 20)
                        _out(f"{name} ({provider.config.type}): " + (", ".join(models[:60]) or "(none reported)"))
                    except Exception as exc:
                        _out(f"{name}: unavailable — {exc}")
            elif args.prov_cmd == "test":
                from ..core.types import Message
                from ..models.base import ModelRequest

                role = args.role or "default"
                model = rt.router.for_role(role)
                started = time.monotonic()
                resp = await model.generate(ModelRequest(messages=[Message.user("Reply with exactly: OK")], max_tokens=64, metadata={"role": role}))
                _out(f"{role}: {model.last_ref} answered {resp.text()[:80]!r} in {time.monotonic() - started:.1f}s (tokens in/out {resp.usage.input_tokens}/{resp.usage.output_tokens}{' est.' if resp.usage.estimated else ''})")
            else:
                for name in rt.router.registry.names():
                    cfg = rt.settings.models.providers.get(name)
                    key = f" key={cfg.api_key_env} ({'set' if os.environ.get(cfg.api_key_env) else 'NOT SET'})" if cfg and cfg.api_key_env else ""
                    _out(f"{name}: type={cfg.type if cfg else '?'}{' url=' + cfg.base_url if cfg and cfg.base_url else ''}{key}")
                roles = rt.router.configured_roles()
                _out("Roles: " + ("; ".join(f"{r} → {' → '.join(c)}" for r, c in roles.items()) if roles else "none configured (set AIE_MODEL)"))
            return 0
        finally:
            await rt.aclose()

    return asyncio.run(go())


def _latest_task_id(rt: Any) -> str | None:
    tasks = rt.store.list_tasks(top_level=True, limit=1)
    return tasks[0].id if tasks else None


def cmd_logs(args: argparse.Namespace) -> int:
    rt = _open(args)

    async def go() -> int:
        try:
            task_id = args.task_id or _latest_task_id(rt)
            if task_id is None:
                _err("No tasks yet.")
                return 1
            task = rt.store.require_task(task_id)
            types = {t.upper() for t in args.type or []}

            def show(events: list[dict[str, Any]]) -> None:
                for e in events:
                    if types and e["type"] not in types:
                        continue
                    _out(f"{e['ts']} {e['type']:<22} {e['message']}")

            events, _, _ = rt.store.events_page(task.id, None, args.limit)
            show(events)
            last = events[-1]["id"] if events else None
            while args.follow:
                await asyncio.sleep(1.0)
                more, _, _ = rt.store.events_page(task.id, last, 500)
                show(more)
                last = more[-1]["id"] if more else last
                current = rt.store.require_task(task.id)
                if not more and current.status != TaskStatus.RUNNING:
                    _out(f"-- task {current.status}")
                    break
            return 0
        finally:
            await rt.aclose()

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        return 0


def cmd_restore(args: argparse.Namespace) -> int:
    args.cp_cmd = "restore"
    return cmd_checkpoints(args)


def cmd_project(args: argparse.Namespace) -> int:
    rt = _open(args)

    async def go() -> int:
        try:
            if args.project_cmd == "invalidate":
                return await _invalidate_project(rt)
            profile = await rt.ensure_profile(refresh=args.refresh)
            _out(f"Workspace: {rt.workspace}")
            if rt.git is not None:
                status = await rt.git.status()
                _out(f"Git: branch {status.branch or '(detached)'}; {'clean' if status.clean else f'{len(status.entries)} uncommitted path(s)'}")
            else:
                _out("Git: not a repository (file-backup checkpoints)")
            data = profile.model_dump() if profile is not None else {}
            for key in ("primary_language", "languages", "frameworks", "package_managers", "file_count"):
                if data.get(key):
                    value = data[key]
                    if isinstance(value, dict):
                        value = ", ".join(
                            f"{k} ({v.get('files')} files)" if isinstance(v, dict) and "files" in v else f"{k} {v}"
                            for k, v in list(value.items())[:8]
                        )
                    elif isinstance(value, list):
                        value = ", ".join(str(v) for v in value[:12])
                    _out(f"{key.replace('_', ' ').capitalize()}: {value}")
            for kind, vc in rt.validation.checks().items():
                _out(f"  {kind}: {vc.command if vc else 'not detected'}" + ("" if vc is None or vc.available else f" (unavailable: {vc.unavailable_reason})"))
            if rt.index is not None:
                stats = rt.index.stats()
                _out("Index: " + ", ".join(f"{k} {v}" for k, v in stats.items() if isinstance(v, (int, float, str)))[:300])
            counts: dict[str, int] = {}
            for t in rt.store.list_tasks(top_level=True, limit=1000):
                counts[str(t.status)] = counts.get(str(t.status), 0) + 1
            _out("Tasks: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"))
            if rt.memory is not None:
                _out(f"Memory: {rt.memory.project.count()} project item(s)")
            return 0
        finally:
            await rt.aclose()

    return asyncio.run(go())


async def _invalidate_project(rt: Any) -> int:
    """Drop cached project knowledge so the next run re-learns it from the repository."""
    removed = []
    profile = rt.state_dir / "project.json"
    if profile.exists():
        profile.unlink()
        removed.append("repository profile")
    if rt.index is not None:
        with contextlib.suppress(Exception):
            rt.index.close()
        rt.index = None
    for path in sorted((rt.state_dir / "indexes").glob("index.db*")):
        with contextlib.suppress(OSError):
            path.unlink()
            if path.name == "index.db":
                removed.append("code index")
    invalidated = rt.memory.invalidate(layer="project") if rt.memory is not None else 0
    _out("Invalidated: " + ", ".join([*removed, f"{invalidated} project memory item(s)"]))
    _out("They are rebuilt from the repository on the next run (or now with `aie index` / `aie project --refresh`).")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    from ..tester.models import CheckKind

    rt = _open(args)
    order = [CheckKind.TEST, CheckKind.LINT, CheckKind.TYPECHECK, CheckKind.BUILD, CheckKind.FORMAT, CheckKind.AUDIT]
    if args.kind == ["all"]:
        kinds = order
    elif args.kind:
        kinds = [CheckKind(k) for k in args.kind]
    else:
        kinds = [CheckKind.TEST, CheckKind.LINT, CheckKind.TYPECHECK]

    async def go() -> int:
        try:
            await rt.ensure_profile()
            results = []
            for kind in kinds:
                result = await rt.validation.run_check(kind, targeted_files=args.files or None)
                result.output_tail = rt.redactor.redact_text(result.output_tail or "")
                results.append(result)
                counts = "" if result.passed is None and result.failed is None else f" ({result.passed or 0} passed, {result.failed or 0} failed)"
                _out(f"{result.outcome():<17} {kind:<10}{counts} {result.summary or result.status}" + (f"\n                  $ {result.command}" if result.command else ""))
                if args.verbose and result.output_tail and not result.ok():
                    _out(result.output_tail[-3000:])
            if args.json:
                _out(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
            outcomes = {r.outcome() for r in results}
            if "FAIL" in outcomes:
                return 1
            if outcomes & {"ENVIRONMENT_ERROR", "UNAVAILABLE", "UNVERIFIED", "CANCELLED"}:
                return 2  # nothing failed, but not everything was verified
            return 0
        finally:
            await rt.aclose()

    return asyncio.run(go())


def cmd_review(args: argparse.Namespace) -> int:
    from ..reviewer.review import review_change
    from ..security.secrets import redact_secret_files_in_diff

    rt = _open(args)

    async def go() -> int:
        try:
            if args.checkpoint:
                diff = await rt.checkpoints.diff_since(args.checkpoint)
                changed = await rt.checkpoints.changed_files_since(args.checkpoint)
            elif rt.git is None:
                _err("Not a git repository: review a checkpoint instead (aie review --checkpoint <id>).")
                return 64
            elif args.staged:
                diff = await rt.git.diff(staged=True)
                changed = [f for f in (await rt.git.run("diff", "--cached", "--name-only", "-z")).split("\0") if f]
            else:
                base = args.base or "HEAD"
                diff = await rt.git.diff_since(base)
                changed = await rt.git.changed_files_since(base)
            diff = redact_secret_files_in_diff(diff, rt.tool_ctx.guard.is_secret_file)
            if not diff.strip():
                _out("Nothing to review: no changes.")
                return 0
            await rt.ensure_profile()
            result, findings = await review_change(
                None if args.no_model else rt.router.for_role("reviewer"),
                task=args.task or "Review these changes for correctness, security and maintainability.",
                criteria=args.criterion or [], diff=diff, validation_summary="(validation not run by `aie review`; use `aie test`)",
                deleted_files=[f for f in changed if not (rt.workspace / f).exists()],
                repo_has_tests=bool(rt.profile and rt.profile.test_files), project_brief=rt.context.project_brief(max_chars=2000),
                use_model=not args.no_model,
            )
            if args.json:
                _out(json.dumps({"review": result.model_dump(), "security_findings": [f.to_dict() for f in findings]}, indent=2))
            else:
                _out(result.render())
            if result.source in ("deterministic", "deterministic-only"):
                _out("UNVERIFIED: no independent model review ran (automated checks only).")
                return 2 if result.verdict == "approve" else 1
            return 0 if result.verdict == "approve" else 1
        finally:
            await rt.aclose()

    return asyncio.run(go())


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import FAIL, Doctor, as_json, render

    checks = Doctor(_workspace(args), _overrides(args), connect=not args.offline, timeout_s=args.timeout).run()
    _out(json.dumps(as_json(checks), indent=2) if args.json else render(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


def cmd_inspect(args: argparse.Namespace) -> int:
    rt = _open(args)

    async def go() -> int:
        try:
            profile = await rt.ensure_profile(refresh=True)
            if args.json:
                _out(profile.model_dump_json(indent=2))
            else:
                from ..tools.builtin.search import RepoOverviewInput, RepoOverviewTool

                result = await RepoOverviewTool().run(RepoOverviewInput(), rt.tool_ctx)
                _out(result.content)
            return 0
        finally:
            await rt.aclose()

    return asyncio.run(go())


def cmd_index(args: argparse.Namespace) -> int:
    rt = _open(args)

    async def go() -> int:
        try:
            stats = await rt.refresh_index()
            _out(json.dumps(stats, indent=2))
            if rt.index is not None:
                _out(json.dumps(rt.index.stats(), indent=2, default=str))
            return 0
        finally:
            await rt.aclose()

    return asyncio.run(go())


def cmd_config(args: argparse.Namespace) -> int:
    from ..config import load_settings

    ws = _workspace(args)
    if args.config_cmd == "path":
        _out(f"project: {ws / '.agent' / 'config.toml'}\nglobal: {global_config_dir() / 'config.toml'}")
        return 0
    settings = load_settings(ws, _overrides(args))
    _out(json.dumps(settings.model_dump(mode="json"), indent=2))
    return 0


def cmd_improve(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    reports = ws / ".agent" / "reports"
    files = sorted(reports.glob(f"{args.task_id or ''}*.improvements.json"))
    if not files:
        _out("No improvement suggestions recorded yet.")
        return 0
    for f in files[-args.limit:]:
        data = json.loads(f.read_text(encoding="utf-8"))
        _out(f"{f.name.removesuffix('.improvements.json')}:")
        for finding in data.get("findings", []):
            _out(f"  - [{finding['kind']}] {finding['observation']} → {finding['suggestion']}")
    _out("\nThese are advisory. Changes to the agent itself must go through normal testing and review.")
    return 0


_DURATION = re.compile(r"^(\d+(?:\.\d+)?)([smhd])$")


def _parse_every(text: str) -> float:
    m = _DURATION.match(text.strip())
    if not m:
        raise ConfigError("use a duration like 30m, 6h or 1d")
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def cmd_schedule(args: argparse.Namespace) -> int:
    rt = _open(args)
    try:
        if args.sched_cmd == "add":
            interval = _parse_every(args.every)
            sched = rt.store.add_schedule(Schedule(description=" ".join(args.description), interval_s=interval, next_run=time.time() + (0 if args.now else interval)))
            _out(f"Added {sched.id}; run `aie daemon` to execute schedules.")
        elif args.sched_cmd == "remove":
            _out("removed" if rt.store.delete_schedule(args.schedule_id) else "not found")
        elif args.sched_cmd in ("enable", "disable"):
            _out("ok" if rt.store.set_schedule_enabled(args.schedule_id, args.sched_cmd == "enable") else "not found")
        else:
            for s in rt.store.schedules():
                _out(f"{s.id}  every {s.interval_s / 3600:.2f}h  next {time.strftime('%Y-%m-%d %H:%M', time.localtime(s.next_run))}  {'on ' if s.enabled else 'off'} {s.description[:70]}")
        return 0
    finally:
        asyncio.run(rt.aclose())


def cmd_bench(args: argparse.Namespace) -> int:
    from ..benchmark.runner import main as bench_main

    return bench_main(args)


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from .web.server import serve
    except ImportError as exc:
        _err(f"The web UI needs extra packages: pip install 'ai-engineer[web]' ({exc})")
        return 1
    return serve(_workspace(args), _overrides(args), host=args.host, port=args.port)


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aie", description="AI Engineer: a self-hosted, model-agnostic autonomous software engineering agent.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-C", "--workspace", help="workspace directory (default: nearest repository root)")
    p.add_argument("--debug", action="store_true", help="verbose logging and full traces (secrets are still redacted)")
    sub = p.add_subparsers(dest="command", required=True)

    def run_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--mode", choices=["safe", "assisted", "developer", "autonomous"])
        sp.add_argument("--model", help="provider:model[,provider:model...] fallback chain for all roles")
        sp.add_argument("--review-model", help="independent reviewer chain")
        sp.add_argument("--max-steps", type=int)
        sp.add_argument("--no-interactive", action="store_true", help="never prompt; actions needing approval are denied")
        sp.add_argument("-v", "--verbose", action="store_true")
        sp.add_argument("-q", "--quiet", action="store_true")
        sp.add_argument("--priority", type=int, default=50)

    sp = sub.add_parser("init", help="create .agent/, detect the project and build the index")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("run", help="run an engineering task end to end")
    sp.add_argument("task", nargs="+", help="task description ('-' reads stdin)")
    run_opts(sp)
    sp.add_argument("--queue", action="store_true", help="only queue the task")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("ask", help="answer a question about the repository (read-only)")
    sp.add_argument("task", nargs="+")
    run_opts(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("plan", help="understand and plan a task without executing it")
    sp.add_argument("task", nargs="+")
    run_opts(sp)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("resume", help="resume an interrupted, blocked or planned task")
    sp.add_argument("task_id", nargs="?")
    sp.add_argument("--answer", action="append", metavar="TEXT", help="answer to a question the task is blocked on (repeat, in order)")
    run_opts(sp)
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("status", help="recent tasks")
    sp.add_argument("--limit", type=int, default=15)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("tasks", aliases=["task"], help="manage tasks and the queue")
    tsub = sp.add_subparsers(dest="tasks_cmd")
    tl = tsub.add_parser("list")
    tl.add_argument("--all", action="store_true", help="include subtasks")
    ts = tsub.add_parser("show")
    ts.add_argument("task_id")
    ts.add_argument("--full", action="store_true")
    for name in ("stop", "cancel"):
        tx = tsub.add_parser(name)
        tx.add_argument("task_id")
    ta = tsub.add_parser("add")
    ta.add_argument("description", nargs="+")
    ta.add_argument("--priority", type=int, default=50)
    te = tsub.add_parser("events")
    te.add_argument("task_id")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--all", action="store_true")
    sp.set_defaults(func=cmd_tasks, tasks_cmd="list")

    sp = sub.add_parser("queue", help="run queued tasks (and due schedules) once")
    run_opts(sp)
    sp.set_defaults(func=cmd_queue, interval=60)

    sp = sub.add_parser("daemon", help="continuously run queued tasks and recurring schedules")
    run_opts(sp)
    sp.add_argument("--interval", type=float, default=60.0)
    sp.set_defaults(func=cmd_daemon)

    sp = sub.add_parser("report", help="print a task's engineering report")
    sp.add_argument("task_id")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("checkpoints", aliases=["checkpoint"], help="list, diff or restore checkpoints")
    csub = sp.add_subparsers(dest="cp_cmd")
    cl = csub.add_parser("list")
    cl.add_argument("--task")
    cr = csub.add_parser("restore")
    cr.add_argument("checkpoint_id")
    cd = csub.add_parser("diff")
    cd.add_argument("checkpoint_id")
    cd.add_argument("--stat", action="store_true")
    sp.set_defaults(func=cmd_checkpoints, cp_cmd="list", task=None)

    sp = sub.add_parser("restore", help="restore a checkpoint (the current state is checkpointed first, so it can be undone)")
    sp.add_argument("checkpoint_id")
    sp.set_defaults(func=cmd_restore, task=None)

    sp = sub.add_parser("logs", help="show a task's activity log (latest task by default)")
    sp.add_argument("task_id", nargs="?")
    sp.add_argument("--limit", type=int, default=200)
    sp.add_argument("--type", action="append", help="only these event types (repeatable), e.g. TEST_FAILED")
    sp.add_argument("-f", "--follow", action="store_true", help="keep printing new events until the task stops")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("project", help="show what the agent knows about this project, or invalidate it")
    psub2 = sp.add_subparsers(dest="project_cmd")
    psub2.add_parser("show")
    psub2.add_parser("invalidate", help="drop the cached profile, code index and project memory")
    sp.add_argument("--refresh", action="store_true", help="re-detect the repository profile")
    sp.set_defaults(func=cmd_project, project_cmd="show")

    sp = sub.add_parser("test", help="run the project's tests and checks (no model involved)")
    sp.add_argument("files", nargs="*", help="restrict to these files where the runner supports it")
    sp.add_argument("--kind", action="append", choices=["test", "lint", "typecheck", "build", "format", "audit", "all"],
                    help="checks to run (repeatable; default: test, lint, typecheck)")
    sp.add_argument("-v", "--verbose", action="store_true", help="print output of failing checks")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("review", help="independently review the current changes")
    sp.add_argument("--base", help="compare the working tree with this ref (default HEAD)")
    sp.add_argument("--staged", action="store_true", help="review only staged changes")
    sp.add_argument("--checkpoint", help="review changes since this checkpoint")
    sp.add_argument("--task", help="what the change is meant to do")
    sp.add_argument("--criterion", action="append", help="acceptance criterion (repeatable)")
    sp.add_argument("--no-model", action="store_true", help="automated checks only (result is UNVERIFIED)")
    sp.add_argument("--json", action="store_true")
    run_opts(sp)
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("memory", help="search and edit agent memory")
    msub = sp.add_subparsers(dest="mem_cmd")
    ms = msub.add_parser("search")
    ms.add_argument("query", nargs="+")
    ma = msub.add_parser("add")
    ma.add_argument("content", nargs="+")
    ma.add_argument("--layer", default="project", choices=["project", "engineering", "decision", "command", "session"])
    ma.add_argument("--kind", default="fact")
    ma.add_argument("--confidence", type=float, default=0.9)
    mf = msub.add_parser("forget")
    mf.add_argument("item_id")
    mu = msub.add_parser("update", help="replace an item's content (keeps its history)")
    mu.add_argument("item_id")
    mu.add_argument("content", nargs="+")
    mu.add_argument("--confidence", type=float)
    mi = msub.add_parser("invalidate", help="flag items as stale so the repository is re-checked")
    mi.add_argument("--layer", choices=["project", "engineering", "decision", "command", "session"])
    mi.add_argument("--kind")
    mi.add_argument("--id")
    mi.add_argument("--all", action="store_true")
    mh = msub.add_parser("history")
    mh.add_argument("item_id")
    ml = msub.add_parser("list")
    ml.add_argument("--layer")
    sp.add_argument("--limit", type=int, default=30)
    sp.set_defaults(func=cmd_memory, mem_cmd="list", layer=None)

    sp = sub.add_parser("providers", help="list providers, their models, or test a role")
    psub = sp.add_subparsers(dest="prov_cmd")
    pm = psub.add_parser("models")
    pm.add_argument("provider", nargs="?")
    pt = psub.add_parser("test")
    pt.add_argument("role", nargs="?")
    psub.add_parser("list")
    sp.add_argument("--model")
    sp.set_defaults(func=cmd_providers, prov_cmd="list")

    sp = sub.add_parser("doctor", help="check environment, configuration and provider health")
    sp.add_argument("--offline", action="store_true", help="skip provider connectivity checks")
    sp.add_argument("--timeout", type=float, default=15.0, help="seconds per provider connectivity check")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("inspect", help="show the detected repository profile")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("index", help="build or refresh the repository index")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("config", help="show effective configuration")
    sp.add_argument("config_cmd", nargs="?", choices=["show", "path"], default="show")
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("improve", help="show retrospective improvement suggestions")
    sp.add_argument("task_id", nargs="?")
    sp.add_argument("--limit", type=int, default=5)
    sp.set_defaults(func=cmd_improve)

    sp = sub.add_parser("schedule", help="recurring maintenance tasks")
    ssub = sp.add_subparsers(dest="sched_cmd")
    sa = ssub.add_parser("add")
    sa.add_argument("description", nargs="+")
    sa.add_argument("--every", required=True, help="interval such as 6h or 1d")
    sa.add_argument("--now", action="store_true", help="first run immediately")
    for name in ("remove", "enable", "disable"):
        sx = ssub.add_parser(name)
        sx.add_argument("schedule_id")
    ssub.add_parser("list")
    sp.set_defaults(func=cmd_schedule, sched_cmd="list")

    sp = sub.add_parser("bench", aliases=["benchmark"], help="run the benchmark suite")
    sp.add_argument("bench_cmd", nargs="?", choices=["run", "list"], default="run")
    sp.add_argument("--suite", default="harness", help="'harness' (scripted, offline) or 'model' (uses configured models)")
    sp.add_argument("--only", help="comma-separated benchmark ids")
    sp.add_argument("--model")
    sp.add_argument("--out", help="directory for results (default: ./benchmark-results)")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("ui", help="start the web dashboard")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--mode", choices=["safe", "assisted", "developer", "autonomous"])
    sp.add_argument("--model")
    sp.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        _err(f"Configuration error: {exc}")
        return 78
    except StateError as exc:
        _err(f"State error: {exc}")
        return 65
    except AIEngineerError as exc:
        _err(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        _err("Interrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
