"""Subprocess execution: timeouts, bounded output, process-tree termination, sandboxing."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import TerminalSettings
from ..core.cancel import CancellationToken
from ..core.ids import new_id
from ..security.secrets import is_secret_env_name

IS_WINDOWS = sys.platform == "win32"

# Never passed to child processes regardless of configuration.
ALWAYS_STRIP = {
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY", "AIE_WEB_TOKEN",
}

NON_INTERACTIVE_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "PYTHONUNBUFFERED": "1",
    "NO_COLOR": "1",
    "FORCE_COLOR": "0",
    "TERM": "dumb",
    "CI": "true",
    "DEBIAN_FRONTEND": "noninteractive",
    "PIP_NO_INPUT": "1",
    "npm_config_yes": "true",
}


def build_env(settings: TerminalSettings, extra: dict[str, str] | None = None, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    allow = set(settings.env_allow)
    for name in list(env):
        if name in allow:
            continue
        if name in ALWAYS_STRIP or (settings.strip_secret_env and is_secret_env_name(name)):
            env.pop(name, None)
    env.update(NON_INTERACTIVE_ENV)
    if extra:
        env.update(extra)
    return env


@dataclass
class CommandOutcome:
    command: str
    exit_code: int | None
    output: str
    duration_s: float
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False
    total_chars: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.cancelled


class _BoundedBuffer:
    """Keeps the head and tail of a stream; errors usually live at the end."""

    def __init__(self, max_chars: int) -> None:
        self.head_limit = max_chars // 3
        self.tail_limit = max_chars - self.head_limit
        self.head: list[str] = []
        self.head_len = 0
        self.tail: collections.deque[str] = collections.deque()
        self.tail_len = 0
        self.total = 0

    def write(self, text: str) -> None:
        self.total += len(text)
        if self.head_len < self.head_limit:
            take = text[: self.head_limit - self.head_len]
            self.head.append(take)
            self.head_len += len(take)
            text = text[len(take) :]
        if text:
            self.tail.append(text)
            self.tail_len += len(text)
            while self.tail_len > self.tail_limit and self.tail:
                excess = self.tail_len - self.tail_limit
                first = self.tail[0]
                if len(first) <= excess:
                    self.tail.popleft()
                    self.tail_len -= len(first)
                else:
                    self.tail[0] = first[excess:]
                    self.tail_len -= excess

    @property
    def truncated(self) -> bool:
        return self.total > self.head_len + self.tail_len

    def value(self) -> str:
        head = "".join(self.head)
        tail = "".join(self.tail)
        if self.truncated:
            omitted = self.total - len(head) - len(tail)
            return f"{head}\n... [{omitted} characters of output omitted] ...\n{tail}"
        return head + tail


def default_shell(settings: TerminalSettings) -> str | None:
    if settings.shell:
        return settings.shell
    if IS_WINDOWS:
        return None  # COMSPEC (cmd.exe)
    for candidate in ("/bin/bash", "/usr/bin/bash", "/bin/sh"):
        if os.path.exists(candidate):
            return candidate
    return shutil.which("sh")


def docker_wrap(command: str, workspace: Path, cwd: Path, settings: TerminalSettings) -> str:
    """Build a ``docker run`` invocation that executes ``command`` inside a container."""
    rel = cwd.resolve().relative_to(workspace.resolve()).as_posix() if cwd != workspace else ""
    workdir = "/workspace" + (f"/{rel}" if rel and rel != "." else "")
    parts = [
        "docker", "run", "--rm", "-i",
        f"--network={settings.docker_network}",
        "-v", f"{workspace.resolve()}:/workspace",
        "-w", workdir,
        "-e", "CI=true", "-e", "PYTHONUNBUFFERED=1",
    ]
    if not IS_WINDOWS and hasattr(os, "getuid"):
        parts += ["--user", f"{os.getuid()}:{os.getgid()}"]
    parts += [settings.docker_image, "sh", "-c", command]
    return " ".join(shlex.quote(p) for p in parts)


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        if IS_WINDOWS:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
                return
            except TimeoutError:
                pass
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    with contextlib.suppress(TimeoutError, ProcessLookupError):
        await asyncio.wait_for(proc.wait(), timeout=5)


async def _spawn(command: str, cwd: Path, env: dict[str, str], shell: str | None) -> asyncio.subprocess.Process:
    kwargs: dict = {
        "cwd": str(cwd),
        "env": env,
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
    }
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
        if shell:
            kwargs["executable"] = shell
    return await asyncio.create_subprocess_shell(command, **kwargs)


async def run_shell(
    command: str,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
    max_output_chars: int,
    cancel: CancellationToken | None = None,
    shell: str | None = None,
) -> CommandOutcome:
    """Run ``command`` through the platform shell with a timeout and bounded output."""
    started = time.monotonic()
    buffer = _BoundedBuffer(max_output_chars)
    proc = await _spawn(command, cwd, env, shell)

    async def pump() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            buffer.write(chunk.decode("utf-8", errors="replace"))

    pump_task = asyncio.ensure_future(pump())
    wait_task = asyncio.ensure_future(proc.wait())
    cancel_task = asyncio.ensure_future(cancel.wait()) if cancel is not None else None
    waiters = {wait_task} | ({cancel_task} if cancel_task else set())
    timed_out = cancelled = False
    try:
        done, _ = await asyncio.wait(waiters, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        if wait_task not in done:
            if cancel_task is not None and cancel_task in done:
                cancelled = True
            else:
                timed_out = True
            await _kill_tree(proc)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(pump_task, timeout=5)
    except asyncio.CancelledError:
        await _kill_tree(proc)
        raise
    finally:
        if cancel_task is not None:
            cancel_task.cancel()
        if not pump_task.done():
            pump_task.cancel()
    text = buffer.value()
    if timed_out:
        text += f"\n[command timed out after {timeout_s:.0f}s and was terminated]"
    if cancelled:
        text += "\n[command cancelled]"
    return CommandOutcome(
        command=command,
        exit_code=proc.returncode if not (timed_out or cancelled) else None,
        output=text,
        duration_s=time.monotonic() - started,
        timed_out=timed_out,
        cancelled=cancelled,
        truncated=buffer.truncated,
        total_chars=buffer.total,
    )


@dataclass
class BackgroundProcess:
    id: str
    command: str
    cwd: str
    started: float
    proc: asyncio.subprocess.Process
    buffer: _BoundedBuffer
    pump: asyncio.Task[None] | None = None
    ended: float | None = None
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def running(self) -> bool:
        return self.proc.returncode is None

    def describe(self) -> dict[str, object]:
        return {
            "id": self.id,
            "pid": self.proc.pid,
            "command": self.command,
            "cwd": self.cwd,
            "running": self.running,
            "exit_code": self.proc.returncode,
            "uptime_s": round((self.ended or time.monotonic()) - self.started, 1),
        }


class ProcessManager:
    """Long-running processes started by the agent (dev servers, watchers)."""

    def __init__(self, max_processes: int = 8) -> None:
        self.max_processes = max_processes
        self._procs: dict[str, BackgroundProcess] = {}

    async def start(self, command: str, cwd: Path, env: dict[str, str], shell: str | None, max_output_chars: int = 50000) -> BackgroundProcess:
        running = [p for p in self._procs.values() if p.running]
        if len(running) >= self.max_processes:
            raise RuntimeError(f"too many background processes ({len(running)}); stop one first")
        proc = await _spawn(command, cwd, env, shell)
        bp = BackgroundProcess(new_id("proc"), command, str(cwd), time.monotonic(), proc, _BoundedBuffer(max_output_chars))

        async def pump() -> None:
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                bp.buffer.write(chunk.decode("utf-8", errors="replace"))
            await proc.wait()
            bp.ended = time.monotonic()

        bp.pump = asyncio.ensure_future(pump())
        self._procs[bp.id] = bp
        return bp

    def list(self) -> list[BackgroundProcess]:
        return list(self._procs.values())

    def get(self, proc_id: str) -> BackgroundProcess | None:
        return self._procs.get(proc_id)

    async def stop(self, proc_id: str) -> bool:
        bp = self._procs.get(proc_id)
        if bp is None:
            return False
        await _kill_tree(bp.proc)
        if bp.pump is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(bp.pump, timeout=5)
        bp.ended = bp.ended or time.monotonic()
        return True

    async def stop_all(self) -> None:
        for proc_id in list(self._procs):
            await self.stop(proc_id)
