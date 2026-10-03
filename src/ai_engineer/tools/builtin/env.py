"""Environment inspection (deterministic, used by the agent and `aie doctor`)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..base import Tool, ToolContext, ToolInput, ToolResult

RUNTIMES = {
    "python": ["python3", "python"],
    "node": ["node"],
    "deno": ["deno"],
    "bun": ["bun"],
    "go": ["go"],
    "rustc": ["rustc"],
    "cargo": ["cargo"],
    "java": ["java"],
    "dotnet": ["dotnet"],
    "ruby": ["ruby"],
    "php": ["php"],
}
TOOLS = ["git", "rg", "docker", "make", "npm", "pnpm", "yarn", "pip", "uv", "poetry", "pipenv", "conda", "mvn", "gradle", "gcc", "clang", "cmake"]

_VERSION_ARGS = {"go": ["version"], "java": ["-version"], "dotnet": ["--version"]}


def _version(exe: str, name: str) -> str:
    args = _VERSION_ARGS.get(name, ["--version"])
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    text = (out.stdout or out.stderr).strip().splitlines()
    return text[0][:120] if text else "unknown"


def _memory_bytes() -> int | None:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None
    return None


def inspect_environment(workspace: Path | None = None, with_versions: bool = True) -> dict[str, Any]:
    runtimes: dict[str, str] = {}
    for name, candidates in RUNTIMES.items():
        for candidate in candidates:
            exe = shutil.which(candidate)
            if exe:
                runtimes[name] = _version(exe, name) if with_versions else exe
                break
    tools = {name: shutil.which(name) or "" for name in TOOLS}
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "runtimes": runtimes,
        "tools": {k: bool(v) for k, v in tools.items()},
    }
    if workspace is not None:
        try:
            usage = shutil.disk_usage(workspace)
            info["disk_free_bytes"] = usage.free
        except OSError:
            pass
    return info


def format_environment(info: dict[str, Any]) -> str:
    mem = info.get("memory_bytes")
    lines = [
        f"OS: {info['os']} {info['os_release']} ({info['machine']})",
        f"CPUs: {info['cpu_count']}; memory: {round(mem / 2**30, 1)} GiB" if mem else f"CPUs: {info['cpu_count']}",
        f"Python (agent): {info['python']}",
        "Runtimes: " + (", ".join(f"{k}: {v}" for k, v in info["runtimes"].items()) or "none detected"),
        "Tools available: " + ", ".join(k for k, v in info["tools"].items() if v),
        "Tools missing: " + ", ".join(k for k, v in info["tools"].items() if not v),
    ]
    if "disk_free_bytes" in info:
        lines.append(f"Disk free: {round(info['disk_free_bytes'] / 2**30, 1)} GiB")
    return "\n".join(lines)


class EnvironmentInfoInput(ToolInput):
    pass


class EnvironmentInfoTool(Tool):
    name = "environment_info"
    description = "Report OS, CPU/memory, installed runtimes and developer tools (with versions)."
    Input = EnvironmentInfoInput
    timeout_s = 60.0

    async def run(self, args: EnvironmentInfoInput, ctx: ToolContext) -> ToolResult:
        import asyncio

        info = await asyncio.to_thread(inspect_environment, ctx.workspace)
        return ToolResult(content=format_environment(info), data=info)


ENV_TOOLS: list[type[Tool]] = [EnvironmentInfoTool]
