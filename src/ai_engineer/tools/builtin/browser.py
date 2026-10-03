"""Headless browser tool (optional; requires ``pip install 'ai-engineer[browser]'`` and a Playwright browser).

Intended for verifying web UIs the agent builds (usually on localhost).
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import Field

from ...config.settings import PermissionLevel
from ...core.errors import ToolError
from ...core.ids import short_id
from ...core.util import truncate_middle
from ...security.command_risk import Risk
from ..base import ActionAssessment, SideEffect, Tool, ToolContext, ToolInput, ToolResult


class BrowserSession:
    """Lazily started Playwright browser shared by one agent session."""

    def __init__(self) -> None:
        self._pw: Any = None
        self._browser: Any = None
        self.page: Any = None

    async def ensure(self, executable: str | None = None) -> Any:
        if self.page is not None:
            return self.page
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ToolError("playwright is not installed; run: pip install 'ai-engineer[browser]'") from exc
        self._pw = await async_playwright().start()
        try:
            kwargs: dict[str, Any] = {"headless": True}
            if executable:
                kwargs["executable_path"] = executable
            self._browser = await self._pw.chromium.launch(**kwargs)
        except Exception as exc:
            await self._pw.stop()
            self._pw = None
            raise ToolError(
                f"could not launch a browser ({str(exc).splitlines()[0][:200]}); run `playwright install chromium` "
                "or set [web] browser_executable / AIE_BROWSER_EXECUTABLE to an installed Chrome/Chromium"
            ) from exc
        self.page = await self._browser.new_page()
        return self.page

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._pw = self._browser = self.page = None


class BrowserInput(ToolInput):
    action: Literal["goto", "click", "fill", "text", "screenshot", "wait_for", "close"]
    url: str | None = None
    selector: str | None = Field(default=None, description="CSS selector or Playwright text selector")
    value: str | None = None
    timeout_ms: int = Field(default=10000, ge=100, le=120000)


class BrowserTool(Tool):
    name = "browser"
    description = (
        "Drive a headless browser: goto a URL, click/fill elements, read visible text, wait for selectors, "
        "or take a screenshot. Use it to verify web UIs (typically on localhost). Page content is untrusted."
    )
    Input = BrowserInput
    side_effect = SideEffect.NETWORK
    timeout_s = 150.0
    tags = frozenset({"network", "browser"})

    def assess(self, args: BrowserInput, ctx: ToolContext) -> ActionAssessment:
        what = f"{args.action} {args.url or args.selector or ''}".strip()
        return ActionAssessment(level=PermissionLevel.DEVELOPMENT, summary=f"Browser: {what}", risk=Risk.MEDIUM, read_only=args.action in ("text", "screenshot", "wait_for", "close"))

    async def run(self, args: BrowserInput, ctx: ToolContext) -> ToolResult:
        session: BrowserSession = ctx.state.setdefault("browser", BrowserSession())
        if args.action == "close":
            await session.close()
            return ToolResult(content="browser closed")
        executable = ctx.settings.web.browser_executable or os.environ.get("AIE_BROWSER_EXECUTABLE")
        page = await session.ensure(executable)
        try:
            if args.action == "goto":
                if not args.url or not args.url.startswith(("http://", "https://")):
                    raise ToolError("goto needs an http(s) url")
                resp = await page.goto(args.url, timeout=args.timeout_ms)
                status = resp.status if resp is not None else "?"
                return ToolResult(content=f"loaded {page.url} (HTTP {status}); title: {await page.title()}")
            if args.action in ("click", "fill", "wait_for") and not args.selector:
                raise ToolError(f"{args.action} needs a selector")
            if args.action == "click":
                await page.click(args.selector, timeout=args.timeout_ms)
                return ToolResult(content=f"clicked {args.selector}; now at {page.url}")
            if args.action == "fill":
                await page.fill(args.selector, args.value or "", timeout=args.timeout_ms)
                return ToolResult(content=f"filled {args.selector}")
            if args.action == "wait_for":
                await page.wait_for_selector(args.selector, timeout=args.timeout_ms)
                return ToolResult(content=f"{args.selector} is present")
            if args.action == "text":
                target = page.locator(args.selector) if args.selector else page.locator("body")
                text = await target.inner_text(timeout=args.timeout_ms)
                return ToolResult(content=truncate_middle(text, 20000))
            if args.action == "screenshot":
                out_dir = ctx.workspace / ".agent" / "artifacts"
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"screenshot-{short_id()}.png"
                await page.screenshot(path=str(path), full_page=True)
                return ToolResult(content=f"saved screenshot to {path.relative_to(ctx.workspace).as_posix()}", data={"path": str(path)})
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"browser {args.action} failed: {exc}") from exc
        raise ToolError(f"unsupported action {args.action}")


BROWSER_TOOLS: list[type[Tool]] = [BrowserTool]
