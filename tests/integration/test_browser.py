"""Browser tool against a real local page (skipped when Playwright/Chromium are unavailable)."""

from __future__ import annotations

import asyncio
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ai_engineer.config.settings import Settings
from ai_engineer.core.types import ToolUseBlock
from ai_engineer.tools.approval import AllowAllBroker
from ai_engineer.tools.builtin.browser import BrowserTool
from ai_engineer.tools.executor import ToolExecutor, ToolRegistry
from ai_engineer.tools.factory import make_context
from ai_engineer.tools.permissions import PermissionPolicy

pytestmark = pytest.mark.browser

PAGE = """<!doctype html><html><head><title>Signup</title></head><body>
<form onsubmit="event.preventDefault(); document.getElementById('out').textContent = 'Welcome, ' + document.getElementById('name').value;">
<input id="name"><button id="go" type="submit">Go</button></form><p id="out"></p></body></html>"""


def _executable() -> str | None:
    exe = os.environ.get("AIE_BROWSER_EXECUTABLE") or "/opt/pw-browsers/chromium"
    return exe if Path(exe).exists() else None


def _browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    for exe in (None, _executable()):
        try:
            with sync_playwright() as p:
                p.chromium.launch(headless=True, **({"executable_path": exe} if exe else {})).close()
            return True
        except Exception:
            continue
    return False


@pytest.mark.skipif(not _browser_available(), reason="Playwright Chromium not available")
async def test_browser_tool_drives_a_local_page(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(PAGE)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = Settings()
    settings.web.browser_executable = _executable()
    ctx = make_context(ws, settings)
    ex = ToolExecutor(ToolRegistry([BrowserTool()]), PermissionPolicy(settings.permissions), AllowAllBroker())

    async def call(**args):
        return await ex.execute(ToolUseBlock(id="b", name="browser", input=args), ctx)

    try:
        r = await call(action="goto", url=url)
        assert not r.is_error and "Signup" in r.content, r.content
        assert not (await call(action="fill", selector="#name", value="Ada")).is_error
        assert not (await call(action="click", selector="#go")).is_error
        r = await call(action="text", selector="#out")
        assert r.content.strip() == "Welcome, Ada"
        r = await call(action="screenshot")
        assert not r.is_error and list((ws / ".agent" / "artifacts").glob("screenshot-*.png"))
        r = await call(action="click", selector="#missing", timeout_ms=500)
        assert r.is_error and "browser click failed" in r.content
    finally:
        await call(action="close")
        server.shutdown()
        await asyncio.sleep(0)
