"""`aie demo`: a deterministic, offline end-to-end run of the whole pipeline.

A small project with a real bug and a missing feature is created in a fresh directory, and the
agent works on it with a *scripted* model (a fixed transcript, no network, no API key). Everything
else is real: repository discovery, planning, file edits through the guarded tools, real test runs
(stdlib ``unittest``), a deliberately wrong first fix, diagnosis, repair, re-test, independent
review, checkpoints and commits on a task branch, quality gates and the engineering report.

The scripted transcript shows what the harness does; it says nothing about the quality of any model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

STOCK = '''"""Stock keeping for a small shop."""


class Inventory:
    def __init__(self) -> None:
        self._items: dict[str, int] = {}

    def add(self, sku: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("quantity must be positive")
        self._items[sku] = self._items.get(sku, 0) + qty

    def remove(self, sku: str, qty: int) -> None:
        self._items[sku] = self._items.get(sku, 0) - qty

    def count(self, sku: str) -> int:
        return self._items.get(sku, 0)
'''

TESTS = '''import unittest

from inventory.stock import Inventory


class InventoryTest(unittest.TestCase):
    def test_add_and_count(self):
        inv = Inventory()
        inv.add("apple", 3)
        inv.add("apple", 2)
        self.assertEqual(inv.count("apple"), 5)

    def test_remove_all_stock(self):
        inv = Inventory()
        inv.add("pear", 4)
        inv.remove("pear", 4)
        self.assertEqual(inv.count("pear"), 0)

    def test_cannot_remove_more_than_available(self):
        inv = Inventory()
        inv.add("plum", 1)
        with self.assertRaises(ValueError):
            inv.remove("plum", 2)
        self.assertEqual(inv.count("plum"), 1)


if __name__ == "__main__":
    unittest.main()
'''

TASK = (
    "Stock can currently go negative: make Inventory.remove() raise ValueError when more units are removed "
    "than are in stock (leaving the stock unchanged), and add Inventory.total_units() returning the number of "
    "units across all SKUs, with a test."
)

CRITERIA = [
    "remove() raises ValueError and leaves stock unchanged when removing more than available",
    "removing exactly the available stock works",
    "total_units() returns the sum of all stock and is tested",
]


def _git(root: Path, *args: str) -> None:
    env_args = ["-c", "user.name=AI Engineer demo", "-c", "user.email=demo@localhost", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", *env_args, *args], cwd=root, check=True, capture_output=True)


def create_demo_repo(root: Path) -> Path:
    """Write the demo project and commit it (when git is available)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "inventory").mkdir(exist_ok=True)
    (root / "inventory" / "__init__.py").write_text('"""Demo shop inventory."""\n', encoding="utf-8")
    (root / "inventory" / "stock.py").write_text(STOCK, encoding="utf-8")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "test_stock.py").write_text(TESTS, encoding="utf-8")
    (root / "README.md").write_text("# Demo shop\n\nRun the tests with `python -m unittest discover -s tests -t .`\n", encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    try:
        _git(root, "init", "-q", "-b", "main")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "Demo shop")
    except (OSError, subprocess.CalledProcessError):
        pass  # without git the demo still runs, with file-backup checkpoints
    return root


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": name, "input": args}]}


def _approve(summary: str, met: list[str]) -> str:
    return json.dumps({
        "verdict": "approve", "summary": summary, "issues": [],
        "requirements": [{"criterion": c, "status": "met", "evidence": "inventory/stock.py; tests/test_stock.py pass"} for c in met],
    })


REMOVE_OLD = "        self._items[sku] = self._items.get(sku, 0) - qty\n"
REMOVE_WRONG = (
    "        if qty >= self.count(sku):\n"
    '            raise ValueError(f"only {self.count(sku)} unit(s) of {sku} in stock")\n'
    "        self._items[sku] = self._items.get(sku, 0) - qty\n"
)
REMOVE_RIGHT = REMOVE_WRONG.replace("qty >= self.count(sku)", "qty > self.count(sku)")
COUNT_OLD = "    def count(self, sku: str) -> int:\n        return self._items.get(sku, 0)\n"
COUNT_NEW = COUNT_OLD + "\n    def total_units(self) -> int:\n        return sum(self._items.values())\n"
TEST_ANCHOR = '\n\nif __name__ == "__main__":'
TEST_NEW = (
    "\n\n    def test_total_units(self):\n        inv = Inventory()\n        inv.add(\"apple\", 3)\n"
    "        inv.add(\"pear\", 4)\n        inv.remove(\"pear\", 1)\n        self.assertEqual(inv.total_units(), 6)"
    + TEST_ANCHOR
)


def demo_script() -> dict[str, list[Any]]:
    """The fixed model transcript, per role."""
    understanding = json.dumps({
        "summary": "Prevent negative stock in Inventory.remove() and add Inventory.total_units() with a test",
        "task_type": "change", "complexity": "medium",
        "requirements": ["remove() rejects removing more than available", "total_units() sums all stock"],
        "acceptance_criteria": CRITERIA, "needs": {"tests": True},
        "relevant_paths": ["inventory/stock.py", "tests/test_stock.py"],
    })
    plan = json.dumps({
        "goal": "Correct stock removal and report total units",
        "approach": "Guard remove() before mutating state; add total_units() as a read-only sum; cover both with unittest.",
        "subtasks": [
            {"id": "s1", "title": "Reject removing more stock than available", "description": "Make remove() raise ValueError without changing stock when qty exceeds the count",
             "acceptance_criteria": CRITERIA[:2], "files_hint": ["inventory/stock.py"]},
            {"id": "s2", "title": "Add total_units() with a test", "description": "Return the sum of all stock; add test_total_units",
             "acceptance_criteria": CRITERIA[2:], "depends_on": ["s1"], "files_hint": ["inventory/stock.py", "tests/test_stock.py"]},
        ],
        "decisions": ["Validate before mutating so a rejected removal leaves the inventory unchanged"],
    })
    return {
        "classifier": [understanding],
        "planner": [plan],
        "coder": [
            # s1 — the first attempt contains an off-by-one comparison (>= instead of >)
            _call("read_file", path="inventory/stock.py"),
            _call("edit_file", path="inventory/stock.py", old_string=REMOVE_OLD, new_string=REMOVE_WRONG),
            _call("submit_work", summary="remove() now refuses to go below zero", files_changed=["inventory/stock.py"]),
            # s2
            _call("read_file", path="inventory/stock.py"),
            _call("edit_file", path="inventory/stock.py", old_string=COUNT_OLD, new_string=COUNT_NEW),
            _call("read_file", path="tests/test_stock.py"),
            _call("edit_file", path="tests/test_stock.py", old_string=TEST_ANCHOR, new_string=TEST_NEW),
            _call("run_tests"),
            _call("submit_work", summary="added total_units() and test_total_units", files_changed=["inventory/stock.py", "tests/test_stock.py"]),
        ],
        "debugger": [
            _call("read_file", path="inventory/stock.py"),
            _call("run_tests", files=["tests/test_stock.py"]),
            _call("edit_file", path="inventory/stock.py", old_string="qty >= self.count(sku)", new_string="qty > self.count(sku)"),
            _call("run_tests", files=["tests/test_stock.py"]),
            _call("submit_work", summary="root cause: the guard used >=, so removing exactly the available stock was rejected; it now uses >",
                  verification="python -m unittest: all tests pass"),
        ],
        "reviewer": [
            _approve("remove() validates before mutating; boundary case covered by test_remove_all_stock", CRITERIA[:2]),
            _approve("total_units() is a simple sum and is tested", CRITERIA[2:]),
            _approve("Both requirements are met and covered by tests", CRITERIA),
        ],
    }


def demo_runtime(root: Path) -> Any:
    from .config.settings import ModelsSettings
    from .models.registry import ProviderRegistry
    from .providers.scripted import ScriptedProvider
    from .runtime import Runtime

    registry = ProviderRegistry(ModelsSettings())
    registry.register_instance("demo", ScriptedProvider("demo", by_role=demo_script()))
    python = Path(sys.executable).as_posix()
    config = {
        "models": {"providers": {"demo": {"type": "scripted"}}, "roles": {"default": ["demo:transcript"]}},
        "validation": {"test_command": f'"{python}" -m unittest discover -s tests -t . -v', "dependency_audit": False},
        "permissions": {"mode": "developer"},
    }
    return Runtime.open(root, config, registry=registry, use_global_config=False)


async def run_demo(root: Path, verbose: bool = False) -> Any:
    """Create the demo project in ``root`` and run the task end to end. Returns the finished task."""
    from .ui.render import TerminalRenderer

    create_demo_repo(root)
    rt = demo_runtime(root)
    rt.bus.subscribe(TerminalRenderer(verbose=verbose))
    try:
        task = rt.create_task(TASK, title="Prevent negative stock and add total_units()")
        return await rt.run_task(task.id)
    finally:
        await rt.aclose()
