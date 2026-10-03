"""Benchmark cases: fixture repository + task + hidden verification (+ a scripted oracle).

Hidden verification lives outside the repository and is run only after the agent
finishes, so a run is counted as *verified* only when independent checks pass.

The scripted oracle is a fixed transcript of tool calls used by the `harness`
suite. It measures the agent infrastructure (tools, validation, repair, gates,
git, reporting) end to end. It does not measure model intelligence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _j(data: dict[str, Any]) -> str:
    return json.dumps(data)


def read(path: str) -> dict[str, Any]:
    return {"tool_calls": [{"name": "read_file", "input": {"path": path}}]}


def edit(path: str, old: str, new: str) -> dict[str, Any]:
    return {"tool_calls": [{"name": "edit_file", "input": {"path": path, "old_string": old, "new_string": new}}]}


def write(path: str, content: str) -> dict[str, Any]:
    return {"tool_calls": [{"name": "write_file", "input": {"path": path, "content": content}}]}


def run_tests(*files: str) -> dict[str, Any]:
    return {"tool_calls": [{"name": "run_tests", "input": {"files": list(files)}}]}


def search(pattern: str) -> dict[str, Any]:
    return {"tool_calls": [{"name": "search_text", "input": {"pattern": pattern}}]}


def submit(summary: str, verification: str = "ran the relevant tests") -> dict[str, Any]:
    return {"tool_calls": [{"name": "submit_work", "input": {"summary": summary, "verification": verification}}]}


def understanding(summary: str, criteria: list[str], task_type: str = "change", complexity: str = "small", docs: bool = False, security: bool = False) -> str:
    return _j({
        "summary": summary, "task_type": task_type, "complexity": complexity,
        "requirements": [summary], "acceptance_criteria": criteria,
        "needs": {"tests": task_type == "change", "docs_update": docs, "security_review": security},
    })


def approve(criteria: list[str]) -> str:
    return _j({"verdict": "approve", "summary": "Correct, minimal and tested.", "issues": [],
               "requirements": [{"criterion": c, "status": "met", "evidence": "validated by tests"} for c in criteria]})


PYPROJECT = '[project]\nname = "{name}"\nversion = "0.1.0"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
GITIGNORE = "__pycache__/\n.pytest_cache/\n"


@dataclass
class BenchCase:
    id: str
    category: str
    title: str
    prompt: str
    files: dict[str, str]
    oracle: Callable[[], dict[str, list[Any]]]
    hidden_tests: dict[str, str] = field(default_factory=dict)
    verify: Callable[[Path, dict[str, Any]], tuple[bool, str]] | None = None
    expect_question: bool = False


# ---- 1. repository exploration ----------------------------------------------------------

def _explore_verify(repo: Path, state: dict[str, Any]) -> tuple[bool, str]:
    answer = json.dumps(state.get("answer") or {})
    ok = "billing/invoice.py" in answer and "tests/test_invoice.py" in answer
    return ok, "answer cites implementation and test file" if ok else f"answer missing citations: {answer[:300]}"


EXPLORE = BenchCase(
    id="explore-01", category="repository exploration", title="Locate tax computation and its tests",
    prompt="Which function computes the tax on an invoice, where is it implemented, and which tests cover it?",
    files={
        "pyproject.toml": PYPROJECT.format(name="shop"),
        ".gitignore": GITIGNORE,
        "billing/__init__.py": "",
        "billing/invoice.py": "TAX_RATE = 0.2\n\n\ndef compute_tax(subtotal: float) -> float:\n    return round(subtotal * TAX_RATE, 2)\n\n\ndef total(subtotal: float) -> float:\n    return subtotal + compute_tax(subtotal)\n",
        "billing/customers.py": "def display_name(first: str, last: str) -> str:\n    return f\"{first} {last}\".strip()\n",
        "tests/test_invoice.py": "from billing.invoice import compute_tax, total\n\n\ndef test_tax():\n    assert compute_tax(10) == 2.0\n\n\ndef test_total():\n    assert total(10) == 12.0\n",
    },
    oracle=lambda: {
        "classifier": [understanding("Find the tax computation and its tests", [], task_type="question")],
        "coder": [
            {"tool_calls": [{"name": "find_symbol", "input": {"name": "compute_tax"}}]},
            {"tool_calls": [{"name": "related_tests", "input": {"path": "billing/invoice.py"}}]},
            {"tool_calls": [{"name": "submit_answer", "input": {
                "answer": "`compute_tax` in billing/invoice.py computes tax as subtotal * TAX_RATE (0.2), rounded to cents. It is covered by tests/test_invoice.py (test_tax, and indirectly test_total).",
                "evidence": ["billing/invoice.py:4", "tests/test_invoice.py:4"], "confidence": "high"}}]},
        ],
    },
    verify=_explore_verify,
    expect_question=True,
)

# ---- 2. bug fixing ---------------------------------------------------------------------------

BUGFIX = BenchCase(
    id="bugfix-01", category="bug fixing", title="Fix off-by-one in pagination",
    prompt="Pagination returns the wrong items: page 2 of size 3 over [1..10] should be [4, 5, 6]. Fix paginate().",
    files={
        "pyproject.toml": PYPROJECT.format(name="pager"),
        ".gitignore": GITIGNORE,
        "pager/__init__.py": "def paginate(items, page, size):\n    \"\"\"Return the given 1-based page.\"\"\"\n    start = page * size\n    return items[start:start + size]\n",
        "tests/test_pager.py": "from pager import paginate\n\n\ndef test_first_page():\n    assert paginate(list(range(1, 11)), 1, 3) == [1, 2, 3]\n",
    },
    oracle=lambda: {
        "classifier": [understanding("Fix paginate() to use 1-based pages", ["paginate(range 1..10, 2, 3) == [4, 5, 6]", "existing tests pass"])],
        "coder": [
            read("pager/__init__.py"),
            # deliberately wrong first attempt (exercises the repair loop)
            edit("pager/__init__.py", "start = page * size", "start = (page + 1) * size"),
            submit("adjusted the start offset"),
        ],
        "debugger": [
            run_tests("tests/test_pager.py"),
            read("pager/__init__.py"),
            edit("pager/__init__.py", "start = (page + 1) * size", "start = (page - 1) * size"),
            run_tests("tests/test_pager.py"),
            submit("root cause: pages are 1-based, so the offset is (page - 1) * size"),
        ],
        "reviewer": [approve(["paginate(range 1..10, 2, 3) == [4, 5, 6]", "existing tests pass"])],
    },
    hidden_tests={"test_hidden_pager.py": "from pager import paginate\n\n\ndef test_page_two():\n    assert paginate(list(range(1, 11)), 2, 3) == [4, 5, 6]\n\n\ndef test_last_partial_page():\n    assert paginate(list(range(1, 11)), 4, 3) == [10]\n\n\ndef test_past_end():\n    assert paginate(list(range(1, 11)), 5, 3) == []\n"},
)

# ---- 3. feature implementation ------------------------------------------------------------------

FEATURE_CODE = '''def add(a, b):
    return a + b


def divide(a, b):
    """Divide a by b. Raises ValueError with a clear message when b is zero."""
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b
'''

FEATURE = BenchCase(
    id="feature-01", category="feature implementation", title="Add divide() with zero handling",
    prompt="Add a divide(a, b) function to calc that returns a / b and raises ValueError('cannot divide by zero') when b is 0. Add tests.",
    files={
        "pyproject.toml": PYPROJECT.format(name="calc"),
        ".gitignore": GITIGNORE,
        "calc/__init__.py": "def add(a, b):\n    return a + b\n",
        "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    },
    oracle=lambda: {
        "classifier": [understanding("Add divide() with ValueError on zero", ["divide(6, 3) == 2", "divide(1, 0) raises ValueError"])],
        "coder": [
            read("calc/__init__.py"),
            write("calc/__init__.py", FEATURE_CODE),
            read("tests/test_calc.py"),
            edit("tests/test_calc.py", "from calc import add\n", "import pytest\n\nfrom calc import add, divide\n"),
            edit("tests/test_calc.py", "    assert add(1, 2) == 3\n", "    assert add(1, 2) == 3\n\n\ndef test_divide():\n    assert divide(6, 3) == 2\n\n\ndef test_divide_by_zero():\n    with pytest.raises(ValueError, match=\"cannot divide by zero\"):\n        divide(1, 0)\n"),
            run_tests("tests/test_calc.py"),
            submit("added divide() with zero check and tests"),
        ],
        "reviewer": [approve(["divide(6, 3) == 2", "divide(1, 0) raises ValueError"])],
    },
    hidden_tests={"test_hidden_divide.py": "import pytest\nfrom calc import divide\n\n\ndef test_divide():\n    assert divide(9, 3) == 3\n    assert divide(1, 4) == 0.25\n\n\ndef test_zero():\n    with pytest.raises(ValueError, match='zero'):\n        divide(5, 0)\n"},
)

# ---- 4. refactoring --------------------------------------------------------------------------------

REFACTOR = BenchCase(
    id="refactor-01", category="refactoring", title="Rename get_usr to get_user across modules",
    prompt="Rename the function get_usr to get_user everywhere (definition and all call sites). Behaviour must not change.",
    files={
        "pyproject.toml": PYPROJECT.format(name="app"),
        ".gitignore": GITIGNORE,
        "app/__init__.py": "",
        "app/users.py": "USERS = {1: 'ada', 2: 'grace'}\n\n\ndef get_usr(user_id):\n    return USERS.get(user_id)\n",
        "app/service.py": "from app.users import get_usr\n\n\ndef greeting(user_id):\n    name = get_usr(user_id)\n    return f\"hello {name}\" if name else \"unknown user\"\n",
        "tests/test_service.py": "from app.service import greeting\n\n\ndef test_greeting():\n    assert greeting(1) == 'hello ada'\n    assert greeting(9) == 'unknown user'\n",
    },
    oracle=lambda: {
        "classifier": [understanding("Rename get_usr to get_user", ["no reference to get_usr remains", "behaviour unchanged"])],
        "coder": [
            search("get_usr"),
            read("app/users.py"),
            edit("app/users.py", "def get_usr(", "def get_user("),
            read("app/service.py"),
            edit("app/service.py", "get_usr", "get_user") | {"tool_calls": [{"name": "edit_file", "input": {"path": "app/service.py", "old_string": "get_usr", "new_string": "get_user", "replace_all": True}}]},
            run_tests(),
            submit("renamed get_usr to get_user in definition and call sites"),
        ],
        "reviewer": [approve(["no reference to get_usr remains", "behaviour unchanged"])],
    },
    hidden_tests={"test_hidden_rename.py": "import pathlib\nimport app.users as users\nfrom app.service import greeting\n\n\ndef test_new_name():\n    assert users.get_user(2) == 'grace'\n    assert not hasattr(users, 'get_usr')\n\n\ndef test_no_old_references():\n    for p in pathlib.Path('app').rglob('*.py'):\n        assert 'get_usr' not in p.read_text(), p\n\n\ndef test_behaviour():\n    assert greeting(2) == 'hello grace'\n"},
)

# ---- 5. test generation -------------------------------------------------------------------------------

SLUG = "import re\n\n\ndef slugify(text: str) -> str:\n    text = text.strip().lower()\n    text = re.sub(r\"[^a-z0-9]+\", \"-\", text)\n    return text.strip(\"-\")\n"

TESTGEN_TESTS = '''from textutil import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_and_spaces():
    assert slugify("  Ready, set... GO!  ") == "ready-set-go"


def test_empty():
    assert slugify("") == ""
'''


def _testgen_verify(repo: Path, state: dict[str, Any]) -> tuple[bool, str]:
    """The generated tests must pass on the real code and catch a seeded bug (mutation check)."""
    import subprocess
    import sys

    tests = list((repo / "tests").glob("test_*.py"))
    if not tests:
        return False, "no tests were written"
    env_cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
    ok = subprocess.run(env_cmd, cwd=repo, capture_output=True, text=True, timeout=120, check=False).returncode == 0
    if not ok:
        return False, "generated tests fail on the original code"
    original = (repo / "textutil.py").read_text()
    try:
        (repo / "textutil.py").write_text(original.replace('.strip("-")', ""))
        caught = subprocess.run(env_cmd, cwd=repo, capture_output=True, text=True, timeout=120, check=False).returncode != 0
    finally:
        (repo / "textutil.py").write_text(original)
    return caught, "tests pass and detect the seeded mutant" if caught else "tests did not detect a seeded bug (weak tests)"


TESTGEN = BenchCase(
    id="testgen-01", category="test generation", title="Write tests for slugify",
    prompt="Write thorough pytest tests for textutil.slugify in tests/test_textutil.py.",
    files={"pyproject.toml": PYPROJECT.format(name="textutil"), ".gitignore": GITIGNORE, "textutil.py": SLUG, "tests/.gitkeep": ""},
    oracle=lambda: {
        "classifier": [understanding("Write tests for slugify", ["tests cover normal, punctuation and empty input", "tests pass"])],
        "coder": [read("textutil.py"), write("tests/test_textutil.py", TESTGEN_TESTS), run_tests("tests/test_textutil.py"), submit("added tests for slugify")],
        "reviewer": [approve(["tests cover normal, punctuation and empty input", "tests pass"])],
    },
    verify=_testgen_verify,
)

# ---- 6. debugging ---------------------------------------------------------------------------------------

DEBUG = BenchCase(
    id="debug-01", category="debugging", title="Diagnose a failing statistics test",
    prompt="tests/test_stats.py::test_median_even fails. Find the root cause and fix it.",
    files={
        "pyproject.toml": PYPROJECT.format(name="stats"),
        ".gitignore": GITIGNORE,
        "stats.py": "def median(values):\n    data = sorted(values)\n    n = len(data)\n    mid = n // 2\n    if n % 2:\n        return data[mid]\n    return (data[mid] + data[mid + 1]) / 2\n",
        "tests/test_stats.py": "from stats import median\n\n\ndef test_median_odd():\n    assert median([3, 1, 2]) == 2\n\n\ndef test_median_even():\n    assert median([4, 1, 3, 2]) == 2.5\n",
    },
    oracle=lambda: {
        "classifier": [understanding("Fix median() for even-length input", ["test_median_even passes", "test_median_odd still passes"])],
        "coder": [
            run_tests("tests/test_stats.py"),
            read("stats.py"),
            edit("stats.py", "(data[mid] + data[mid + 1]) / 2", "(data[mid - 1] + data[mid]) / 2"),
            run_tests("tests/test_stats.py"),
            submit("root cause: even-length median used indices mid and mid+1 instead of mid-1 and mid"),
        ],
        "reviewer": [approve(["test_median_even passes", "test_median_odd still passes"])],
    },
    hidden_tests={"test_hidden_stats.py": "from stats import median\n\n\ndef test_even():\n    assert median([1, 2, 3, 4, 5, 6]) == 3.5\n    assert median([10, 20]) == 15\n\n\ndef test_odd():\n    assert median([5]) == 5\n"},
)

# ---- 7. documentation --------------------------------------------------------------------------------------

CLI_CODE = "import argparse\n\n\ndef build_parser():\n    p = argparse.ArgumentParser(prog='greet')\n    p.add_argument('--name', default='world', help='who to greet')\n    p.add_argument('--shout', action='store_true', help='uppercase the greeting')\n    return p\n\n\ndef main(argv=None):\n    args = build_parser().parse_args(argv)\n    text = f'hello {args.name}'\n    print(text.upper() if args.shout else text)\n"

DOCS_README = "# greet\n\nA tiny greeting CLI.\n\n## Usage\n\n```\npython -m greet [--name NAME] [--shout]\n```\n\n- `--name NAME`: who to greet (default: `world`)\n- `--shout`: print the greeting in upper case\n"


def _docs_verify(repo: Path, state: dict[str, Any]) -> tuple[bool, str]:
    readme = (repo / "README.md").read_text() if (repo / "README.md").exists() else ""
    missing = [flag for flag in ("--name", "--shout") if flag not in readme]
    invented = "--loud" in readme or "--verbose" in readme
    ok = not missing and not invented and "## Usage" in readme
    return ok, "README documents every real flag" if ok else f"missing={missing} invented={invented}"


DOCS = BenchCase(
    id="docs-01", category="documentation", title="Document the CLI in the README",
    prompt="Add a Usage section to README.md documenting every command-line option of greet.py accurately.",
    files={"pyproject.toml": PYPROJECT.format(name="greet"), ".gitignore": GITIGNORE, "greet.py": CLI_CODE, "README.md": "# greet\n\nA tiny greeting CLI.\n",
           "tests/test_greet.py": "from greet import main\n\n\ndef test_default(capsys):\n    main([])\n    assert capsys.readouterr().out.strip() == 'hello world'\n"},
    oracle=lambda: {
        "classifier": [understanding("Document the CLI options in README", ["README has a Usage section", "every real option is documented"], docs=True)],
        "coder": [read("greet.py"), read("README.md"), write("README.md", DOCS_README), submit("documented --name and --shout")],
        "reviewer": [approve(["README has a Usage section", "every real option is documented"])],
    },
    verify=_docs_verify,
)

# ---- 8. dependency / API upgrade ------------------------------------------------------------------------------

UPGRADE = BenchCase(
    id="upgrade-01", category="dependency upgrade", title="Replace deprecated datetime.utcnow()",
    prompt="Replace the deprecated datetime.utcnow() usage with timezone-aware datetime.now(timezone.utc) and keep behaviour (timestamps in UTC).",
    files={
        "pyproject.toml": PYPROJECT.format(name="clock"),
        ".gitignore": GITIGNORE,
        "clock.py": "from datetime import datetime\n\n\ndef now_iso():\n    return datetime.utcnow().isoformat()\n",
        "tests/test_clock.py": "from clock import now_iso\n\n\ndef test_iso():\n    assert 'T' in now_iso()\n",
    },
    oracle=lambda: {
        "classifier": [understanding("Use timezone-aware UTC timestamps", ["no datetime.utcnow() remains", "timestamps are UTC and timezone-aware"])],
        "coder": [
            read("clock.py"),
            write("clock.py", "from datetime import datetime, timezone\n\n\ndef now_iso():\n    return datetime.now(timezone.utc).isoformat()\n"),
            run_tests(),
            submit("replaced utcnow() with datetime.now(timezone.utc)"),
        ],
        "reviewer": [approve(["no datetime.utcnow() remains", "timestamps are UTC and timezone-aware"])],
    },
    hidden_tests={"test_hidden_clock.py": "import warnings\nfrom datetime import datetime\nfrom clock import now_iso\n\n\ndef test_aware_utc():\n    with warnings.catch_warnings():\n        warnings.simplefilter('error', DeprecationWarning)\n        value = now_iso()\n    parsed = datetime.fromisoformat(value)\n    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0\n\n\ndef test_no_utcnow():\n    assert 'utcnow' not in open('clock.py').read()\n"},
)

# ---- 9. security remediation -------------------------------------------------------------------------------------

SQL_FIXED = "import sqlite3\n\n\ndef find_user(con: sqlite3.Connection, name: str):\n    return con.execute(\"SELECT id, name FROM users WHERE name = ?\", (name,)).fetchall()\n"

SECURITY = BenchCase(
    id="security-01", category="security remediation", title="Fix SQL injection",
    prompt="find_user() is vulnerable to SQL injection. Fix it using parameterized queries without changing results for normal input.",
    files={
        "pyproject.toml": PYPROJECT.format(name="db"),
        ".gitignore": GITIGNORE,
        "db.py": "import sqlite3\n\n\ndef find_user(con: sqlite3.Connection, name: str):\n    return con.execute(f\"SELECT id, name FROM users WHERE name = '{name}'\").fetchall()\n",
        "tests/test_db.py": "import sqlite3\nfrom db import find_user\n\n\ndef make():\n    con = sqlite3.connect(':memory:')\n    con.execute('CREATE TABLE users (id INTEGER, name TEXT)')\n    con.executemany('INSERT INTO users VALUES (?, ?)', [(1, 'ada'), (2, 'bob')])\n    return con\n\n\ndef test_find():\n    assert find_user(make(), 'ada') == [(1, 'ada')]\n",
    },
    oracle=lambda: {
        "classifier": [understanding("Parameterize the SQL query in find_user", ["injection payloads return no rows", "normal lookups unchanged"], security=True)],
        "coder": [read("db.py"), write("db.py", SQL_FIXED), run_tests(), submit("use a bound parameter instead of string formatting")],
        "reviewer": [approve(["injection payloads return no rows", "normal lookups unchanged"])],
    },
    hidden_tests={"test_hidden_injection.py": "import sqlite3\nfrom db import find_user\n\n\ndef make():\n    con = sqlite3.connect(':memory:')\n    con.execute('CREATE TABLE users (id INTEGER, name TEXT)')\n    con.executemany('INSERT INTO users VALUES (?, ?)', [(1, 'ada'), (2, 'bob')])\n    return con\n\n\ndef test_injection_blocked():\n    assert find_user(make(), \"x' OR '1'='1\") == []\n\n\ndef test_normal():\n    assert find_user(make(), 'bob') == [(2, 'bob')]\n"},
)

# ---- 10. multi-file architecture change -----------------------------------------------------------------------------

STORAGE = '''from typing import Protocol


class Storage(Protocol):
    def save(self, key: str, value: str) -> None: ...

    def load(self, key: str) -> str | None: ...


class MemoryStorage:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def save(self, key: str, value: str) -> None:
        self._data[key] = value

    def load(self, key: str) -> str | None:
        return self._data.get(key)
'''

NOTES_SERVICE = '''from notes.storage import MemoryStorage, Storage


class NotesService:
    def __init__(self, storage: Storage | None = None) -> None:
        self.storage: Storage = storage or MemoryStorage()

    def add(self, title: str, body: str) -> None:
        self.storage.save(title, body)

    def get(self, title: str) -> str | None:
        return self.storage.load(title)
'''

ARCH = BenchCase(
    id="arch-01", category="multi-file architecture change", title="Introduce a pluggable storage interface",
    prompt="NotesService stores notes in a private dict. Introduce a Storage protocol (save/load) in notes/storage.py with a MemoryStorage implementation, and make NotesService accept any Storage via its constructor (defaulting to MemoryStorage).",
    files={
        "pyproject.toml": PYPROJECT.format(name="notes"),
        ".gitignore": GITIGNORE,
        "notes/__init__.py": "",
        "notes/service.py": "class NotesService:\n    def __init__(self):\n        self._notes = {}\n\n    def add(self, title, body):\n        self._notes[title] = body\n\n    def get(self, title):\n        return self._notes.get(title)\n",
        "tests/test_notes.py": "from notes.service import NotesService\n\n\ndef test_roundtrip():\n    s = NotesService()\n    s.add('a', 'b')\n    assert s.get('a') == 'b'\n",
    },
    oracle=lambda: {
        "classifier": [understanding("Introduce Storage protocol and inject it into NotesService", ["NotesService accepts a Storage", "MemoryStorage is the default", "existing behaviour preserved"], complexity="medium")],
        "planner": [_j({"goal": "pluggable storage", "approach": "Protocol + default in-memory implementation, constructor injection",
                        "subtasks": [{"id": "s1", "title": "Add storage module", "description": "Create notes/storage.py with Storage protocol and MemoryStorage", "acceptance_criteria": ["MemoryStorage save/load works"]},
                                     {"id": "s2", "title": "Inject storage into NotesService", "description": "Constructor injection with MemoryStorage default", "depends_on": ["s1"], "acceptance_criteria": ["NotesService accepts a Storage", "existing behaviour preserved"]}],
                        "decisions": ["Use typing.Protocol for structural typing so callers need no inheritance"]})],
        "coder": [
            write("notes/storage.py", STORAGE),
            submit("added Storage protocol and MemoryStorage"),
            read("notes/service.py"),
            write("notes/service.py", NOTES_SERVICE),
            run_tests(),
            submit("NotesService now takes a Storage, defaulting to MemoryStorage"),
        ],
        "reviewer": [approve(["MemoryStorage save/load works"]), approve(["NotesService accepts a Storage", "existing behaviour preserved"]),
                     approve(["NotesService accepts a Storage", "MemoryStorage is the default", "existing behaviour preserved"])],
    },
    hidden_tests={"test_hidden_storage.py": "from notes.service import NotesService\nfrom notes.storage import MemoryStorage\n\n\nclass Spy:\n    def __init__(self):\n        self.saved = {}\n\n    def save(self, key, value):\n        self.saved[key] = value\n\n    def load(self, key):\n        return self.saved.get(key)\n\n\ndef test_injected():\n    spy = Spy()\n    s = NotesService(spy)\n    s.add('k', 'v')\n    assert spy.saved == {'k': 'v'} and s.get('k') == 'v'\n\n\ndef test_default():\n    assert isinstance(NotesService().storage, MemoryStorage)\n"},
)


ALL_CASES: list[BenchCase] = [EXPLORE, BUGFIX, FEATURE, REFACTOR, TESTGEN, DEBUG, DOCS, UPGRADE, SECURITY, ARCH]
