"""Turn raw test / lint / type-check / build output into a structured :class:`CheckResult`.

Parsers are deliberately tolerant: anything not recognised falls back to exit-code
semantics plus a few output heuristics, so an unknown tool still yields a usable result.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .detect import executable_of, norm_program, program_tokens
from .models import CheckKind, CheckResult, CheckStatus, Diagnostic, TestCaseFailure

OUTPUT_TAIL_CHARS = 6000
MAX_ITEMS = 200

ERROR_CLASSIFICATIONS = frozenset({"command_not_found", "missing_dependency", "environment"})

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_FILE = r"(?P<file>(?:[A-Za-z]:)?[^\s:][^:\n]*?)"


@dataclass
class _Parsed:
    tool: str = ""
    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    skipped: int | None = None
    failures: list[TestCaseFailure] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    classification: str = ""
    no_tests: bool = False
    force_fail: bool = False
    detail: str = ""
    extra: str = ""  # additional summary text, e.g. "2 packages ok"

    def structured(self) -> bool:
        return bool(self.failures or self.diagnostics or self.passed or self.failed)


# --- helpers -----------------------------------------------------------------------------------


def _clean(output: str) -> str:
    return _ANSI.sub("", output).replace("\r\n", "\n").replace("\r", "\n")


def _norm_file(path: str | None) -> str | None:
    if not path:
        return None
    path = path.strip().strip("\"'")
    if path.startswith(("./", ".\\")):
        path = path[2:]
    return path or None


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


_ASSERTISH = re.compile(r"(?i)\bassert|\bexpect(?:ed)?\b|expect\(|\bto(?:Be|Equal)\b|!=|\bwant\b|\bgot\b")


def _failure_type(message: str, default: str = "unknown") -> str:
    if re.search(r"(?i)\btime(?:d)? ?out\b|TimeoutError|Failed: Timeout", message):
        return "timeout"
    if re.search(r"ModuleNotFoundError|ImportError|No module named|cannot import name|Cannot find module|ERR_MODULE_NOT_FOUND|Failed to load url", message):
        return "import"
    if re.search(r"SyntaxError|IndentationError|TabError|syntax error|Parsing error", message):
        return "syntax"
    if re.search(r"AssertionError|AssertError|assertion", message) or re.match(r"\s*assert\b", message):
        return "assertion"
    if re.search(r"\bTypeError\b|error TS\d+", message):
        return "type"
    if _ASSERTISH.search(message) and default in ("unknown", "assertion"):
        return "assertion"
    if re.search(r"\b\w*(?:Error|Exception)\b|\bpanic", message):
        return "error"
    return default


def _add_failure(parsed: _Parsed, failure: TestCaseFailure) -> None:
    if len(parsed.failures) >= MAX_ITEMS:
        return
    if any(f.test_id == failure.test_id and f.file == failure.file for f in parsed.failures):
        return
    parsed.failures.append(failure)


def _counts(text: str, pattern: re.Pattern[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for num, word in pattern.findall(text):
        out[word] = out.get(word, 0) + int(num)
    return out


# --- generic heuristics ---------------------------------------------------------------------------

# (needle, pattern) pairs: a pattern only runs on a bounded window around lines containing its needle
_NOT_FOUND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("not found", re.compile(r"^.*?:\s*(?:line \d+:\s*)?(?:\d+:\s*)?(?P<name>[^\s:]+): (?:command )?not found\s*$")),
    ("command not found: ", re.compile(r"command not found: (?P<name>\S+)")),
    ("is not recognized", re.compile(r"'(?P<name>[^']+)' is not recognized as an internal or external command")),
    ("is not recognized", re.compile(r"The term '(?P<name>[^']+)' is not recognized as (?:the )?name of a cmdlet")),
    ("issing script", re.compile(r"(?:npm|pnpm|yarn)(?: ERR!| error|:)? Missing script: \"?(?P<name>[\w:.-]+)\"?", re.I)),
]
_NO_SUCH_FILE = ": No such file or directory"
_WINDOW = 300


def _windows(text: str, needle: str) -> list[str]:
    """Short slices of the lines containing ``needle`` (keeps regex cost linear)."""
    if needle not in text:
        return []
    out: list[str] = []
    for line in text.split("\n"):
        idx = line.find(needle)
        if idx >= 0:
            out.append(line[max(0, idx - _WINDOW) : idx + len(needle) + _WINDOW].strip())
    return out


_CANT_OPEN = re.compile(r"can't open file .*No such file or directory")
_NO_MODULE = re.compile(r"No module named '?(?P<mod>[\w.]+)'?")
_NODE_MISSING = re.compile(r"Cannot find (?:module|package) '(?P<mod>[^']+)'|ERR_MODULE_NOT_FOUND")
_SYNTAX = re.compile(r"\b(?:SyntaxError|IndentationError|TabError)\b")
_NETWORK = re.compile(
    r"ENOTFOUND|EAI_AGAIN|ECONNREFUSED|ECONNRESET|ETIMEDOUT|Temporary failure in name resolution|"
    r"Could not resolve host|Network is unreachable|Failed to establish a new connection|"
    r"NewConnectionError|ConnectionError|getaddrinfo|network request .* failed",
    re.I,
)


def _command_not_found(text: str, command: str) -> str:
    """The line saying the program could not be found ('' when not applicable)."""
    for needle, pattern in _NOT_FOUND_PATTERNS:
        for window in _windows(text, needle):
            if pattern.search(window):
                return window
    exe = executable_of(command)
    exe_name = norm_program(exe) if exe else ""
    for window in _windows(text, _NO_SUCH_FILE):
        before = window.split(_NO_SUCH_FILE, 1)[0].rstrip("'\u2019\"")
        name = re.split(r"[\s:\u2018'\"]", before)[-1] if before else ""
        if exe_name and name and norm_program(name) == exe_name:
            return window
    for window in _windows(text, "can't open file"):
        if _CANT_OPEN.search(window):
            return window
    return ""


def _looks_local(module: str, text: str) -> bool:
    top = re.escape(module.split(".")[0])
    return bool(re.search(rf"(?:^|[\s/\\'\"(]){top}(?:[/\\]|\.py\b)", text, re.M))


def _import_classification(message: str, text: str, tests_ran: bool) -> str:
    m = _NO_MODULE.search(message) or _NO_MODULE.search(text)
    if m:
        if tests_ran or _looks_local(m.group("mod"), text):
            return "import_error"
        return "missing_dependency"
    n = _NODE_MISSING.search(message) or _NODE_MISSING.search(text)
    if n and n.group("mod"):
        local = n.group("mod").startswith((".", "/"))
        return "import_error" if (tests_ran or local) else "missing_dependency"
    return "import_error"


def _line_of(pattern: re.Pattern[str], text: str) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    start = text.rfind("\n", 0, m.start()) + 1
    end = text.find("\n", m.end())
    return text[start : end if end >= 0 else len(text)].strip()


def _generic_classification(text: str, command: str) -> tuple[str, str]:
    """Heuristics for output nothing else understood: ``(classification, key line)``."""
    line = _command_not_found(text, command)
    if line:
        return "command_not_found", line
    if _NO_MODULE.search(text) or _NODE_MISSING.search(text):
        key = _line_of(_NO_MODULE, text) or _line_of(_NODE_MISSING, text)
        return _import_classification(key, text, tests_ran=False), key
    if _SYNTAX.search(text):
        return "syntax_error", _line_of(_SYNTAX, text)
    if _NETWORK.search(text):
        return "environment", _line_of(_NETWORK, text)
    return "", ""


# --- diagnostics (linters, type checkers, compilers, formatters) -------------------------------------

_ARROW = re.compile(r"^\s*-->\s+(?P<file>.+?):(?P<line>\d+)(?::(?P<col>\d+))?\s*$")
_HEAD_RUST = re.compile(r"^(?P<sev>error|warning)(?:\[(?P<code>[\w-]+)\])?: (?P<msg>.+)$")
_HEAD_RUFF = re.compile(r"^(?P<code>[A-Z]{1,8}\d{1,5}) (?:\[\*\] )?(?P<msg>.+)$")
_HEAD_LABEL = re.compile(r"^(?P<code>[a-z][a-z0-9]*(?:-[a-z0-9]+)*): (?P<msg>.+)$")
_TSC_PAREN = re.compile(r"^(?P<file>[^\s(][^(\n]*?)\((?P<line>\d+),(?P<col>\d+)\): (?P<sev>error|warning) (?P<code>TS\d+): (?P<msg>.*)$")
_DASH = re.compile(rf"^\s*{_FILE}:(?P<line>\d+):(?P<col>\d+) - (?P<sev>error|warning)(?: (?P<code>TS\d+))?: (?P<msg>.*)$")
_MYPY = re.compile(
    rf"^{_FILE}:(?P<line>\d+)(?::(?P<col>\d+))?: (?P<sev>error|warning|note|fatal error): (?P<msg>.*?)(?:\s{{2,}}\[(?P<code>[\w-]+)\])?\s*$"
)
_PYLINT = re.compile(rf"^{_FILE}:(?P<line>\d+):(?P<col>\d+): (?P<code>[CRWEFI]\d{{4}}): (?P<msg>.*?)(?: \((?P<sym>[a-z][\w-]*)\))?\s*$")
_RUFF = re.compile(rf"^{_FILE}:(?P<line>\d+):(?P<col>\d+): (?P<code>[A-Z]{{1,8}}\d{{1,5}}) (?:\[\*\] )?(?P<msg>.*)$")
_ESLINT_UNIX = re.compile(rf"^{_FILE}:(?P<line>\d+):(?P<col>\d+): (?P<msg>.*) \[(?P<sev>Error|Warning)(?:/(?P<code>[^\]]+))?\]\s*$")
_ESLINT_COMPACT = re.compile(
    r"^(?P<file>.+?): line (?P<line>\d+), col (?P<col>\d+), (?P<sev>Error|Warning) - (?P<msg>.*?)(?: \((?P<code>[^()]+)\))?\s*$"
)
_STYLISH = re.compile(r"^\s+(?P<line>\d+):(?P<col>\d+)\s+(?P<sev>error|warning)\s+(?P<msg>.+?)(?:\s{2,}(?P<code>[@\w][\w@/.-]*))?\s*$")
_MAVEN = re.compile(r"^\[ERROR\] (?P<file>(?:[A-Za-z]:)?[^\s:][^:\n]*?\.\w+):\[(?P<line>\d+),(?P<col>\d+)\] (?P<msg>.*)$")
_GENERIC = re.compile(r"^(?P<file>(?:[A-Za-z]:)?[^\s:][^:\n]*?\.[A-Za-z0-9]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<msg>\S.*)$")
_PYRIGHT_RULE = re.compile(r"\s*\((?P<rule>report\w+)\)\s*$")

_FMT_RUFF_OLD = re.compile(r"^Would reformat: (?P<file>.+?)\s*$")
_FMT_BLACK = re.compile(r"^would reformat (?P<file>.+?)\s*$")
_FMT_BLACK_ERR = re.compile(r"^error: cannot format (?P<file>.+?): (?P<msg>.+)$")
_FMT_PRETTIER = re.compile(r"^\[warn\] (?P<file>.+?)\s*$")
_FMT_PRETTIER_ERR = re.compile(r"^\[error\] (?P<file>[^:\s][^:]*?): (?P<msg>.+)$")
_FMT_CARGO = re.compile(r"^Diff in (?P<file>.+?)(?: at line (?P<line>\d+)|:(?P<line2>\d+)):?\s*$")
_FMT_ISORT = re.compile(r"^ERROR: (?P<file>.+?) Imports are incorrectly sorted")
_GOFMT_FILE = re.compile(r"^(?P<file>[^\s:][^:\n]*\.go)\s*$")

_MAX_LINE = 2000  # longer lines are never diagnostics; bounding them keeps matching cheap
_IGNORED_LABELS = {"note", "help", "hint", "info", "warning", "error", "caused by", "debug"}


def _sev(value: str | None) -> str:
    return "warning" if value and value.lower() in ("warning", "note") else "error"


def _diag(file: str | None, line: str | int | None, col: str | int | None, code: str | None, msg: str, sev: str = "error") -> Diagnostic:
    return Diagnostic(
        file=_norm_file(file),
        line=_int(str(line)) if line is not None else None,
        column=_int(str(col)) if col is not None else None,
        code=code or None,
        message=msg.strip() or "(no message)",
        severity="warning" if sev == "warning" else "error",
    )


def _parse_diagnostics(text: str, *, format_mode: bool = False, gofmt: bool = False) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    seen: set[tuple[object, ...]] = set()
    pending: tuple[str | None, str, str] | None = None  # (code, message, severity) awaiting a "-->" line
    stylish_file: str | None = None

    def add(d: Diagnostic) -> None:
        key = (d.file, d.line, d.column, d.code, d.message)
        if key not in seen and len(out) < MAX_ITEMS:
            seen.add(key)
            out.append(d)

    for raw in text.split("\n"):
        line = raw[:_MAX_LINE].rstrip()
        if not line.strip():
            continue
        if pending is not None:
            m = _ARROW.match(line)
            if m:
                code, msg, sev = pending
                add(_diag(m["file"], m["line"], m["col"], code, msg, sev))
                pending = None
                continue
            pending = None
        if format_mode:
            fm = (
                _FMT_RUFF_OLD.match(line) or _FMT_BLACK.match(line) or _FMT_ISORT.match(line)
                or _FMT_CARGO.match(line)
            )
            if fm:
                ln = fm.groupdict().get("line") or fm.groupdict().get("line2")
                add(_diag(fm["file"], ln, None, "format", "file is not formatted"))
                continue
            fm = _FMT_BLACK_ERR.match(line) or _FMT_PRETTIER_ERR.match(line)
            if fm:
                add(_diag(fm["file"], None, None, None, fm["msg"]))
                continue
            fm = _FMT_PRETTIER.match(line)
            if fm:
                if "code style issues" not in line.lower() and not line.lower().startswith("[warn] checking"):
                    add(_diag(fm["file"], None, None, "format", "file is not formatted"))
                continue
        if gofmt:
            gm = _GOFMT_FILE.match(line)
            if gm:
                add(_diag(gm["file"], None, None, "gofmt", "file is not formatted"))
                continue
        m = _HEAD_RUST.match(line)
        if m:
            pending = (m["code"], m["msg"], m["sev"])
            continue
        m = _HEAD_RUFF.match(line)
        if m:
            pending = (m["code"], m["msg"], "error")
            continue
        m = _HEAD_LABEL.match(line)
        if m and m["code"] not in _IGNORED_LABELS:
            pending = (m["code"], m["msg"], "error")
            continue
        m = _TSC_PAREN.match(line)
        if m:
            add(_diag(m["file"], m["line"], m["col"], m["code"], m["msg"], _sev(m["sev"])))
            continue
        m = _DASH.match(line)
        if m:
            msg, code = m["msg"], m["code"]
            rule = _PYRIGHT_RULE.search(msg)
            if rule and not code:
                code, msg = rule["rule"], msg[: rule.start()]
            add(_diag(m["file"], m["line"], m["col"], code, msg, _sev(m["sev"])))
            continue
        m = _MYPY.match(line)
        if m:
            if m["sev"] != "note":
                add(_diag(m["file"], m["line"], m["col"], m["code"], m["msg"], _sev(m["sev"])))
            continue
        m = _PYLINT.match(line)
        if m:
            sev = "error" if m["code"][0] in "EF" else "warning"
            msg = m["msg"] + (f" ({m['sym']})" if m["sym"] else "")
            add(_diag(m["file"], m["line"], m["col"], m["code"], msg, sev))
            continue
        m = _RUFF.match(line)
        if m:
            add(_diag(m["file"], m["line"], m["col"], m["code"], m["msg"]))
            continue
        m = _ESLINT_UNIX.match(line) or _ESLINT_COMPACT.match(line)
        if m:
            add(_diag(m["file"], m["line"], m["col"], m["code"], m["msg"], _sev(m["sev"])))
            continue
        m = _MAVEN.match(line)
        if m:
            add(_diag(m["file"], m["line"], m["col"], None, m["msg"]))
            continue
        m = _STYLISH.match(line)
        if m and stylish_file:
            add(_diag(stylish_file, m["line"], m["col"], m["code"], m["msg"], _sev(m["sev"])))
            continue
        m = _GENERIC.match(line)
        if m:
            msg = m["msg"]
            sev = "error"
            low = msg.lower()
            if low.startswith("warning:"):
                sev, msg = "warning", msg[8:]
            elif low.startswith(("error:", "fatal error:")):
                msg = msg.split(":", 1)[1]
            add(_diag(m["file"], m["line"], m["col"], None, msg, sev))
            continue
        if not line[0].isspace() and not line.startswith(("\u2716", "\u00d7")) and re.search(r"[/\\.]", line):
            stylish_file = line.strip()
    return out


def _is_syntax_diag(d: Diagnostic) -> bool:
    code = (d.code or "").lower()
    msg = d.message.lower()
    if code in ("syntax", "invalid-syntax", "e999") or re.fullmatch(r"ts1\d{3}", code):
        return True
    if any(s in msg for s in ("syntaxerror", "syntax error", "parsing error", "cannot parse", "unclosed delimiter",
                              "unexpected closing delimiter", "invalid syntax")):
        return True
    return not code and msg.startswith(("expected ", "unexpected token"))


# --- pytest -----------------------------------------------------------------------------------------

_PYTEST_FINAL = re.compile(r"^=*\s*(?P<body>no tests ran|\d+ [\w ,]+?) in (?P<secs>[\d.]+)s\b[^=\n]*?=*\s*$", re.M)
_PYTEST_COUNT = re.compile(r"(\d+) (failed|passed|skipped|errors?|deselected|xfailed|xpassed)\b")
_PYTEST_SHORT = re.compile(r"^(?P<kind>FAILED|ERROR) (?P<id>\S.*?)(?: - (?P<msg>.*))?\s*$")
_PYTEST_SECTION = re.compile(r"^_{3,} (?P<name>.+?) _{3,}\s*$")
_PYTEST_EXC_LOC = re.compile(r"^(?P<file>(?:[A-Za-z]:)?[^\s:][^:\n]*?\.py):(?P<line>\d+): (?P<what>[\w.]+)\s*$")
_PYTEST_FRAME = re.compile(r"^(?P<file>(?:[A-Za-z]:)?[^\s:][^:\n]*?\.py):(?P<line>\d+): in \S+")
_PYTEST_E_EXC = re.compile(r"^(?:[\w.]+(?:Error|Exception|Exit|Interrupt|Failed|Warning)\b|assert\b|Failed:|AssertionError)")
_E_FILE = re.compile(r'^E?\s*File "(?P<file>.+?)", line (?P<line>\d+)')
_STDLIB = re.compile(r"[\\/](?:unittest|importlib|asyncio|_pytest|pluggy)[\\/]|site-packages|dist-packages|[\\/]lib[\\/]python\d")


def _is_banner(line: str) -> bool:
    """``==== title ====`` / ``!!!! Interrupted !!!!`` separators (string ops: no regex backtracking)."""
    stripped = line.rstrip()
    return any(stripped.startswith(c * 3) and stripped.endswith(c * 3) for c in "=!")


@dataclass
class _Section:
    message: str = ""
    file: str | None = None
    line: int | None = None


def _pytest_sections(lines: list[str]) -> dict[str, _Section]:
    sections: dict[str, _Section] = {}
    name: str | None = None
    body: list[str] = []

    def flush() -> None:
        if name is None:
            return
        e_lines = [ln[1:].strip() for ln in body if ln.startswith("E ") or ln == "E"]
        e_lines = [e for e in e_lines if e]
        message = next((e for e in e_lines if _PYTEST_E_EXC.match(e)), e_lines[0] if e_lines else "")
        sec = _Section(message=message)
        exc = [m for m in (_PYTEST_EXC_LOC.match(ln) for ln in body) if m]
        if "SyntaxError" in message or "IndentationError" in message:
            files = [m for m in (_E_FILE.match(ln) for ln in body) if m]
            if files:
                sec.file, sec.line = files[-1]["file"], int(files[-1]["line"])
        if sec.file is None and exc:
            sec.file, sec.line = exc[-1]["file"], int(exc[-1]["line"])
        if sec.file is None:
            frames = [m for m in (_PYTEST_FRAME.match(ln) for ln in body) if m]
            local = [m for m in frames if not _STDLIB.search(m["file"])]
            pick = (local or frames)[-1] if frames else None
            if pick:
                sec.file, sec.line = pick["file"], int(pick["line"])
        sections[name] = sec

    for ln in lines:
        m = _PYTEST_SECTION.match(ln)
        if m:
            flush()
            name, body = m["name"].strip(), []
            continue
        if _is_banner(ln):
            flush()
            name, body = None, []
            continue
        if name is not None:
            body.append(ln)
    flush()
    return sections


def _parse_pytest(text: str, exit_code: int | None) -> _Parsed:
    p = _Parsed(tool="pytest")
    lines = text.split("\n")
    finals = list(_PYTEST_FINAL.finditer(text))
    if finals:
        body = finals[-1]["body"]
        if body == "no tests ran":
            p.no_tests = True
            p.passed = p.failed = p.errors = 0
        else:
            c = _counts(body, _PYTEST_COUNT)
            p.passed = c.get("passed", 0)
            p.failed = c.get("failed", 0)
            p.errors = c.get("error", 0) + c.get("errors", 0)
            p.skipped = c.get("skipped", 0)
    if exit_code == 5:
        p.no_tests = True
    sections = _pytest_sections(lines)
    collection_ids: set[str] = set()
    short = [m for m in (_PYTEST_SHORT.match(ln) for ln in lines) if m]
    for m in short:
        node = m["id"].strip()
        file = node.split("::", 1)[0]
        if "::" in node:
            name = ".".join(node.split("::")[1:])
            candidates = [name, f"ERROR at setup of {name}", f"ERROR at teardown of {name}"]
        else:
            candidates = [f"ERROR collecting {node}"]
            collection_ids.add(node)
        sec = next((sections[c] for c in candidates if c in sections), None)
        message = (m["msg"] or "").strip() or (sec.message if sec else "")
        default = "error" if m["kind"] == "ERROR" else "unknown"
        _add_failure(p, TestCaseFailure(
            test_id=node,
            file=_norm_file(sec.file if sec and sec.file else file),
            line=sec.line if sec else None,
            message=message,
            failure_type=_failure_type(message, default),
        ))
    if not short:
        for name, sec in sections.items():
            if name.startswith("ERROR collecting "):
                node = name[len("ERROR collecting "):].strip()
                collection_ids.add(node)
                default = "error"
            elif name.startswith(("ERROR at setup of ", "ERROR at teardown of ")):
                node, default = name.split(" of ", 1)[1], "error"
            elif name.startswith(("Captured ", "warnings summary", "coverage")):
                continue
            else:
                node, default = name, "unknown"
            _add_failure(p, TestCaseFailure(
                test_id=node, file=_norm_file(sec.file), line=sec.line, message=sec.message,
                failure_type=_failure_type(sec.message, default),
            ))
    if p.failed is None and p.failures:
        p.failed = sum(1 for f in p.failures if f.test_id not in collection_ids)
        p.errors = len(p.failures) - p.failed
    collection = [f for f in p.failures if f.test_id in collection_ids]
    if collection and len(collection) == len(p.failures):
        types = {f.failure_type for f in collection}
        if "syntax" in types:
            p.classification = "syntax_error"
        elif types == {"import"}:
            p.classification = _import_classification(collection[0].message, text, tests_ran=bool(p.passed))
        else:
            p.classification = "collection_error"
    elif not p.failures and exit_code in (3, 4):
        p.classification = "environment"
        p.detail = _line_of(re.compile(r"^(?:ERROR|INTERNALERROR|\w+: error):?.*$", re.M), text)
    return p


# --- unittest ----------------------------------------------------------------------------------------

_UT_SEP = re.compile(r"^={20,}\s*$")
_UT_DASH = re.compile(r"^-{20,}\s*$")
_UT_HEAD = re.compile(r"^(?P<kind>FAIL|ERROR|UNEXPECTED SUCCESS): (?P<rest>.+)$")
_UT_ID = re.compile(r"^(?P<name>\S+) \((?P<path>[\w.]+)\)")
_UT_RAN = re.compile(r"^Ran (\d+) tests? in ([\d.]+)s", re.M)
_UT_FINAL = re.compile(r"^(?P<st>OK|FAILED)(?: \((?P<body>[^)]*)\))?\s*$", re.M)
_UT_COUNT = re.compile(r"(failures|errors|skipped|expected failures|unexpected successes)=(\d+)")


def _parse_unittest(text: str, exit_code: int | None) -> _Parsed:
    p = _Parsed(tool="unittest")
    lines = text.split("\n")
    module_failures: set[str] = set()
    i = 0
    while i < len(lines):
        head = _UT_HEAD.match(lines[i])
        if not (head and i > 0 and _UT_SEP.match(lines[i - 1])):
            i += 1
            continue
        j = i + 1
        for _ in range(3):  # optional docstring line(s) before the dashes
            if j < len(lines) and _UT_DASH.match(lines[j]):
                j += 1
                break
            j += 1
        body: list[str] = []
        while j < len(lines) and not _UT_SEP.match(lines[j]):
            if _UT_DASH.match(lines[j]) and j + 1 < len(lines) and (_UT_RAN.match(lines[j + 1]) or not lines[j + 1].strip()):
                break
            body.append(lines[j])
            j += 1
        rest = head["rest"].strip()
        idm = _UT_ID.match(rest)
        if idm:
            name, path = idm["name"], idm["path"]
            if "_FailedTest" in path:
                test_id = name
                module_failures.add(test_id)
            elif path == name or path.endswith("." + name):
                test_id = path
            else:
                test_id = f"{path}.{name}"
            test_id += rest[idm.end():].rstrip()
        else:
            test_id = rest
        frames = [m for m in (_E_FILE.match(ln) for ln in body) if m]
        message = ""
        if frames:
            last_idx = max(k for k, ln in enumerate(body) if _E_FILE.match(ln))
            message = next((ln.strip() for ln in body[last_idx + 1:] if ln.strip() and not ln[0].isspace()), "")
        if not message:
            message = next((ln.strip() for ln in body if ln.strip()), "")
        local = [m for m in frames if not _STDLIB.search(m["file"])]
        pick = (local or frames)[-1] if frames else None
        default = "assertion" if head["kind"] == "FAIL" else "error"
        _add_failure(p, TestCaseFailure(
            test_id=test_id,
            file=_norm_file(pick["file"]) if pick else None,
            line=int(pick["line"]) if pick else None,
            message=message,
            failure_type=_failure_type(message, default),
        ))
        i = j
    ran = _UT_RAN.findall(text)
    if ran:
        total = int(ran[-1][0])
        finals = list(_UT_FINAL.finditer(text))
        c = {k: int(v) for k, v in _UT_COUNT.findall(finals[-1]["body"] or "")} if finals else {}
        p.failed = c.get("failures", 0) + c.get("unexpected successes", 0)
        p.errors = c.get("errors", 0)
        p.skipped = c.get("skipped", 0)
        if not finals:
            p.failed = sum(1 for f in p.failures if f.failure_type == "assertion")
            p.errors = len(p.failures) - p.failed
        p.passed = max(total - p.failed - p.errors - p.skipped, 0)
        if total == 0:
            p.no_tests = True
    if "NO TESTS RAN" in text or exit_code == 5:
        p.no_tests = True
    mods = [f for f in p.failures if f.test_id in module_failures]
    if mods and len(mods) == len(p.failures) and not p.passed:
        types = {f.failure_type for f in mods}
        if "syntax" in types:
            p.classification = "syntax_error"
        elif types == {"import"}:
            p.classification = _import_classification(mods[0].message, text, tests_ran=False)
    return p


# --- jest / vitest / mocha ------------------------------------------------------------------------------

_JS_COUNT = re.compile(r"(\d+) (failed|passed|skipped|todo|pending)\b")
_JEST_TESTS = re.compile(r"^Tests:\s+(?P<body>.+)$", re.M)
_JEST_FILE = re.compile(r"^\s*(?P<st>FAIL|PASS)\s+(?P<file>\S+)(?:\s+\([\d.]+\s*m?s\))?\s*$")
_JEST_BULLET = re.compile(r"^\s*● (?P<title>.+?)\s*$")
_JS_STACK = re.compile(r"\(?(?P<file>(?:[A-Za-z]:)?[^\s()]+?\.(?:[cm]?[jt]sx?|vue|svelte)):(?P<line>\d+):(?P<col>\d+)\)?")
_CODE_FRAME = re.compile(r"^\s*(?:>\s*)?\d+\s*\|")


def _js_location(block: list[str]) -> tuple[str | None, int | None]:
    for ln in block:
        s = ln.strip()[:500]
        if not (s.startswith(("at ", "\u276f ")) or _JS_STACK.match(s)):
            continue
        m = _JS_STACK.search(s)
        if m and "node_modules" not in m["file"]:
            return _norm_file(m["file"]), int(m["line"])
    return None, None


def _js_message(block: list[str]) -> str:
    out: list[str] = []
    for ln in block:
        s = ln.strip()
        if not s:
            if out:
                break
            continue
        if _CODE_FRAME.match(ln) or s.startswith(("at ", "\u276f ")):
            break
        out.append(s)
        if len(out) >= 3:
            break
    return "\n".join(out)


def _parse_jest(text: str, exit_code: int | None) -> _Parsed:
    p = _Parsed(tool="jest")
    lines = text.split("\n")
    current_file: str | None = None
    i = 0
    while i < len(lines):
        fm = _JEST_FILE.match(lines[i])
        if fm:
            current_file = fm["file"]
            i += 1
            continue
        bm = _JEST_BULLET.match(lines[i])
        if not bm:
            i += 1
            continue
        title = bm["title"]
        j = i + 1
        while j < len(lines) and not (_JEST_BULLET.match(lines[j]) or _JEST_FILE.match(lines[j]) or lines[j].startswith(("Test Suites:", "Summary of all failing"))):
            j += 1
        block = lines[i + 1 : j]
        i = j
        if title.startswith("Console"):
            continue
        message = _js_message(block)
        file, line = _js_location(block)
        suite = title == "Test suite failed to run"
        _add_failure(p, TestCaseFailure(
            test_id=(current_file or "test suite") if suite else title,
            file=file or _norm_file(current_file),
            line=line,
            message=message,
            failure_type=_failure_type(message, "error" if suite else "assertion"),
        ))
    tm = list(_JEST_TESTS.finditer(text))
    if tm:
        c = _counts(tm[-1]["body"], _JS_COUNT)
        p.failed, p.passed = c.get("failed", 0), c.get("passed", 0)
        p.skipped = c.get("skipped", 0) + c.get("todo", 0) + c.get("pending", 0)
        p.errors = 0
    if re.search(r"No tests found|No test files found", text):
        p.no_tests = True
    elif tm and not p.failed and not p.passed and not p.failures:
        p.no_tests = True
    suites = [f for f in p.failures if f.failure_type in ("import", "syntax", "type", "error") and not p.passed]
    if suites and len(suites) == len(p.failures) and not p.failed:
        types = {f.failure_type for f in suites}
        if types == {"import"}:
            p.classification = _import_classification(suites[0].message, text, tests_ran=False)
        elif "syntax" in types:
            p.classification = "syntax_error"
    return p


_VT_TESTS = re.compile(r"^\s*Tests\s+(?P<body>\d+ \w+.*?)\s*(?:\((?P<total>\d+)\))?\s*$", re.M)
_VT_FAIL = re.compile(r"^\s*FAIL\s+(?P<file>\S+?)\s+>\s+(?P<title>.+?)\s*$")
_VT_FAIL_SUITE = re.compile(r"^\s*FAIL\s+(?P<file>\S+)\s+\[\s*(?P<inner>.+?)\s*\]\s*$")
_VT_FILE_LINE = re.compile(r"^\s*\u276f\s+(?P<file>\S+?\.\w+)\s+\(\d+ tests?")
_VT_X = re.compile(r"^\s*[\u00d7\u2717]\s+(?P<title>.+?)(?:\s+\d+\s*ms)?\s*$")


def _parse_vitest(text: str, exit_code: int | None) -> _Parsed:
    p = _Parsed(tool="vitest")
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = _VT_FAIL_SUITE.match(lines[i]) or _VT_FAIL.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not (_VT_FAIL.match(lines[j]) or _VT_FAIL_SUITE.match(lines[j]) or lines[j].lstrip().startswith("⎯")):
            j += 1
        block = lines[i + 1 : j]
        message = _js_message(block)
        file, line = _js_location(block)
        suite = "inner" in m.groupdict()
        title = m["file"] if suite else m["title"]
        _add_failure(p, TestCaseFailure(
            test_id=title, file=file or _norm_file(m["file"]), line=line, message=message,
            failure_type=_failure_type(message, "error" if suite else "assertion"),
        ))
        i = j
    if not p.failures:
        current: str | None = None
        for k, ln in enumerate(lines):
            fm = _VT_FILE_LINE.match(ln)
            if fm:
                current = fm["file"]
                continue
            xm = _VT_X.match(ln)
            if xm:
                nxt = lines[k + 1].strip() if k + 1 < len(lines) else ""
                msg = nxt[1:].strip() if nxt.startswith("→") else ""
                _add_failure(p, TestCaseFailure(test_id=xm["title"], file=_norm_file(current), message=msg,
                                                failure_type=_failure_type(msg, "assertion")))
    tm = list(_VT_TESTS.finditer(text))
    if tm:
        c = _counts(tm[-1]["body"], _JS_COUNT)
        p.failed, p.passed = c.get("failed", 0), c.get("passed", 0)
        p.skipped = c.get("skipped", 0) + c.get("todo", 0)
        p.errors = 0
    if re.search(r"No test files found|No test suite found", text):
        p.no_tests = True
    if p.failures and not p.passed and not p.failed:
        types = {f.failure_type for f in p.failures}
        if types == {"import"}:
            p.classification = _import_classification(p.failures[0].message, text, tests_ran=False)
        elif "syntax" in types:
            p.classification = "syntax_error"
    return p


_MOCHA_COUNT = re.compile(r"^\s*(\d+) (passing|failing|pending)\b", re.M)
_MOCHA_FAIL = re.compile(r"^\s{2}(\d+)\) (?P<title>.+?)\s*$")


def _parse_mocha(text: str, exit_code: int | None) -> _Parsed:
    p = _Parsed(tool="mocha")
    c = {word: int(n) for n, word in _MOCHA_COUNT.findall(text)}
    if c:
        p.passed, p.failed, p.skipped, p.errors = c.get("passing", 0), c.get("failing", 0), c.get("pending", 0), 0
    lines = text.split("\n")
    in_failures = False
    for k, ln in enumerate(lines):
        if re.match(r"^\s*\d+ failing", ln):
            in_failures = True
            continue
        m = _MOCHA_FAIL.match(ln) if in_failures else None
        if not m:
            continue
        title_parts = [m["title"]]
        j = k + 1
        while j < len(lines) and not title_parts[-1].endswith(":") and lines[j].strip().endswith(":"):
            title_parts.append(lines[j].strip())
            j += 1
        block = lines[j:j + 12]
        message = _js_message(block)
        file, line = _js_location(block)
        _add_failure(p, TestCaseFailure(test_id=" ".join(t.rstrip(":") for t in title_parts), file=file, line=line,
                                        message=message, failure_type=_failure_type(message, "assertion")))
    if c and not c.get("passing") and not c.get("failing"):
        p.no_tests = True
    return p


# --- go test -------------------------------------------------------------------------------------------

_GO_RESULT = re.compile(r"^(?P<indent>\s*)--- (?P<st>FAIL|PASS|SKIP): (?P<name>\S+) \((?P<secs>[\d.]+)s\)")
_GO_RUN = re.compile(r"^=== (?:RUN|CONT|PAUSE)\s+(?P<name>\S+)")
_GO_LOC = re.compile(r"^\s+(?P<file>[\w./\\-]+\.go):(?P<line>\d+): (?P<msg>.*)$")
_GO_PKG = re.compile(r"^(?P<st>ok|FAIL|\?)\s+(?P<pkg>[^\s\[]+)(?:\s+(?P<rest>.*))?$")
_GO_PANIC = re.compile(r"^panic: (?P<msg>.+)$")
_GO_RUNNING = re.compile(r"^\s+(?P<name>Test\w+(?:/\S+)?) \(")


def _parse_go_test(text: str, exit_code: int | None) -> _Parsed:
    p = _Parsed(tool="go test")
    lines = text.split("\n")
    locs: dict[str, list[tuple[str, int, str]]] = {}
    fails: list[list[str]] = []  # [name, package]
    pending: list[list[str]] = []
    current: str | None = None
    top = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    saw_verbose = False
    pkgs = {"ok": 0, "FAIL": 0, "?": 0}
    no_tests_to_run = 0
    panic_msg = ""
    timeout = False
    build_failed = setup_failed = False
    for k, ln in enumerate(lines):
        m = _GO_RUN.match(ln)
        if m:
            current, saw_verbose = m["name"], True
            continue
        m = _GO_RESULT.match(ln)
        if m:
            if not m["indent"]:
                top[m["st"]] += 1
            if m["st"] == "PASS":
                saw_verbose = True
            if m["st"] == "FAIL":
                entry = [m["name"], ""]
                fails.append(entry)
                pending.append(entry)
                current = m["name"]
            continue
        m = _GO_LOC.match(ln)
        if m and current:
            locs.setdefault(current, []).append((m["file"], int(m["line"]), m["msg"].strip()))
            continue
        m = _GO_PANIC.match(ln)
        if m:
            panic_msg = panic_msg or m["msg"].strip()
            if "test timed out" in panic_msg:
                timeout = True
                for nxt in lines[k + 1 : k + 20]:
                    rm = _GO_RUNNING.match(nxt)
                    if rm and not any(f[0] == rm["name"] for f in fails):
                        entry = [rm["name"], ""]
                        fails.append(entry)
                        pending.append(entry)
            continue
        m = _GO_PKG.match(ln)
        if m:
            st, rest = m["st"], m["rest"] or ""
            pkgs[st] += 1
            if "[no tests to run]" in rest:
                no_tests_to_run += 1
            if "[build failed]" in rest:
                build_failed = True
            if "[setup failed]" in rest:
                setup_failed = True
            for entry in pending:
                entry[1] = m["pkg"]
            pending = []
    for name, pkg in fails:
        loc = locs.get(name) or []
        message = "\n".join(msg for _, _, msg in loc[:3]) or (panic_msg if panic_msg else "")
        ftype = "timeout" if timeout and not loc else _failure_type(message, "assertion" if loc else "error")
        _add_failure(p, TestCaseFailure(
            test_id=f"{pkg}.{name}" if pkg else name,
            file=_norm_file(loc[0][0]) if loc else None,
            line=loc[0][1] if loc else None,
            message=message,
            failure_type=ftype,
        ))
    p.failed = top["FAIL"] if top["FAIL"] or fails else (0 if pkgs["ok"] or pkgs["FAIL"] else None)
    if not top["FAIL"] and fails:
        p.failed = len(fails)
    if saw_verbose:
        p.passed, p.skipped = top["PASS"], top["SKIP"]
    p.errors = 0 if p.failed is not None else None
    if any(pkgs.values()):
        parts = [f"{n} package{'s' if n != 1 else ''} {label}" for label, n in (("ok", pkgs["ok"]), ("failed", pkgs["FAIL"])) if n]
        p.extra = ", ".join(parts)
    if (pkgs["ok"] == no_tests_to_run) and not pkgs["FAIL"] and (pkgs["?"] or no_tests_to_run) and not fails:
        p.no_tests = True
    if setup_failed:
        p.classification = "missing_dependency" if re.search(r"no required module provides|cannot find module providing|missing go\.sum entry", text) else "build_error"
    elif build_failed and not fails:
        p.classification = "build_error"
    return p


# --- cargo test ----------------------------------------------------------------------------------------

_CARGO_RESULT = re.compile(r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored", re.M)
_CARGO_TEST = re.compile(r"^test (?P<name>\S+) \.\.\. (?P<st>ok|FAILED|ignored)")
_CARGO_STDOUT = re.compile(r"^---- (?P<name>\S+) stdout ----\s*$")
_CARGO_PANIC_NEW = re.compile(r"^thread '(?P<t>[^']*)' panicked at (?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+):\s*$")
_CARGO_PANIC_OLD = re.compile(r"^thread '(?P<t>[^']*)' panicked at '(?P<msg>.*)', (?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+)\s*$")


def _parse_cargo_test(text: str, exit_code: int | None) -> _Parsed:
    p = _Parsed(tool="cargo test")
    lines = text.split("\n")
    results = _CARGO_RESULT.findall(text)
    if results:
        p.passed = sum(int(r[0]) for r in results)
        p.failed = sum(int(r[1]) for r in results)
        p.skipped = sum(int(r[2]) for r in results)
        p.errors = 0
        if p.passed + p.failed == 0:
            p.no_tests = True
    failed_names = [m["name"] for m in (_CARGO_TEST.match(ln) for ln in lines) if m and m["st"] == "FAILED"]
    details: dict[str, tuple[str | None, int | None, str]] = {}
    for k, ln in enumerate(lines):
        sm = _CARGO_STDOUT.match(ln)
        if not sm:
            continue
        file: str | None = None
        line_no: int | None = None
        message = ""
        for j in range(k + 1, min(k + 40, len(lines))):
            cur = lines[j]
            if _CARGO_STDOUT.match(cur) or cur.startswith(("failures:", "test result:")):
                break
            om = _CARGO_PANIC_OLD.match(cur)
            if om:
                file, line_no, message = om["file"], int(om["line"]), om["msg"]
                break
            nm = _CARGO_PANIC_NEW.match(cur)
            if nm:
                file, line_no = nm["file"], int(nm["line"])
                msg_lines = [x.strip() for x in lines[j + 1 : j + 4] if x.strip() and not x.startswith("note:")]
                message = "\n".join(msg_lines)
                break
        details[sm["name"]] = (file, line_no, message)
    for name in failed_names or list(details):
        file, line_no, message = details.get(name, (None, None, ""))
        _add_failure(p, TestCaseFailure(test_id=name, file=_norm_file(file), line=line_no, message=message,
                                        failure_type=_failure_type(message, "assertion")))
    return p


# --- audits --------------------------------------------------------------------------------------------

_PIP_AUDIT_ROW = re.compile(r"^(?P<name>[A-Za-z0-9][\w.\-]*)\s+(?P<ver>\S+)\s+(?P<id>(?:PYSEC|GHSA|CVE|OSV|GSD|PVE)-[\w.-]+)(?:\s+(?P<fix>\S.*?))?\s*$")
_NPM_SEVERITY = re.compile(r"^Severity: (?P<sev>\w+)")


def _parse_audit(text: str) -> _Parsed:
    p = _Parsed(tool="audit")
    lines = text.split("\n")
    for k, ln in enumerate(lines):
        m = _PIP_AUDIT_ROW.match(ln.strip())
        if m:
            fix = f" (fix: {m['fix']})" if m["fix"] else ""
            p.diagnostics.append(Diagnostic(code=m["id"], message=f"{m['name']} {m['ver']}: {m['id']}{fix}"))
            continue
        s = _NPM_SEVERITY.match(ln.strip())
        if s:
            package = next((x.strip() for x in reversed(lines[:k]) if x.strip()), "")
            title = lines[k + 1].strip() if k + 1 < len(lines) else ""
            p.diagnostics.append(Diagnostic(code=s["sev"].lower(), message=f"{package}: {title}".strip(": ")))
    p.diagnostics = p.diagnostics[:MAX_ITEMS]
    if p.diagnostics:
        p.classification = "vulnerabilities"
    elif _NETWORK.search(text):
        p.classification = "environment"
        p.detail = _line_of(_NETWORK, text)
    return p


# --- tool detection --------------------------------------------------------------------------------------

_COMMAND_TOOLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p), name)
    for p, name in [
        (r"\bpy\.?test\b", "pytest"),
        (r"\bvitest\b", "vitest"),
        (r"\bjest\b", "jest"),
        (r"\bmocha\b", "mocha"),
        (r"-m\s+unittest\b|\bunittest\b|manage\.py\s+test\b", "unittest"),
        (r"\bgo\s+test\b", "go test"),
        (r"\bcargo\s+(?:test|nextest)\b", "cargo test"),
        (r"\bgofmt\b", "gofmt"),
        (r"\bgoimports\b", "goimports"),
        (r"\bcargo\s+fmt\b|\brustfmt\b", "rustfmt"),
        (r"\bcargo\s+clippy\b", "clippy"),
        (r"\bcargo\s+(?:build|check)\b", "cargo"),
        (r"\bgo\s+vet\b", "go vet"),
        (r"\bgo\s+build\b", "go build"),
        (r"\bgolangci-lint\b", "golangci-lint"),
        (r"\bmypy\b", "mypy"),
        (r"\bpyright\b", "pyright"),
        (r"\btsc\b", "tsc"),
        (r"\bruff\s+format\b", "ruff format"),
        (r"\bruff\b", "ruff"),
        (r"\bflake8\b", "flake8"),
        (r"\bpylint\b", "pylint"),
        (r"\beslint\b", "eslint"),
        (r"\bblack\b", "black"),
        (r"\bisort\b", "isort"),
        (r"\bprettier\b", "prettier"),
        (r"\bbiome\b", "biome"),
        (r"\bpip-audit\b", "pip-audit"),
        (r"\b(?:npm|pnpm|yarn)\s+audit\b", "npm audit"),
    ]
]
_TEST_PARSERS: dict[str, Callable[[str, int | None], _Parsed]] = {
    "pytest": _parse_pytest,
    "unittest": _parse_unittest,
    "jest": _parse_jest,
    "vitest": _parse_vitest,
    "mocha": _parse_mocha,
    "go test": _parse_go_test,
    "cargo test": _parse_cargo_test,
}
_FORMATTERS = {"gofmt", "goimports", "rustfmt", "ruff format", "black", "isort", "prettier", "biome"}


def _sniff_test_tool(text: str) -> str:
    if re.search(r"^=+ (?:test session starts|FAILURES|ERRORS|short test summary info) =+\s*$", text, re.M) or re.search(
        r"^=+ (?:\d+ \w+.*|no tests ran) in [\d.]+s.*=+\s*$|^\d+ (?:passed|failed|errors?|skipped|deselected)\b[\w ,]* in [\d.]+s",
        text, re.M,
    ):
        return "pytest"
    if re.search(r"^Ran \d+ tests? in ", text, re.M):
        return "unittest"
    if re.search(r"^\s*Test Files\s+\d+ \w+", text, re.M) or re.search(r"^\s*Tests\s+\d+ (?:failed|passed)\b.*\|", text, re.M):
        return "vitest"
    if re.search(r"^Tests:\s+\d+", text, re.M) or re.search(r"^Test Suites:", text, re.M):
        return "jest"
    if re.search(r"^test result: (?:ok|FAILED)\.", text, re.M):
        return "cargo test"
    if re.search(r"^\s*--- (?:FAIL|PASS):", text, re.M) or re.search(
        r"^(?:ok|FAIL|\?)\s+\S+\s+(?:[\d.]+s|\[no test files\]|\[build failed\]|\(cached\))", text, re.M
    ):
        return "go test"
    if re.search(r"^\s*\d+ passing \(", text, re.M):
        return "mocha"
    return ""


def detect_tool(kind: CheckKind, command: str, text: str) -> str:
    lowered = command.lower()
    for pattern, name in _COMMAND_TOOLS:
        if pattern.search(lowered):
            return name
    if kind == CheckKind.TEST:
        return _sniff_test_tool(text)
    return ""


def _display_name(tool: str, command: str) -> str:
    if tool:
        return tool
    tokens = program_tokens(command)
    if not tokens:
        return command.strip()[:40] or "command"
    names = [norm_program(t) if k == 0 else t for k, t in enumerate(tokens)]
    if len(names) >= 3 and names[1] == "-m":
        return names[2]
    if len(names) >= 3 and names[1] in ("run", "exec", "run-script"):
        return " ".join(names[:3])[:40]
    if len(names) >= 2 and re.fullmatch(r"[\w:.-]+", names[1]) and not names[1].startswith("-"):
        return " ".join(names[:2])[:40]
    return names[0][:40]


# --- summary --------------------------------------------------------------------------------------------


def _plural(n: int, word: str) -> str:
    if word == "error":
        return f"{n} error{'s' if n != 1 else ''}"
    return f"{n} {word}"


def _counts_text(p: _Parsed) -> str:
    parts = []
    for value, word in ((p.failed, "failed"), (p.errors, "error"), (p.passed, "passed"), (p.skipped, "skipped")):
        if value:
            parts.append(_plural(value, word))
    return ", ".join(parts)


_FORMAT_CODES = {"format", "unformatted", "gofmt"}


def _diag_text(diags: list[Diagnostic]) -> str:
    if diags and all(d.code in _FORMAT_CODES for d in diags):
        n = len({d.file for d in diags})
        return f"{n} file{'s' if n != 1 else ''} need{'s' if n == 1 else ''} formatting"
    errors = [d for d in diags if d.severity == "error"]
    warnings = len(diags) - len(errors)
    files = {d.file for d in diags if d.file}
    text = _plural(len(errors), "error") if errors else ""
    if warnings:
        text = ", ".join(t for t in (text, f"{warnings} warning{'s' if warnings != 1 else ''}") if t)
    if len(files) > 1:
        text += f" in {len(files)} files"
    elif len(files) == 1:
        text += f" in {next(iter(files))}"
    return text


def _summary(name: str, status: str, classification: str, p: _Parsed, duration_s: float, exit_code: int | None, detail: str) -> str:
    dur = f" ({duration_s:.1f}s)" if duration_s > 0 else ""
    if status == "timeout":
        counts = _counts_text(p)
        return f"{name}: timed out{dur}" + (f" after {counts}" if counts else "")
    if status == "passed":
        what = _counts_text(p) or p.extra or ("no issues found" if not p.diagnostics else _diag_text(p.diagnostics) + " (non-blocking)")
        return f"{name}: {what}{dur}"
    if classification == "no_tests":
        return f"{name}: no tests ran{dur}"
    if classification in ERROR_CLASSIFICATIONS:
        label = classification.replace("_", " ")
        return f"{name}: {label}" + (f" — {detail[:160]}" if detail else "") + dur
    what = _counts_text(p)
    if not what and p.diagnostics:
        what = _diag_text(p.diagnostics)
    if not what and p.failures:
        what = f"{len(p.failures)} failed"
    if not what:
        what = f"failed (exit code {exit_code})"
    hint = ""
    if classification not in ("test_failure", "type_errors", "lint_errors", "unknown", "") and detail:
        hint = f" — {classification.replace('_', ' ')}: {detail[:160]}"
    elif classification == "unknown" and detail:
        hint = f" — {detail[:160]}"
    return f"{name}: {what}{dur}{hint}"


# --- entry point ----------------------------------------------------------------------------------------


def _failing_classification(kind: CheckKind, p: _Parsed, text: str, command: str, exit_code: int | None) -> tuple[str, str]:
    if exit_code in (127, 9009):
        return "command_not_found", _command_not_found(text, command) or f"exit code {exit_code}"
    if not p.structured():
        line = _command_not_found(text, command)
        if line:
            return "command_not_found", line
    if p.classification:
        detail = p.detail or (p.failures[0].message.split("\n")[0] if p.failures else "")
        if not detail and p.diagnostics:
            detail = p.diagnostics[0].message
        return p.classification, detail
    if p.no_tests:
        return "no_tests", ""
    if p.failures:
        first = p.failures[0].message.split("\n")[0]
        if {f.failure_type for f in p.failures} == {"syntax"}:
            return "syntax_error", first
        return "test_failure", first
    errors = [d for d in p.diagnostics if d.severity == "error"] or p.diagnostics
    if errors:
        syntax = next((d for d in errors if _is_syntax_diag(d)), None)
        if syntax is not None:
            return "syntax_error", syntax.message
        by_kind = {
            CheckKind.TEST: "build_error", CheckKind.BUILD: "build_error", CheckKind.TYPECHECK: "type_errors",
            CheckKind.LINT: "lint_errors", CheckKind.FORMAT: "lint_errors", CheckKind.AUDIT: "vulnerabilities",
        }
        return by_kind[kind], errors[0].message
    if not p.structured():
        cls, line = _generic_classification(text, command)
        if cls:
            return cls, line
    if kind == CheckKind.BUILD:
        return "build_error", ""
    tail = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return "unknown", (tail[-1][:200] if tail else "")


def parse_output(
    kind: CheckKind,
    command: str,
    output: str,
    exit_code: int | None,
    timed_out: bool = False,
    *,
    duration_s: float = 0.0,
) -> CheckResult:
    """Parse ``output`` of ``command`` (a ``kind`` check) into a :class:`CheckResult`."""
    kind = CheckKind(kind)
    text = _clean(output or "")
    tool = detect_tool(kind, command, text)
    if tool in _TEST_PARSERS:
        p = _TEST_PARSERS[tool](text, exit_code)
        if tool not in ("pytest", "unittest"):
            for d in _parse_diagnostics(text):
                if d not in p.diagnostics and len(p.diagnostics) < MAX_ITEMS:
                    p.diagnostics.append(d)
    elif kind == CheckKind.AUDIT or tool in ("pip-audit", "npm audit"):
        p = _parse_audit(text)
        p.tool = tool or p.tool
    else:
        gofmt = tool in ("gofmt", "goimports") and bool(re.search(r"(?:^|\s)-l(?:\s|$)", command))
        p = _Parsed(tool=tool, diagnostics=_parse_diagnostics(
            text, format_mode=kind == CheckKind.FORMAT or tool in _FORMATTERS, gofmt=gofmt,
        ))
        if gofmt and any(d.code == "gofmt" for d in p.diagnostics):
            p.force_fail = True
    name = _display_name(p.tool or tool, command)

    detail = ""
    status: CheckStatus
    if timed_out:
        status, classification = "timeout", "timeout"
    elif exit_code is None:
        status, classification = "error", "unknown"
    elif exit_code == 0 and p.force_fail:
        status, classification = "failed", "lint_errors"
    elif exit_code == 0 and kind == CheckKind.TEST and p.no_tests:
        status, classification = "failed", "no_tests"
    elif exit_code == 0:
        status, classification = "passed", "passed"
    else:
        classification, detail = _failing_classification(kind, p, text, command, exit_code)
        status = "error" if classification in ERROR_CLASSIFICATIONS else "failed"

    return CheckResult(
        kind=kind,
        command=command,
        status=status,
        exit_code=exit_code,
        duration_s=round(duration_s, 3),
        passed=p.passed,
        failed=p.failed,
        errors=p.errors,
        skipped=p.skipped,
        failures=p.failures[:MAX_ITEMS],
        diagnostics=p.diagnostics[:MAX_ITEMS],
        classification=classification,
        summary=_summary(name, status, classification, p, duration_s, exit_code, detail),
        output_tail=text[-OUTPUT_TAIL_CHARS:],
    )
