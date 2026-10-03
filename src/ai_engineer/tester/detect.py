"""Choose the concrete validation commands for a project.

Configuration overrides win; otherwise the best evidence-based suggestion from
repository discovery is used. Formatting is always planned in *check* mode so
validation never rewrites files. Availability is checked with ``which`` so a
missing tool is reported instead of failing obscurely.
"""

from __future__ import annotations

import posixpath
import re
import shlex
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config.settings import ValidationSettings
from .models import CheckKind, ValidationCommand

Which = Callable[[str], str | None]

_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_OPERATORS = {"&&", "||", ";", "|", "&"}
_SHELL_BUILTINS = {"cd", "pushd", "popd", "export", "set", "source", ".", "true", ":", "setlocal"}
_JS_MANAGERS = ("npm", "pnpm", "yarn", "bun")
_PY_MANAGERS = {"pip", "poetry", "uv", "pdm", "pipenv", "conda"}

TIMEOUTS: dict[CheckKind, float] = {
    CheckKind.LINT: 300.0,
    CheckKind.FORMAT: 300.0,
    CheckKind.TYPECHECK: 600.0,
    CheckKind.BUILD: 600.0,
    CheckKind.AUDIT: 300.0,
}

# --- tokenising -----------------------------------------------------------------------------


def split_command(command: str) -> list[str]:
    """Tokenise a command line; never raises (Windows paths keep their backslashes)."""
    try:
        if "\\" in command:
            return [t.strip("\"'") for t in shlex.split(command, posix=False)]
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def norm_program(token: str) -> str:
    """``C:\\x\\Ruff.EXE`` -> ``ruff``; ``./node_modules/.bin/jest`` -> ``jest``."""
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _OPERATORS:
            segments.append([])
        else:
            segments[-1].append(tok)
    return [s for s in segments if s]


def _strip_prefix(tokens: list[str]) -> list[str]:
    """Drop leading ``VAR=value`` assignments and an ``env`` wrapper."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _ASSIGNMENT.match(tok) or (norm_program(tok) == "env" and i == 0):
            i += 1
            continue
        break
    return tokens[i:]


def program_tokens(command: str) -> list[str]:
    """Tokens of the first segment that runs a real program (skips ``cd x &&`` and assignments)."""
    for segment in _segments(split_command(command)):
        tokens = _strip_prefix(segment)
        if tokens and norm_program(tokens[0]) not in _SHELL_BUILTINS:
            return tokens
    return []


def executable_of(command: str) -> str | None:
    tokens = program_tokens(command)
    return tokens[0] if tokens else None


def has_operators(command: str) -> bool:
    return any(tok in _OPERATORS for tok in split_command(command))


def _join(tokens: Sequence[str]) -> str:
    return " ".join(shlex.quote(t) for t in tokens)


def _quote_path(path: str) -> str:
    return shlex.quote(path.replace("\\", "/"))


# --- profile access ---------------------------------------------------------------------------


def _suggestions(profile: Any, kind: str) -> list[tuple[str, str, float]]:
    """``(command, source, confidence)`` suggestions for ``kind``, best first."""
    commands = getattr(profile, "commands", None) if profile is not None else None
    if not isinstance(commands, Mapping):
        return []
    out: list[tuple[str, str, float]] = []
    for item in commands.get(kind) or []:
        if isinstance(item, str):
            cmd, source, conf = item, "profile", 0.5
        elif isinstance(item, Mapping):
            cmd, source, conf = item.get("command") or "", item.get("source") or "profile", item.get("confidence", 0.5)
        else:
            cmd = getattr(item, "command", "") or ""
            source = getattr(item, "source", "") or "profile"
            conf = getattr(item, "confidence", 0.5)
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            confidence = 0.5
        if cmd.strip():
            out.append((cmd.strip(), str(source), confidence))
    return sorted(out, key=lambda s: -s[2])


def _frameworks(profile: Any) -> set[str]:
    return {str(f).lower() for f in (getattr(profile, "frameworks", None) or [])}


# --- formatter modes ----------------------------------------------------------------------------

_CHECK_HINT = re.compile(r"(?i)(?:\bcheck\b|[-:_]check|check[-:_]|--diff\b|\bverify\b|--list-different|(?:^|\s)-l(?:\s|$))")
_SCRIPT_RUNNERS = {"npm", "pnpm", "yarn", "bun"}


def _script_name(tokens: list[str]) -> str | None:
    """``npm run format:check`` -> ``format:check``; ``yarn fmt`` -> ``fmt``."""
    if not tokens or norm_program(tokens[0]) not in _SCRIPT_RUNNERS:
        return None
    rest = [t for t in tokens[1:] if not t.startswith("-")]
    if rest and rest[0] in ("run", "run-script"):
        rest = rest[1:]
    return rest[0] if rest else None


def _find(tokens: list[str], *names: str) -> int:
    for i, tok in enumerate(tokens):
        if norm_program(tok) in names:
            return i
    return -1


def format_check_command(command: str, check_scripts: Iterable[str] = (), *, trust_unknown: bool = False) -> str | None:
    """Convert a formatter invocation to its non-modifying check mode.

    Returns None when no check mode can be derived (e.g. ``npm run format`` without a
    ``format:check`` script). Unknown commands are kept only when they look like checks
    (or when ``trust_unknown``, used for explicit configuration).
    """
    looks_check = bool(_CHECK_HINT.search(command))
    tokens = program_tokens(command)
    if not tokens or has_operators(command) or "\\" in command:
        return command if (looks_check or trust_unknown) else None
    full = split_command(command)
    offset = full.index(tokens[0]) if tokens[0] in full else 0
    toks = list(tokens)
    prog = norm_program(toks[0])

    def rebuilt(new: list[str]) -> str:
        return _join([*full[:offset], *new])

    i = _find(toks, "ruff")
    if i >= 0 and i + 1 < len(toks) and toks[i + 1] == "format":
        if "--check" in toks or "--diff" in toks:
            return command
        return rebuilt([*toks[: i + 2], "--check", *toks[i + 2 :]])
    i = _find(toks, "black")
    if i >= 0:
        if "--check" in toks or "--diff" in toks:
            return command
        return rebuilt([*toks[: i + 1], "--check", *toks[i + 1 :]])
    i = _find(toks, "prettier")
    if i >= 0:
        new = ["--check" if t in ("--write", "-w") else t for t in toks]
        if not {"--check", "-c", "--list-different", "-l"} & set(new):
            new = [*new[: i + 1], "--check", *new[i + 1 :]]
        return rebuilt(new)
    i = _find(toks, "gofmt", "goimports")
    if i >= 0:
        new = [t for t in toks if t not in ("-w", "-d", "-l")]
        return rebuilt([*new[: i + 1], "-l", *new[i + 1 :]])
    i = _find(toks, "cargo")
    if i >= 0 and i + 1 < len(toks) and toks[i + 1] == "fmt":
        if "--check" in toks:
            return command
        return rebuilt([*toks[: i + 2], "--check", *toks[i + 2 :]])
    i = _find(toks, "rustfmt")
    if i >= 0:
        return command if "--check" in toks else rebuilt([*toks[: i + 1], "--check", *toks[i + 1 :]])
    i = _find(toks, "isort")
    if i >= 0:
        if {"--check-only", "--check", "-c", "--diff"} & set(toks):
            return command
        return rebuilt([*toks[: i + 1], "--check-only", *toks[i + 1 :]])
    i = _find(toks, "biome")
    if i >= 0:
        return rebuilt([t for t in toks if t not in ("--write", "--apply", "--fix")])
    i = _find(toks, "dotnet")
    if i >= 0 and i + 1 < len(toks) and toks[i + 1] == "format":
        return command if "--verify-no-changes" in toks else rebuilt([*toks, "--verify-no-changes"])
    if prog in _SCRIPT_RUNNERS or prog in ("make", "just", "task"):
        if looks_check:
            return command
        for alt in check_scripts:
            if _CHECK_HINT.search(alt):
                return alt
        return None
    return command if (looks_check or trust_unknown) else None


def format_write_command(command: str) -> str | None:
    """Convert a (check-mode) formatter invocation to the mode that rewrites files."""
    tokens = program_tokens(command)
    if not tokens or has_operators(command) or "\\" in command:
        return None if _CHECK_HINT.search(command) else command
    full = split_command(command)
    offset = full.index(tokens[0]) if tokens[0] in full else 0
    toks = list(tokens)

    def rebuilt(new: list[str]) -> str:
        return _join([*full[:offset], *new])

    if _find(toks, "ruff") >= 0 or _find(toks, "black") >= 0 or _find(toks, "rustfmt") >= 0:
        return rebuilt([t for t in toks if t not in ("--check", "--diff")])
    i = _find(toks, "cargo")
    if i >= 0 and i + 1 < len(toks) and toks[i + 1] == "fmt":
        new = [t for t in toks if t != "--check"]
        return rebuilt(new[:-1] if new and new[-1] == "--" else new)
    i = _find(toks, "prettier")
    if i >= 0:
        new = [t for t in toks if t not in ("--check", "-c", "--list-different", "-l")]
        if "--write" not in new and "-w" not in new:
            new = [*new[: i + 1], "--write", *new[i + 1 :]]
        return rebuilt(new)
    i = _find(toks, "gofmt", "goimports")
    if i >= 0:
        new = [t for t in toks if t not in ("-l", "-d", "-w")]
        return rebuilt([*new[: i + 1], "-w", *new[i + 1 :]])
    i = _find(toks, "isort")
    if i >= 0:
        return rebuilt([t for t in toks if t not in ("--check-only", "--check", "-c", "--diff")])
    i = _find(toks, "biome")
    if i >= 0:
        return command if "--write" in toks else rebuilt([*toks[: i + 2], "--write", *toks[i + 2 :]])
    i = _find(toks, "dotnet")
    if i >= 0:
        return rebuilt([t for t in toks if t != "--verify-no-changes"])
    return None if _CHECK_HINT.search(command) else command


# --- availability ------------------------------------------------------------------------------

_PY_TOKEN = re.compile(r"(?<![\w./\\-])python(?![\w.-])")


def _check_available(command: str, which: Which, root: str | None) -> tuple[str, bool, str]:
    """Returns ``(command, available, reason)``; may rewrite ``python`` to ``python3``."""
    exe = executable_of(command)
    if not exe:
        return command, True, ""
    if "/" in exe or "\\" in exe:
        if not root:
            return command, True, ""
        base = Path(root)
        for candidate in (exe, exe + ".bat", exe + ".cmd", exe + ".exe"):
            if (base / candidate).exists():
                return command, True, ""
        return command, False, f"{exe} not found in the project"
    if which(exe):
        return command, True, ""
    if exe == "python" and which("python3"):
        return _PY_TOKEN.sub("python3", command, count=1), True, ""
    return command, False, f"{exe} not found on PATH"


def with_availability(vc: ValidationCommand, which: Which, root: str | None = None) -> ValidationCommand:
    command, available, reason = _check_available(vc.command, which, root)
    return vc.model_copy(update={"command": command, "available": available, "unavailable_reason": reason})


# --- planning ----------------------------------------------------------------------------------


def _timeout(kind: CheckKind, settings: ValidationSettings) -> float:
    return settings.test_timeout_s if kind == CheckKind.TEST else TIMEOUTS.get(kind, 600.0)


def _audit_command(profile: Any, settings: ValidationSettings, which: Which) -> ValidationCommand | None:
    if not settings.dependency_audit or profile is None:
        return None
    managers = [str(m).lower() for m in (getattr(profile, "package_managers", None) or [])]
    primary = str(getattr(profile, "primary_language", None) or "").lower()
    has_python = primary == "python" or bool(_PY_MANAGERS & set(managers))
    js_manager = next((m for m in managers if m in ("npm", "pnpm", "yarn")), None)
    order = ["js", "python"] if primary in ("javascript", "typescript") else ["python", "js"]
    timeout = TIMEOUTS[CheckKind.AUDIT]
    for candidate in order:
        if candidate == "python" and has_python and which("pip-audit"):
            return ValidationCommand(kind=CheckKind.AUDIT, command="pip-audit", source="python project", timeout_s=timeout, confidence=0.6)
        if candidate == "js" and js_manager:
            command = {
                "npm": "npm audit --audit-level=high",
                "pnpm": "pnpm audit --audit-level high",
                "yarn": "yarn audit --level high",
            }[js_manager]
            vc = ValidationCommand(kind=CheckKind.AUDIT, command=command, source=f"{js_manager} project", timeout_s=timeout, confidence=0.6)
            return with_availability(vc, which)
    return None


def plan_checks(
    profile: Any, settings: ValidationSettings, which: Which = shutil.which
) -> dict[CheckKind, ValidationCommand | None]:
    """Pick one command per check kind (None when nothing suitable is known)."""
    root = getattr(profile, "root", None) if profile is not None else None
    root = str(root) if root else None
    format_scripts = [c for c, _, _ in _suggestions(profile, "format")]
    plan: dict[CheckKind, ValidationCommand | None] = {}
    for kind in (CheckKind.TEST, CheckKind.LINT, CheckKind.FORMAT, CheckKind.TYPECHECK, CheckKind.BUILD):
        override = getattr(settings, f"{kind.value}_command", None)
        vc: ValidationCommand | None = None
        if override is not None:
            command: str | None = override.strip()
            if command and kind == CheckKind.FORMAT:
                command = format_check_command(command, format_scripts, trust_unknown=True)
            if command:
                vc = ValidationCommand(kind=kind, command=command, source="config", timeout_s=_timeout(kind, settings))
        else:
            for cmd, source, confidence in _suggestions(profile, kind.value):
                chosen: str | None = cmd
                if kind == CheckKind.FORMAT:
                    chosen = format_check_command(cmd, format_scripts)
                if chosen:
                    vc = ValidationCommand(
                        kind=kind, command=chosen, source=source, confidence=confidence, timeout_s=_timeout(kind, settings)
                    )
                    break
        plan[kind] = with_availability(vc, which, root) if vc is not None else None
    plan[CheckKind.AUDIT] = _audit_command(profile, settings, which)
    return plan


# --- targeting -----------------------------------------------------------------------------------

_PYTEST_VALUE_OPTS = {
    "-k", "-m", "-p", "-c", "-o", "-n", "-W", "-r", "--maxfail", "--rootdir", "--cov", "--cov-report",
    "--cov-config", "--cov-fail-under", "--tb", "--durations", "--deselect", "--ignore", "--ignore-glob",
    "--confcutdir", "--basetemp", "--junitxml", "--junit-xml", "--log-level", "--timeout", "--dist",
    "--import-mode", "--override-ini", "--config-file", "--numprocesses", "--reruns",
}
_GO_VALUE_OPTS = {
    "-run", "-timeout", "-count", "-tags", "-p", "-parallel", "-coverprofile", "-covermode", "-coverpkg",
    "-bench", "-benchtime", "-cpu", "-skip", "-shuffle", "-exec", "-o", "-mod", "-ldflags", "-gcflags",
}
_UNITTEST_FLAGS = {"-v", "-q", "-b", "-f", "-c", "--locals", "--buffer", "--failfast", "--catch", "--verbose", "--quiet"}
_JS_TEST_RUNNERS = ("jest", "vitest", "mocha")


def _keep_options(args: list[str], value_opts: set[str]) -> list[str]:
    """Keep option tokens (with their values); drop positional (path) arguments."""
    kept: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("-") and tok != "-":
            kept.append(tok)
            if "=" not in tok and tok in value_opts and i + 1 < len(args):
                kept.append(args[i + 1])
                i += 2
                continue
        i += 1
    return kept


def _dotted_module(path: str) -> str | None:
    p = path.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    if not p.endswith(".py"):
        return None
    return p[:-3].replace("/", ".")


def targeted_test_command(
    base: ValidationCommand, test_files: list[str], frameworks: Iterable[str] = ()
) -> ValidationCommand | None:
    """A command that runs only ``test_files`` with the same runner, or None if not derivable."""
    files = [f for f in test_files if f.strip()]
    if not files or has_operators(base.command) or "\\" in base.command:
        return None
    full = split_command(base.command)
    tokens = program_tokens(base.command)
    if not tokens:
        return None
    lead = full[: full.index(tokens[0])] if tokens[0] in full else []
    names = [norm_program(t) for t in tokens]
    quoted = [_quote_path(f) for f in files]
    command: str | None = None

    if "pytest" in names or "py.test" in names:
        i = names.index("pytest") if "pytest" in names else names.index("py.test")
        keep = _keep_options(tokens[i + 1 :], _PYTEST_VALUE_OPTS)
        command = " ".join([_join([*lead, *tokens[: i + 1], *keep]), *quoted])
    elif "unittest" in names:
        i = names.index("unittest")
        modules = [m for m in (_dotted_module(f) for f in files) if m]
        if not modules:
            return None
        flags = [t for t in tokens[i + 1 :] if t in _UNITTEST_FLAGS]
        command = _join([*lead, *tokens[: i + 1], *flags, *modules])
    elif "manage.py" in names and "test" in tokens:
        i = tokens.index("test")
        modules = [m for m in (_dotted_module(f) for f in files) if m]
        if not modules:
            return None
        command = _join([*lead, *tokens[: i + 1], *_keep_options(tokens[i + 1 :], set()), *modules])
    elif any(n in _JS_TEST_RUNNERS for n in names):
        command = " ".join([base.command, *quoted])
    elif names[0] in _SCRIPT_RUNNERS and _script_name(tokens) in ("test", "t", "unit", "test:unit"):
        fws = {f.lower() for f in frameworks}
        if not fws & {"jest", "vitest"}:
            return None
        if names[0] == "npm":
            command = " ".join([base.command, *([] if "--" in tokens else ["--"]), *quoted])
        else:
            command = " ".join([base.command, *quoted])
    elif names[0] == "go" and len(tokens) > 1 and tokens[1] == "test":
        dirs: list[str] = []
        for f in files:
            if not f.endswith(".go"):
                continue
            d = posixpath.dirname(f.replace("\\", "/").removeprefix("./"))
            pkg = f"./{d}" if d else "."
            if pkg not in dirs:
                dirs.append(pkg)
        if not dirs:
            return None
        command = _join([*lead, tokens[0], "test", *_keep_options(tokens[2:], _GO_VALUE_OPTS), *dirs])
    if command is None:
        return None
    return base.model_copy(update={"command": command, "scope": "targeted"})


_PY_EXT = (".py", ".pyi")
_JS_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".vue", ".svelte")
_PRETTIER_EXT = (*_JS_EXT, ".json", ".css", ".scss", ".less", ".md", ".mdx", ".yaml", ".yml", ".html", ".graphql")
_TARGETABLE: dict[str, tuple[tuple[str, ...], set[str]]] = {
    "ruff": (_PY_EXT, {"--config", "--select", "--ignore", "--extend-select", "--extend-ignore", "--target-version",
                       "--line-length", "--output-format", "--exclude", "--extend-exclude", "--cache-dir"}),
    "flake8": (_PY_EXT, {"--config", "--select", "--ignore", "--extend-ignore", "--max-line-length", "--format",
                         "--exclude", "--append-config"}),
    "pylint": (_PY_EXT, {"--rcfile", "--disable", "--enable", "--output-format", "-j", "--jobs", "--load-plugins",
                         "-d", "-e", "-f"}),
    "mypy": (_PY_EXT, {"--config-file", "--python-version", "--cache-dir", "--exclude", "--platform"}),
    "pyright": (_PY_EXT, {"-p", "--project", "--pythonpath", "--pythonversion"}),
    "eslint": (_JS_EXT, {"-c", "--config", "--ext", "--rule", "--format", "-f", "--ignore-path", "--parser",
                         "--max-warnings", "--plugin", "-o", "--output-file", "--resolve-plugins-relative-to"}),
    "black": (_PY_EXT, {"-l", "--line-length", "-t", "--target-version", "--config", "--include", "--exclude",
                        "--extend-exclude", "--force-exclude"}),
    "isort": (_PY_EXT, {"--settings-path", "--sp", "--profile", "-l", "--line-length"}),
    "prettier": (_PRETTIER_EXT, {"--config", "--ignore-path", "--plugin", "--parser", "--log-level"}),
    "gofmt": ((".go",), {"-r"}),
    "goimports": ((".go",), {"-local"}),
}


def _retarget(command: str, files: list[str]) -> str | None:
    if not files or has_operators(command) or "\\" in command:
        return None
    full = split_command(command)
    tokens = program_tokens(command)
    if not tokens:
        return None
    lead = full[: full.index(tokens[0])] if tokens[0] in full else []
    names = [norm_program(t) for t in tokens]
    if names[0] in _SCRIPT_RUNNERS and (len(names) < 2 or names[1] not in _TARGETABLE):
        return None
    for i, name in enumerate(names):
        if name not in _TARGETABLE:
            continue
        exts, value_opts = _TARGETABLE[name]
        end = i + 1
        if name == "ruff" and end < len(tokens) and tokens[end] in ("check", "format"):
            end += 1
        matching = [f for f in files if f.lower().endswith(exts)]
        if not matching:
            return None
        head = _join([*lead, *tokens[:end], *_keep_options(tokens[end:], value_opts)])
        return " ".join([head, *(_quote_path(f) for f in matching)])
    return None


def targeted_lint_command(base: ValidationCommand, files: list[str]) -> ValidationCommand | None:
    """Restrict a direct ruff/flake8/pylint/eslint/mypy/pyright/formatter invocation to ``files``."""
    command = _retarget(base.command, [f for f in files if f.strip()])
    if command is None:
        return None
    return base.model_copy(update={"command": command, "scope": "targeted"})


def write_format_command(profile: Any, settings: ValidationSettings, files: list[str] | None = None) -> str | None:
    """The file-rewriting formatter for the project, optionally limited to ``files``."""
    source = settings.format_command
    if source is None:
        suggestions = _suggestions(profile, "format")
        source = suggestions[0][0] if suggestions else None
    if not source or not source.strip():
        return None
    command = format_write_command(source.strip())
    if command is None:
        return None
    if files:
        return _retarget(command, files)
    return command
