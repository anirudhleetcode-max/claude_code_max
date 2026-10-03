"""Deterministic project discovery: languages, tooling, commands and layout.

Everything here is evidence-based: a command is only suggested when a manifest,
config file or dependency in the repository points at the tool. Secret scanning
records locations (path, line, kind) only, never values.
"""

from __future__ import annotations

import contextlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_engineer.core.util import atomic_write_json, is_binary_bytes
from ai_engineer.repo.files import (
    find_test_dir,
    is_generated_content,
    is_generated_path,
    is_git_repo,
    is_lockfile,
    is_test_path,
    language_of,
    list_files,
)
from ai_engineer.security.secrets import scan_text

COMMAND_KINDS: tuple[str, ...] = ("test", "lint", "format", "typecheck", "build", "install", "run")

_MAX_TEST_FILES = 500
_MAX_GENERATED = 200
_MAX_SECRETS = 100
_MAX_LIST = 300
_SECRET_SCAN_MAX_BYTES = 512 * 1024
_SECRET_SCAN_BUDGET_BYTES = 64 * 1024 * 1024
_SECRET_SCAN_MAX_FILES = 20_000
_MANIFEST_MAX_BYTES = 2 * 1024 * 1024
_MANIFEST_MAX_DEPTH = 3

# Languages that describe data/docs/config rather than program code.
_NON_CODE_LANGUAGES: frozenset[str] = frozenset(
    {
        "markdown", "restructuredtext", "asciidoc", "json", "yaml", "toml", "xml", "ini", "html",
        "css", "scss", "less", "dockerfile", "makefile", "cmake", "starlark", "protobuf", "graphql",
        "sql", "batch",
    }
)

_BINARY_EXTS: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".psd", ".pdf", ".zip",
        ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".class", ".so", ".dll", ".dylib",
        ".exe", ".bin", ".o", ".a", ".lib", ".pyc", ".pyo", ".whl", ".egg", ".woff", ".woff2", ".ttf",
        ".otf", ".eot", ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac", ".ogg", ".webm", ".sqlite",
        ".db", ".parquet", ".npy", ".npz", ".pkl", ".pt", ".onnx", ".h5", ".lockb",
    }
)


class CommandSuggestion(BaseModel):
    """A command the project likely uses, with where the evidence came from."""

    command: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class ProjectProfile(BaseModel):
    """Deterministic summary of a repository (cached in ``.agent/project.json``)."""

    root: str
    is_git: bool
    git_branch: str | None = None
    file_count: int = 0
    total_bytes: int = 0
    languages: dict[str, dict[str, int]] = Field(default_factory=dict)
    primary_language: str | None = None
    frameworks: list[str] = Field(default_factory=list)
    package_managers: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    test_dirs: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    commands: dict[str, list[CommandSuggestion]] = Field(default_factory=dict)
    config_files: list[str] = Field(default_factory=list)
    doc_files: list[str] = Field(default_factory=list)
    generated_files: list[str] = Field(default_factory=list)
    secret_findings: list[dict[str, Any]] = Field(default_factory=list)
    notable_files: list[str] = Field(default_factory=list)
    summary: str = ""

    def best_command(self, kind: str) -> str | None:
        """The highest-confidence command for ``kind`` (e.g. ``"test"``), if any."""
        suggestions = self.commands.get(kind) or []
        return suggestions[0].command if suggestions else None


# --------------------------------------------------------------------------- helpers


def _read_text(root: Path, rel: str, max_bytes: int = _MANIFEST_MAX_BYTES) -> str | None:
    path = root / rel
    try:
        if path.stat().st_size > max_bytes:
            return None
        with path.open("rb") as fh:
            data = fh.read(max_bytes + 1)
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    if is_binary_bytes(data):
        return None
    return data.decode("utf-8", errors="replace")


def _load_toml(root: Path, rel: str) -> dict[str, Any] | None:
    text = _read_text(root, rel)
    if text is None:
        return None
    try:
        return tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return None


def _load_json(root: Path, rel: str) -> dict[str, Any] | None:
    text = _read_text(root, rel)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _get(data: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _req_name(spec: str) -> str | None:
    spec = spec.strip()
    if not spec or spec.startswith(("#", "-", "git+", "http:", "https:", "file:", ".")):
        return None
    m = re.match(r"[A-Za-z0-9][A-Za-z0-9._\-]*", spec)
    return m.group(0).lower().replace("_", "-") if m else None


def _depth(path: str) -> int:
    return path.count("/")


def _git_branch(root: Path) -> str | None:
    git = shutil.which("git")
    if git is not None:
        try:
            proc = subprocess.run(
                [git, "symbolic-ref", "--short", "-q", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=15,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    head = root / ".git" / "HEAD"
    with contextlib.suppress(OSError):
        content = head.read_text(encoding="utf-8").strip()
        if content.startswith("ref: refs/heads/"):
            return content.removeprefix("ref: refs/heads/")
    return None


class _Commands:
    def __init__(self) -> None:
        self._items: dict[str, list[CommandSuggestion]] = {k: [] for k in COMMAND_KINDS}

    def add(self, kind: str, command: str, source: str, confidence: float) -> None:
        bucket = self._items[kind]
        for existing in bucket:
            if existing.command == command:
                if confidence > existing.confidence:
                    existing.confidence = confidence
                    existing.source = source
                return
        bucket.append(CommandSuggestion(command=command, source=source, confidence=round(confidence, 2)))

    def result(self) -> dict[str, list[CommandSuggestion]]:
        # Stable sort keeps insertion order among equal confidences.
        return {k: sorted(v, key=lambda s: -s.confidence) for k, v in self._items.items()}


# --------------------------------------------------------------------------- python


class _PyEvidence:
    def __init__(self) -> None:
        self.deps: set[str] = set()
        self.dev_groups: set[str] = set()
        self.pyproject: dict[str, Any] | None = None
        self.has_pyproject = False


def _collect_python(root: Path, files: list[str]) -> _PyEvidence:
    ev = _PyEvidence()
    for rel in files:
        if _depth(rel) > _MANIFEST_MAX_DEPTH:
            continue
        name = posixpath.basename(rel)
        if name == "pyproject.toml":
            data = _load_toml(root, rel)
            if data is None:
                continue
            if rel == "pyproject.toml":
                ev.pyproject = data
                ev.has_pyproject = True
            ev.deps |= _pyproject_deps(data, ev.dev_groups)
        elif name == "Pipfile":
            data = _load_toml(root, rel)
            if data:
                for section in ("packages", "dev-packages"):
                    table = data.get(section)
                    if isinstance(table, dict):
                        ev.deps |= {str(k).lower().replace("_", "-") for k in table}
        elif re.fullmatch(r"(?:dev-|test-)?requirements(?:[-_.][\w.-]+)?\.(?:txt|in)", name) or (
            rel.startswith("requirements/") and name.endswith((".txt", ".in"))
        ):
            text = _read_text(root, rel)
            if text:
                for line in text.splitlines():
                    dep = _req_name(line)
                    if dep:
                        ev.deps.add(dep)
        elif name == "setup.py":
            text = _read_text(root, rel)
            if text:
                ev.deps |= _setup_py_deps(text)
        elif name == "setup.cfg":
            text = _read_text(root, rel)
            if text:
                for line in text.splitlines():
                    if line[:1] in (" ", "\t"):
                        dep = _req_name(line)
                        if dep:
                            ev.deps.add(dep)
    return ev


_SETUP_KW = re.compile(r"\b(?:install_requires|tests_require|setup_requires|extras_require)\s*=\s*([\[{])")
_QUOTED = re.compile(r"""["']([^"'\n]+)["']""")


def _setup_py_deps(text: str) -> set[str]:
    """Requirement names inside ``install_requires=[...]``-style arguments of setup.py."""
    deps: set[str] = set()
    for m in _SETUP_KW.finditer(text):
        opener = m.group(1)
        closer = "]" if opener == "[" else "}"
        depth = 0
        end = len(text)
        for i in range(m.start(1), min(len(text), m.start(1) + 20_000)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    end = i
                    break
        for q in _QUOTED.finditer(text, m.start(1), end):
            name = _req_name(q.group(1))
            if name:
                deps.add(name)
    return deps


def _pyproject_deps(data: dict[str, Any], dev_groups: set[str]) -> set[str]:
    deps: set[str] = set()

    def add_list(items: Any) -> None:
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    name = _req_name(item)
                    if name:
                        deps.add(name)

    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    assert isinstance(project, dict)
    add_list(project.get("dependencies"))
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        for group, items in optional.items():
            dev_groups.add(str(group))
            add_list(items)
    groups = data.get("dependency-groups")
    if isinstance(groups, dict):
        for items in groups.values():
            add_list(items)
    poetry = _get(data, "tool", "poetry")
    if isinstance(poetry, dict):
        for key in ("dependencies", "dev-dependencies"):
            table = poetry.get(key)
            if isinstance(table, dict):
                deps |= {str(k).lower().replace("_", "-") for k in table}
        poetry_groups = poetry.get("group")
        if isinstance(poetry_groups, dict):
            for group in poetry_groups.values():
                table = _get(group, "dependencies")
                if isinstance(table, dict):
                    deps |= {str(k).lower().replace("_", "-") for k in table}
    add_list(_get(data, "tool", "uv", "dev-dependencies"))
    pdm_dev = _get(data, "tool", "pdm", "dev-dependencies")
    if isinstance(pdm_dev, dict):
        for items in pdm_dev.values():
            add_list(items)
    deps.discard("python")
    return deps


def _ini_has_section(text: str | None, section: str) -> bool:
    if not text:
        return False
    return re.search(rf"^\[{re.escape(section)}\]", text, re.M) is not None


def _python_commands(
    root: Path, files: list[str], file_set: set[str], ev: _PyEvidence, cmds: _Commands, frameworks: set[str]
) -> list[str]:
    """Add Python command suggestions; returns entry points discovered."""
    pyproject = ev.pyproject or {}
    tool = pyproject.get("tool") if isinstance(pyproject.get("tool"), dict) else {}
    assert isinstance(tool, dict)
    setup_cfg = _read_text(root, "setup.cfg") if "setup.cfg" in file_set else None
    tox_ini = _read_text(root, "tox.ini") if "tox.ini" in file_set else None
    precommit = _read_text(root, ".pre-commit-config.yaml") if ".pre-commit-config.yaml" in file_set else None
    has_py = any(p.endswith(".py") for p in files)

    # tests
    pytest_sources: list[str] = []
    if "pytest" in ev.deps:
        pytest_sources.append("pytest dependency")
    if isinstance(tool.get("pytest"), dict):
        pytest_sources.append("pyproject.toml [tool.pytest]")
    if "pytest.ini" in file_set:
        pytest_sources.append("pytest.ini")
    if any(posixpath.basename(p) == "conftest.py" for p in files):
        pytest_sources.append("conftest.py")
    if _ini_has_section(setup_cfg, "tool:pytest"):
        pytest_sources.append("setup.cfg [tool:pytest]")
    if _ini_has_section(tox_ini, "pytest"):
        pytest_sources.append("tox.ini [pytest]")
    if pytest_sources:
        frameworks.add("pytest")
        cmds.add("test", "python -m pytest", pytest_sources[0], 0.9)
    elif has_py and any(re.fullmatch(r"test.*\.py", posixpath.basename(p)) for p in files):
        cmds.add("test", "python -m unittest discover", "test*.py files", 0.6)
    if tox_ini is not None:
        cmds.add("test", "tox", "tox.ini", 0.5)

    # lint
    ruff = None
    if isinstance(tool.get("ruff"), dict):
        ruff = "pyproject.toml [tool.ruff]"
    elif "ruff.toml" in file_set or ".ruff.toml" in file_set:
        ruff = "ruff.toml" if "ruff.toml" in file_set else ".ruff.toml"
    elif "ruff" in ev.deps:
        ruff = "ruff dependency"
    elif precommit and "ruff" in precommit:
        ruff = ".pre-commit-config.yaml"
    if ruff:
        cmds.add("lint", "ruff check .", ruff, 0.9)
    flake8 = None
    if ".flake8" in file_set:
        flake8 = ".flake8"
    elif _ini_has_section(setup_cfg, "flake8"):
        flake8 = "setup.cfg [flake8]"
    elif _ini_has_section(tox_ini, "flake8"):
        flake8 = "tox.ini [flake8]"
    if flake8:
        cmds.add("lint", "flake8", flake8, 0.6 if ruff else 0.8)
    if (isinstance(tool.get("pylint"), dict) or ".pylintrc" in file_set or "pylintrc" in file_set) and not ruff:
        cmds.add("lint", "pylint .", "pylint configuration", 0.5)

    # format
    black = None
    if isinstance(tool.get("black"), dict):
        black = "pyproject.toml [tool.black]"
    elif "black" in ev.deps:
        black = "black dependency"
    elif precommit and "psf/black" in precommit:
        black = ".pre-commit-config.yaml"
    if ruff and (_get(tool, "ruff", "format") is not None or not black):
        cmds.add("format", "ruff format .", ruff, 0.85 if _get(tool, "ruff", "format") is not None else 0.6)
    if black:
        cmds.add("format", "black .", black, 0.85)

    # typecheck
    mypy_cfg = tool.get("mypy")
    if isinstance(mypy_cfg, dict) or "mypy.ini" in file_set or ".mypy.ini" in file_set or _ini_has_section(
        setup_cfg, "mypy"
    ):
        configured_targets = isinstance(mypy_cfg, dict) and ("files" in mypy_cfg or "packages" in mypy_cfg)
        source = "pyproject.toml [tool.mypy]" if isinstance(mypy_cfg, dict) else "mypy configuration"
        cmds.add("typecheck", "mypy" if configured_targets else "mypy .", source, 0.85)
    elif "mypy" in ev.deps:
        cmds.add("typecheck", "mypy .", "mypy dependency", 0.6)
    if "pyrightconfig.json" in file_set or isinstance(tool.get("pyright"), dict):
        cmds.add("typecheck", "pyright", "pyright configuration", 0.8)

    # install
    if "poetry.lock" in file_set or isinstance(tool.get("poetry"), dict):
        cmds.add("install", "poetry install", "poetry.lock" if "poetry.lock" in file_set else "[tool.poetry]", 0.9)
    if "uv.lock" in file_set:
        cmds.add("install", "uv sync", "uv.lock", 0.9)
    if "pdm.lock" in file_set:
        cmds.add("install", "pdm install", "pdm.lock", 0.85)
    if "Pipfile" in file_set:
        cmds.add("install", "pipenv install --dev", "Pipfile", 0.8)
    if ev.has_pyproject or "setup.py" in file_set:
        project = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
        extras = _get(project, "optional-dependencies")
        if isinstance(extras, dict) and "dev" in extras:
            cmds.add("install", 'pip install -e ".[dev]"', "pyproject.toml optional-dependencies.dev", 0.75)
        else:
            cmds.add("install", "pip install -e .", "pyproject.toml" if ev.has_pyproject else "setup.py", 0.7)
    for req in ("requirements-dev.txt", "requirements.txt"):
        if req in file_set:
            cmds.add("install", f"pip install -r {req}", req, 0.8 if req == "requirements.txt" else 0.7)
    for env_file in ("environment.yml", "environment.yaml"):
        if env_file in file_set:
            cmds.add("install", f"conda env create -f {env_file}", env_file, 0.6)

    # build
    if isinstance(pyproject.get("build-system"), dict):
        cmds.add("build", "python -m build", "pyproject.toml [build-system]", 0.7)

    # entry points and run
    entry_points: list[str] = []
    scripts = _get(pyproject, "project", "scripts")
    if isinstance(scripts, dict):
        for name, target in sorted(scripts.items()):
            if isinstance(target, str):
                entry_points.append(target)
                cmds.add("run", str(name), "pyproject.toml [project.scripts]", 0.6)
    poetry_scripts = _get(tool, "poetry", "scripts")
    if isinstance(poetry_scripts, dict):
        for name, target in sorted(poetry_scripts.items()):
            if isinstance(target, str):
                entry_points.append(target)
                cmds.add("run", f"poetry run {name}", "[tool.poetry.scripts]", 0.6)
    for rel in files:
        if posixpath.basename(rel) == "__main__.py" and not is_test_path(rel):
            entry_points.append(rel)
            pkg_dir = posixpath.dirname(rel)
            parts = pkg_dir.split("/")
            if parts and parts[0] == "src":
                parts = parts[1:]
            if parts and all(p.isidentifier() for p in parts):
                cmds.add("run", f"python -m {'.'.join(parts)}", rel, 0.55)
    for rel in files:
        if _depth(rel) > 1:
            continue
        name = posixpath.basename(rel)
        if name in ("main.py", "app.py", "manage.py", "wsgi.py", "asgi.py", "server.py") and not rel.startswith(
            ("tests/", "test/")
        ):
            if name == "server.py" and _depth(rel) > 0:
                continue
            entry_points.append(rel)
            if name == "manage.py":
                frameworks.add("django")
                cmds.add("run", f"python {rel} runserver", rel, 0.7)
                if not pytest_sources:
                    cmds.add("test", f"python {rel} test", rel, 0.7)
            elif name in ("main.py", "app.py") and _depth(rel) == 0:
                cmds.add("run", f"python {rel}", rel, 0.5)
    return entry_points


_PY_FRAMEWORKS: dict[str, str] = {
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "starlette": "starlette",
    "aiohttp": "aiohttp",
    "tornado": "tornado",
    "pyramid": "pyramid",
    "streamlit": "streamlit",
    "pytest": "pytest",
}


# --------------------------------------------------------------------------- javascript


_JS_FRAMEWORKS: dict[str, str] = {
    "react": "react",
    "next": "next",
    "express": "express",
    "vue": "vue",
    "@angular/core": "angular",
    "@nestjs/core": "nestjs",
    "svelte": "svelte",
    "@sveltejs/kit": "svelte",
    "vite": "vite",
    "jest": "jest",
    "vitest": "vitest",
    "mocha": "mocha",
    "nuxt": "nuxt",
    "fastify": "fastify",
    "koa": "koa",
    "electron": "electron",
    "@playwright/test": "playwright",
    "cypress": "cypress",
    "@remix-run/react": "remix",
    "astro": "astro",
}

_NPM_DEFAULT_TEST = "no test specified"


def _js_package_manager(file_set: set[str], pkg: dict[str, Any] | None) -> str | None:
    declared = pkg.get("packageManager") if pkg else None
    if isinstance(declared, str):
        name = declared.split("@", 1)[0].strip()
        if name in ("npm", "pnpm", "yarn", "bun"):
            return name
    if "pnpm-lock.yaml" in file_set or "pnpm-workspace.yaml" in file_set:
        return "pnpm"
    if "yarn.lock" in file_set:
        return "yarn"
    if "bun.lockb" in file_set or "bun.lock" in file_set:
        return "bun"
    if "package-lock.json" in file_set or "npm-shrinkwrap.json" in file_set:
        return "npm"
    return "npm" if pkg is not None else None


_PNPM_BUILTINS = frozenset(
    {"add", "install", "i", "update", "up", "remove", "rm", "link", "unlink", "publish", "exec", "dlx",
     "create", "init", "why", "outdated", "list", "ls", "audit", "store", "env", "setup", "patch",
     "server", "rebuild", "prune", "pack", "fetch", "import", "deploy", "config", "run", "root", "bin"}
)


def _run_script(pm: str, script: str) -> str:
    if pm == "npm":
        return f"npm {script}" if script in ("test", "start") else f"npm run {script}"
    if pm == "pnpm":
        return f"pnpm run {script}" if script in _PNPM_BUILTINS else f"pnpm {script}"
    if pm == "yarn":
        return f"yarn {script}"
    return f"bun run {script}"


def _js_commands(
    root: Path, files: list[str], file_set: set[str], cmds: _Commands, frameworks: set[str], managers: set[str]
) -> list[str]:
    entry_points: list[str] = []
    root_pkg: dict[str, Any] | None = None
    all_deps: set[str] = set()
    for rel in files:
        if posixpath.basename(rel) != "package.json" or _depth(rel) > _MANIFEST_MAX_DEPTH:
            continue
        data = _load_json(root, rel)
        if data is None:
            continue
        if rel == "package.json":
            root_pkg = data
        for key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            table = data.get(key)
            if isinstance(table, dict):
                all_deps |= {str(k) for k in table}
        sub_pm = _js_package_manager(file_set, data)
        if sub_pm:
            managers.add(sub_pm)
    for dep, fw in _JS_FRAMEWORKS.items():
        if dep in all_deps:
            frameworks.add(fw)
    if any(re.match(r"vite\.config\.", posixpath.basename(p)) for p in files):
        frameworks.add("vite")
    if any(re.match(r"next\.config\.", posixpath.basename(p)) for p in files):
        frameworks.add("next")
    if "angular.json" in file_set:
        frameworks.add("angular")

    has_tsconfig = "tsconfig.json" in file_set
    if root_pkg is None:
        if has_tsconfig:
            cmds.add("typecheck", "npx tsc --noEmit", "tsconfig.json", 0.6)
        return entry_points

    pm = _js_package_manager(file_set, root_pkg) or "npm"
    managers.add(pm)
    scripts = root_pkg.get("scripts")
    scripts = {str(k): str(v) for k, v in scripts.items()} if isinstance(scripts, dict) else {}

    test_script = scripts.get("test")
    if test_script and _NPM_DEFAULT_TEST not in test_script:
        cmds.add("test", _run_script(pm, "test"), "package.json scripts.test", 0.95)
    for name in sorted(scripts):
        if name.startswith("test:") and _NPM_DEFAULT_TEST not in scripts[name]:
            cmds.add("test", _run_script(pm, name), f"package.json scripts.{name}", 0.6)
    if not cmds.result()["test"]:
        if "vitest" in all_deps:
            cmds.add("test", "npx vitest run", "vitest dependency", 0.6)
        elif "jest" in all_deps:
            cmds.add("test", "npx jest", "jest dependency", 0.6)
        elif "mocha" in all_deps:
            cmds.add("test", "npx mocha", "mocha dependency", 0.5)

    for kind, names in (
        ("lint", ("lint",)),
        ("format", ("format", "fmt", "prettier")),
        ("typecheck", ("typecheck", "type-check", "tsc", "types", "check-types", "typecheck:all")),
        ("build", ("build",)),
    ):
        for name in names:
            if name in scripts:
                cmds.add(kind, _run_script(pm, name), f"package.json scripts.{name}", 0.9)
                break
    if not cmds.result()["lint"] and any(
        re.match(r"(?:\.eslintrc(?:\.\w+)?|eslint\.config\.\w+)$", posixpath.basename(p)) for p in files if _depth(p) == 0
    ):
        cmds.add("lint", "npx eslint .", "eslint configuration", 0.6)
    if not cmds.result()["format"] and any(
        re.match(r"(?:\.prettierrc(?:\.\w+)?|prettier\.config\.\w+)$", posixpath.basename(p)) for p in files if _depth(p) == 0
    ):
        cmds.add("format", "npx prettier --write .", "prettier configuration", 0.5)
    if not cmds.result()["typecheck"] and has_tsconfig:
        cmds.add("typecheck", "npx tsc --noEmit", "tsconfig.json", 0.6)
    if "start" in scripts:
        cmds.add("run", _run_script(pm, "start"), "package.json scripts.start", 0.8)
    if "dev" in scripts:
        cmds.add("run", _run_script(pm, "dev"), "package.json scripts.dev", 0.75)
    lock = {
        "npm": "package-lock.json",
        "pnpm": "pnpm-lock.yaml",
        "yarn": "yarn.lock",
        "bun": "bun.lockb",
    }[pm]
    cmds.add("install", f"{pm} install", lock if lock in file_set else "package.json", 0.9)

    main = root_pkg.get("main")
    if isinstance(main, str) and main.strip():
        entry_points.append(posixpath.normpath(main.strip()).removeprefix("./"))
    bins = root_pkg.get("bin")
    if isinstance(bins, str):
        entry_points.append(posixpath.normpath(bins).removeprefix("./"))
    elif isinstance(bins, dict):
        for _, target in sorted(bins.items()):
            if isinstance(target, str):
                entry_points.append(posixpath.normpath(target).removeprefix("./"))
    return entry_points


# --------------------------------------------------------------------------- other ecosystems


def _other_commands(
    root: Path, files: list[str], file_set: set[str], cmds: _Commands, frameworks: set[str], managers: set[str]
) -> list[str]:
    entry_points: list[str] = []
    if "go.mod" in file_set:
        managers.add("go")
        cmds.add("test", "go test ./...", "go.mod", 0.9)
        cmds.add("build", "go build ./...", "go.mod", 0.9)
        cmds.add("lint", "go vet ./...", "go.mod", 0.8)
        cmds.add("format", "gofmt -l .", "go.mod", 0.7)
        if ".golangci.yml" in file_set or ".golangci.yaml" in file_set:
            cmds.add("lint", "golangci-lint run", ".golangci.yml", 0.85)
        gomod = _read_text(root, "go.mod") or ""
        for dep, fw in (
            ("github.com/gin-gonic/gin", "gin"),
            ("github.com/labstack/echo", "echo"),
            ("github.com/gofiber/fiber", "fiber"),
        ):
            if dep in gomod:
                frameworks.add(fw)
        cmds.add("install", "go mod download", "go.mod", 0.7)
    elif any(posixpath.basename(p) == "go.mod" for p in files):
        managers.add("go")
    for rel in files:
        if rel == "main.go" or re.fullmatch(r"cmd/[^/]+/main\.go", rel):
            entry_points.append(rel)
    if "main.go" in file_set:
        cmds.add("run", "go run .", "main.go", 0.6)

    if "Cargo.toml" in file_set or any(posixpath.basename(p) == "Cargo.toml" for p in files):
        managers.add("cargo")
    if "Cargo.toml" in file_set:
        cmds.add("test", "cargo test", "Cargo.toml", 0.9)
        cmds.add("build", "cargo build", "Cargo.toml", 0.9)
        cmds.add("lint", "cargo clippy", "Cargo.toml", 0.8)
        cmds.add("format", "cargo fmt --check", "Cargo.toml", 0.7)
        cargo_text = _read_text(root, "Cargo.toml") or ""
        for dep, fw in (("actix-web", "actix"), ("axum", "axum"), ("rocket", "rocket")):
            if re.search(rf"^\s*{re.escape(dep)}\s*=", cargo_text, re.M):
                frameworks.add(fw)
        if "src/main.rs" in file_set:
            cmds.add("run", "cargo run", "src/main.rs", 0.7)
    for rel in files:
        if rel.endswith("src/main.rs") or re.search(r"(?:^|/)src/bin/[^/]+\.rs$", rel):
            entry_points.append(rel)

    if "pom.xml" in file_set:
        managers.add("maven")
        wrapper = "./mvnw" if "mvnw" in file_set else "mvn"
        cmds.add("test", f"{wrapper} -q test", "pom.xml", 0.9)
        cmds.add("build", f"{wrapper} -q package", "pom.xml", 0.85)
        cmds.add("install", f"{wrapper} -q install -DskipTests", "pom.xml", 0.6)
        if "org.springframework" in (_read_text(root, "pom.xml") or ""):
            frameworks.add("spring")
    gradle_file = next((g for g in ("build.gradle.kts", "build.gradle") if g in file_set), None)
    if gradle_file or "settings.gradle" in file_set or "settings.gradle.kts" in file_set:
        managers.add("gradle")
        gradle = "./gradlew" if "gradlew" in file_set else "gradle"
        cmds.add("test", f"{gradle} test", gradle_file or "settings.gradle", 0.9)
        cmds.add("build", f"{gradle} build", gradle_file or "settings.gradle", 0.85)
        if gradle_file and "org.springframework" in (_read_text(root, gradle_file) or ""):
            frameworks.add("spring")

    if "Gemfile" in file_set:
        managers.add("bundler")
        cmds.add("install", "bundle install", "Gemfile", 0.85)
        gemfile = _read_text(root, "Gemfile") or ""
        if re.search(r"""^\s*gem\s+['"]rails['"]""", gemfile, re.M):
            frameworks.add("rails")
            cmds.add("test", "bin/rails test", "Gemfile (rails)", 0.7)
        if re.search(r"""^\s*gem\s+['"]rspec""", gemfile, re.M) or ".rspec" in file_set:
            cmds.add("test", "bundle exec rspec", "rspec", 0.85)
        if re.search(r"""^\s*gem\s+['"]rubocop""", gemfile, re.M) or ".rubocop.yml" in file_set:
            cmds.add("lint", "bundle exec rubocop", "rubocop", 0.8)

    if "composer.json" in file_set:
        managers.add("composer")
        cmds.add("install", "composer install", "composer.json", 0.85)
        composer = _load_json(root, "composer.json") or {}
        req = {**(composer.get("require") or {}), **(composer.get("require-dev") or {})}
        if "laravel/framework" in req:
            frameworks.add("laravel")
        if any(str(k).startswith("symfony/") for k in req):
            frameworks.add("symfony")
        if "phpunit/phpunit" in req:
            cmds.add("test", "vendor/bin/phpunit", "composer.json require-dev", 0.8)
        scripts = composer.get("scripts")
        if isinstance(scripts, dict) and "test" in scripts:
            cmds.add("test", "composer test", "composer.json scripts.test", 0.85)

    makefile = next((m for m in ("Makefile", "makefile", "GNUmakefile") if m in file_set), None)
    if makefile:
        text = _read_text(root, makefile) or ""
        targets = set(re.findall(r"^([A-Za-z0-9_.\-]+)[ \t]*:(?!=)", text, re.M))
        for target, kind, conf in (
            ("test", "test", 0.7),
            ("check", "test", 0.7),
            ("lint", "lint", 0.7),
            ("fmt", "format", 0.7),
            ("format", "format", 0.7),
            ("typecheck", "typecheck", 0.7),
            ("build", "build", 0.7),
            ("install", "install", 0.6),
            ("run", "run", 0.6),
        ):
            if target in targets:
                cmds.add(kind, f"make {target}", f"{makefile} target {target}", conf)

    for name in ("Dockerfile", "Containerfile"):
        if name in file_set:
            entry_points.append(name)
    return entry_points


def _manager_evidence(files: list[str], py: _PyEvidence, managers: set[str]) -> None:
    names = {posixpath.basename(p) for p in files if _depth(p) <= _MANIFEST_MAX_DEPTH}
    if "poetry.lock" in names or isinstance(_get(py.pyproject, "tool", "poetry"), dict):
        managers.add("poetry")
    if "uv.lock" in names or isinstance(_get(py.pyproject, "tool", "uv"), dict):
        managers.add("uv")
    if "pdm.lock" in names:
        managers.add("pdm")
    if "Pipfile" in names:
        managers.add("pipenv")
    if "environment.yml" in names or "environment.yaml" in names:
        managers.add("conda")
    pip_evidence = (
        any(re.fullmatch(r"(?:dev-|test-)?requirements(?:[-_.][\w.-]+)?\.txt", n) for n in names)
        or "setup.py" in names
        or (py.has_pyproject and not ({"poetry", "uv", "pdm"} & managers))
    )
    if pip_evidence:
        managers.add("pip")
    if "pom.xml" in names:
        managers.add("maven")
    if {"build.gradle", "build.gradle.kts"} & names:
        managers.add("gradle")
    if "Gemfile" in names:
        managers.add("bundler")
    if "composer.json" in names:
        managers.add("composer")


_MANAGER_ORDER = (
    "npm", "pnpm", "yarn", "bun", "pip", "poetry", "uv", "pdm", "pipenv", "conda", "cargo", "go",
    "maven", "gradle", "bundler", "composer",
)


# --------------------------------------------------------------------------- file classification


_CONFIG_NAMES: frozenset[str] = frozenset(
    {
        "pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "noxfile.py", "pytest.ini", "mypy.ini",
        ".mypy.ini", ".flake8", ".pylintrc", "pylintrc", "ruff.toml", ".ruff.toml", "pyrightconfig.json",
        "pipfile", "environment.yml", "environment.yaml", "package.json", "jsconfig.json", "angular.json",
        ".babelrc", ".npmrc", ".nvmrc", ".node-version", "go.mod", "go.work", "cargo.toml", "rustfmt.toml",
        ".rustfmt.toml", "clippy.toml", "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
        "settings.gradle.kts", "gradle.properties", "gemfile", ".rubocop.yml", ".rspec", "composer.json",
        "makefile", "gnumakefile", "dockerfile", "containerfile", "docker-compose.yml", "docker-compose.yaml",
        "compose.yml", "compose.yaml", ".dockerignore", ".editorconfig", ".pre-commit-config.yaml",
        ".gitignore", ".gitattributes", ".env.example", ".env.sample", ".python-version", ".tool-versions",
        ".golangci.yml", ".golangci.yaml", "renovate.json", ".gitlab-ci.yml", ".travis.yml",
        "azure-pipelines.yml", "bitbucket-pipelines.yml", "jenkinsfile", "cmakelists.txt", "deno.json",
        "turbo.json", "nx.json", "lerna.json", "pnpm-workspace.yaml", "biome.json", ".swcrc", "vercel.json",
        "netlify.toml", "procfile", "app.yaml", "serverless.yml",
    }
)
_CONFIG_PREFIXES: tuple[str, ...] = (
    "tsconfig", ".eslintrc", "eslint.config.", ".prettierrc", "prettier.config.", "vite.config.",
    "vitest.config.", "jest.config.", "webpack.config.", "babel.config.", "next.config.", "nuxt.config.",
    "svelte.config.", "rollup.config.", "tailwind.config.", "postcss.config.", "playwright.config.",
    "cypress.config.", "astro.config.", "requirements", "dockerfile.", "karma.conf",
)


def _is_config(rel: str) -> bool:
    if is_lockfile(rel):
        return False
    lower = posixpath.basename(rel).lower()
    if rel.startswith(".github/workflows/") or rel.startswith(".circleci/") or rel == ".github/dependabot.yml":
        return True
    if _depth(rel) > 2:
        return False
    if lower in _CONFIG_NAMES:
        return True
    if lower.startswith("requirements"):
        return lower.endswith((".txt", ".in"))
    return lower.startswith(_CONFIG_PREFIXES)


_DOC_LANGS = frozenset({"markdown", "restructuredtext", "asciidoc"})


def _is_doc(rel: str, language: str | None) -> bool:
    if language in _DOC_LANGS:
        return True
    parts = rel.split("/")
    if parts[0] in ("docs", "doc", "documentation") and posixpath.splitext(rel)[1].lower() in (".txt", ".html"):
        return True
    return False


_NOTABLE_ROOT = re.compile(
    r"(?i)^(readme(\..*)?|license(\..*)?|licence(\..*)?|copying(\..*)?|contributing(\..*)?|changelog(\..*)?|"
    r"security\.md|code_of_conduct\.md|claude\.md|agents\.md|dockerfile|containerfile|makefile|"
    r"docker-compose\.ya?ml|compose\.ya?ml|jenkinsfile|\.gitlab-ci\.yml|\.travis\.yml|azure-pipelines\.yml|"
    r"bitbucket-pipelines\.yml)$"
)


def _is_notable(rel: str) -> bool:
    name = posixpath.basename(rel)
    if _depth(rel) == 0 and _NOTABLE_ROOT.match(name):
        return True
    if name in ("CLAUDE.md", "AGENTS.md"):
        return True
    if rel.startswith(".github/workflows/") and rel.endswith((".yml", ".yaml")):
        return True
    return rel in (".circleci/config.yml", ".github/CODEOWNERS", ".github/pull_request_template.md")


# --------------------------------------------------------------------------- summary


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _summary(p: ProjectProfile) -> str:
    name = Path(p.root).name or p.root
    parts: list[str] = []
    vcs = f"git repository (branch {p.git_branch})" if p.is_git and p.git_branch else (
        "git repository" if p.is_git else "directory (not a git repository)"
    )
    lead = f"{p.primary_language} project" if p.primary_language else "project"
    parts.append(f"'{name}' is a {lead} in a {vcs} with {p.file_count} files ({_human_bytes(p.total_bytes)}).")
    if p.languages:
        ranked = sorted(p.languages.items(), key=lambda kv: (-kv[1]["bytes"], kv[0]))[:5]
        parts.append("Languages: " + ", ".join(f"{lang} ({v['files']} files)" for lang, v in ranked) + ".")
    if p.frameworks:
        parts.append("Frameworks/tools: " + ", ".join(p.frameworks) + ".")
    if p.package_managers:
        parts.append("Package managers: " + ", ".join(p.package_managers) + ".")
    labels = {"test": "Test", "lint": "Lint", "format": "Format", "typecheck": "Typecheck", "build": "Build"}
    cmd_bits = [f"{label}: `{p.commands[k][0].command}`" for k, label in labels.items() if p.commands.get(k)]
    if cmd_bits:
        parts.append("; ".join(cmd_bits) + ".")
    else:
        parts.append("No test/lint/build commands were detected.")
    if p.test_files:
        where = f" under {', '.join(p.test_dirs[:3])}" if p.test_dirs else ""
        parts.append(f"{len(p.test_files)} test files{where}.")
    if p.entry_points:
        parts.append("Entry points: " + ", ".join(p.entry_points[:5]) + ".")
    if p.secret_findings:
        parts.append(
            f"Potential secrets detected at {len(p.secret_findings)} location(s); values were not recorded."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------- main entry


def discover(root: Path, files: list[str] | None = None) -> ProjectProfile:
    """Build a :class:`ProjectProfile` for ``root`` (deterministic, no network)."""
    root = Path(root)
    is_git = is_git_repo(root)
    files = sorted(files) if files is not None else list_files(root)
    file_set = set(files)

    languages: dict[str, dict[str, int]] = {}
    total_bytes = 0
    sizes: dict[str, int] = {}
    for rel in files:
        try:
            size = os.stat(root / rel).st_size
        except OSError:
            continue
        sizes[rel] = size
        total_bytes += size
        if is_generated_path(rel):
            continue
        lang = language_of(rel)
        if lang is None:
            continue
        entry = languages.setdefault(lang, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += size

    code_langs = {k: v for k, v in languages.items() if k not in _NON_CODE_LANGUAGES}
    ranked_source = code_langs or languages
    primary = (
        min(ranked_source.items(), key=lambda kv: (-kv[1]["bytes"], -kv[1]["files"], kv[0]))[0]
        if ranked_source
        else None
    )

    cmds = _Commands()
    frameworks: set[str] = set()
    managers: set[str] = set()
    py = _collect_python(root, files)
    for dep, fw in _PY_FRAMEWORKS.items():
        if dep in py.deps:
            frameworks.add(fw)
    entry_points: list[str] = []
    # Order matters only for ties: the primary ecosystem's commands are added first.
    ecosystems = ["python", "js", "other"]
    if primary in ("javascript", "typescript"):
        ecosystems = ["js", "python", "other"]
    elif primary not in (None, "python"):
        ecosystems = ["other", "python", "js"]
    for eco in ecosystems:
        if eco == "python":
            if py.has_pyproject or py.deps or any(p.endswith(".py") for p in files):
                entry_points += _python_commands(root, files, file_set, py, cmds, frameworks)
        elif eco == "js":
            entry_points += _js_commands(root, files, file_set, cmds, frameworks, managers)
        else:
            entry_points += _other_commands(root, files, file_set, cmds, frameworks, managers)
    _manager_evidence(files, py, managers)

    test_files = [p for p in files if is_test_path(p)]
    test_dirs = sorted({d for d in (find_test_dir(p) for p in files) if d is not None})

    generated: list[str] = []
    secrets: list[dict[str, Any]] = []
    scanned_bytes = 0
    scanned_files = 0
    for rel in files:
        known_size = sizes.get(rel)
        if known_size is None:
            continue
        size = known_size
        path_generated = is_generated_path(rel)
        if path_generated:
            if len(generated) < _MAX_GENERATED:
                generated.append(rel)
            continue
        if posixpath.splitext(rel)[1].lower() in _BINARY_EXTS or size == 0:
            continue
        if size > _SECRET_SCAN_MAX_BYTES:
            continue
        if scanned_bytes + size > _SECRET_SCAN_BUDGET_BYTES or scanned_files >= _SECRET_SCAN_MAX_FILES:
            continue
        try:
            with (root / rel).open("rb") as fh:
                data = fh.read(_SECRET_SCAN_MAX_BYTES + 1)
        except OSError:
            continue
        if len(data) > _SECRET_SCAN_MAX_BYTES:
            continue
        scanned_bytes += size
        scanned_files += 1
        if is_binary_bytes(data):
            continue
        text = data.decode("utf-8", errors="replace")
        if is_generated_content(text[:2048]):
            if len(generated) < _MAX_GENERATED:
                generated.append(rel)
            continue
        if len(secrets) >= _MAX_SECRETS:
            continue
        seen_here: set[tuple[int, str]] = set()
        for finding in scan_text(text, max_findings=_MAX_SECRETS):
            if (finding.line, finding.kind) in seen_here:
                continue
            seen_here.add((finding.line, finding.kind))
            secrets.append({"path": rel, "line": finding.line, "kind": finding.kind})
            if len(secrets) >= _MAX_SECRETS:
                break

    profile = ProjectProfile(
        root=str(root.resolve()),
        is_git=is_git,
        git_branch=_git_branch(root) if is_git else None,
        file_count=len(files),
        total_bytes=total_bytes,
        languages=dict(sorted(languages.items())),
        primary_language=primary,
        frameworks=sorted(frameworks),
        package_managers=[m for m in _MANAGER_ORDER if m in managers],
        entry_points=list(dict.fromkeys(entry_points))[:_MAX_LIST],
        test_dirs=test_dirs[:_MAX_LIST],
        test_files=test_files[:_MAX_TEST_FILES],
        commands=cmds.result(),
        config_files=[p for p in files if _is_config(p)][:_MAX_LIST],
        doc_files=[p for p in files if _is_doc(p, language_of(p))][:_MAX_LIST],
        generated_files=sorted(generated),
        secret_findings=secrets,
        notable_files=[p for p in files if _is_notable(p)][:_MAX_LIST],
    )
    profile.summary = _summary(profile)
    return profile


def save_profile(profile: ProjectProfile, path: Path) -> None:
    """Write the profile atomically as JSON."""
    atomic_write_json(Path(path), profile.model_dump(mode="json"))


def load_profile(path: Path) -> ProjectProfile | None:
    """Load a saved profile; ``None`` if missing or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ProjectProfile.model_validate(data)
    except (OSError, ValueError):
        return None
