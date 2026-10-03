"""Workspace file listing, language detection and generated-file heuristics."""

from __future__ import annotations

import os
import posixpath
import shutil
import stat
import subprocess
from pathlib import Path

DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".agent",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".next",
        ".nuxt",
        "coverage",
        ".gradle",
        ".idea",
        ".vscode",
        "vendor",
        "bower_components",
        ".terraform",
        ".cache",
    }
)

# Hidden directories that still hold project content (CI configuration).
_ALLOWED_HIDDEN_DIRS: frozenset[str] = frozenset({".github", ".gitlab", ".circleci"})

_GIT_TIMEOUT_S = 60


def _git_available() -> str | None:
    return shutil.which("git")


def is_git_repo(root: Path) -> bool:
    """True when ``root`` is inside a git work tree and ``git`` is usable."""
    root = Path(root)
    if not root.is_dir():
        return False
    try:
        resolved = root.resolve()
    except OSError:
        return False
    if not any((d / ".git").exists() for d in (resolved, *resolved.parents)):
        return False
    git = _git_available()
    if git is None:
        return False
    try:
        proc = subprocess.run(
            [git, "rev-parse", "--is-inside-work-tree"],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _under_ignored_dir(rel: str, include_ignored: bool) -> bool:
    parts = rel.split("/")[:-1]
    if include_ignored:
        return any(p in (".git", ".agent") for p in parts)
    return any(p in DEFAULT_IGNORE_DIRS for p in parts)


def _is_listable_file(root_str: str, rel: str, root_real: str | None) -> bool:
    """Regular files, or symlinks to regular files that stay inside the workspace."""
    full = os.path.join(root_str, rel)
    try:
        st = os.lstat(full)
    except OSError:
        return False
    if stat.S_ISREG(st.st_mode):
        return True
    if stat.S_ISLNK(st.st_mode) and root_real is not None:
        try:
            target = os.path.realpath(full)
        except OSError:
            return False
        if not os.path.isfile(target):
            return False
        try:
            return os.path.commonpath([root_real, target]) == root_real
        except ValueError:  # different drives on Windows
            return False
    return False


def _git_list(root: Path, include_ignored: bool) -> list[str] | None:
    git = _git_available()
    if git is None:
        return None
    args = [git, "-c", "core.quotepath=off", "ls-files", "-co", "-z"]
    if not include_ignored:
        args.append("--exclude-standard")
    try:
        proc = subprocess.run(
            args,
            cwd=root,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        out.append(rel)
    return out


def _walk_list(root: Path, max_files: int, include_ignored: bool) -> list[str]:
    out: list[str] = []
    root_str = str(root)
    for dirpath, dirnames, filenames in os.walk(root_str, topdown=True, followlinks=False):
        kept = []
        for d in dirnames:
            if include_ignored:
                if d in (".git", ".agent"):
                    continue
            elif d in DEFAULT_IGNORE_DIRS or (d.startswith(".") and d not in _ALLOWED_HIDDEN_DIRS):
                continue
            kept.append(d)
        dirnames[:] = sorted(kept)
        rel_dir = os.path.relpath(dirpath, root_str)
        rel_dir = "" if rel_dir == "." else Path(rel_dir).as_posix()
        for name in sorted(filenames):
            out.append(f"{rel_dir}/{name}" if rel_dir else name)
            if len(out) >= max_files * 2:  # leave headroom for later filtering
                return out
    return out


def list_files(root: Path, max_files: int = 200_000, include_ignored: bool = False) -> list[str]:
    """List workspace files as sorted, workspace-relative POSIX paths.

    Git repositories use ``git ls-files -co --exclude-standard`` (tracked plus
    untracked-but-not-ignored); otherwise a walker skips ``DEFAULT_IGNORE_DIRS`` and
    hidden directories (except CI folders such as ``.github``). ``.git`` and
    ``.agent`` are always excluded. Only regular files (or symlinks to files inside
    the workspace) that exist are returned.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    candidates: list[str] | None = None
    if is_git_repo(root):
        candidates = _git_list(root, include_ignored)
    if candidates is None:
        candidates = _walk_list(root, max_files, include_ignored)
    root_str = str(root)
    try:
        root_real: str | None = os.path.realpath(root_str)
    except OSError:
        root_real = None
    result: list[str] = []
    for rel in sorted(candidates):
        if rel.startswith("../") or rel.startswith("/"):
            continue
        if _under_ignored_dir(rel, include_ignored):
            continue
        if not _is_listable_file(root_str, rel, root_real):
            continue
        result.append(rel)
        if len(result) >= max_files:
            break
    return result


_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".pyx": "cython",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".fs": "fsharp",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".m": "objective-c",
    ".mm": "objective-c",
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sc": "scala",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".dart": "dart",
    ".lua": "lua",
    ".r": "r",
    ".pl": "perl",
    ".pm": "perl",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".jl": "julia",
    ".zig": "zig",
    ".nim": "nim",
    ".vue": "vue",
    ".svelte": "svelte",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "scss",
    ".less": "less",
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".rst": "restructuredtext",
    ".adoc": "asciidoc",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".jsonc": "json",
    ".json5": "json",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "ini",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".tf": "terraform",
    ".hcl": "terraform",
    ".dockerfile": "dockerfile",
    ".mk": "makefile",
    ".cmake": "cmake",
    ".bzl": "starlark",
}

_NAME_LANG: dict[str, str] = {
    "dockerfile": "dockerfile",
    "containerfile": "dockerfile",
    "makefile": "makefile",
    "gnumakefile": "makefile",
    "cmakelists.txt": "cmake",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "vagrantfile": "ruby",
    "podfile": "ruby",
    "jenkinsfile": "groovy",
    "build": "starlark",
    "build.bazel": "starlark",
    "workspace": "starlark",
    ".bashrc": "shell",
    ".zshrc": "shell",
    ".profile": "shell",
    "pipfile": "toml",
}


def language_of(path: str) -> str | None:
    """Best-effort language name from a file's name/extension (None if unknown)."""
    name = posixpath.basename(path.replace("\\", "/"))
    lower = name.lower()
    if lower in _NAME_LANG:
        return _NAME_LANG[lower]
    if lower.startswith("dockerfile.") or lower.startswith("containerfile."):
        return "dockerfile"
    if lower.startswith("makefile."):
        return "makefile"
    if lower.endswith(".d.ts"):
        return "typescript"
    ext = posixpath.splitext(lower)[1]
    if not ext:
        return None
    return _EXT_LANG.get(ext)


_LOCKFILES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "bun.lock",
        "poetry.lock",
        "pipfile.lock",
        "pdm.lock",
        "uv.lock",
        "cargo.lock",
        "go.sum",
        "composer.lock",
        "gemfile.lock",
        "podfile.lock",
        "mix.lock",
        "flake.lock",
        "packages.lock.json",
        "pubspec.lock",
        "gradle.lockfile",
        "conda-lock.yml",
    }
)

_GENERATED_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".min.mjs",
    "_pb2.py",
    "_pb2.pyi",
    "_pb2_grpc.py",
    ".pb.go",
    ".pb.gw.go",
    ".pb.cc",
    ".pb.h",
    ".g.dart",
    ".freezed.dart",
    ".designer.cs",
    ".map",
)

_GENERATED_DIRS: frozenset[str] = frozenset({"dist", "build", "__generated__", "generated"})


def is_lockfile(path: str) -> bool:
    """True for dependency lockfiles (package-lock.json, poetry.lock, go.sum, ...)."""
    return posixpath.basename(path.replace("\\", "/")).lower() in _LOCKFILES


def is_generated_path(path: str) -> bool:
    """Heuristic: lockfiles, minified bundles, protobuf output, build output dirs."""
    norm = path.replace("\\", "/")
    name = posixpath.basename(norm).lower()
    if name in _LOCKFILES:
        return True
    if name.endswith(_GENERATED_SUFFIXES):
        return True
    if ".generated." in name:
        return True
    parts = norm.split("/")[:-1]
    return any(p in _GENERATED_DIRS for p in parts)


_GENERATED_MARKERS_CS: tuple[str, ...] = ("@generated", "DO NOT EDIT", "Code generated by")
_GENERATED_MARKERS_CI: tuple[str, ...] = ("autogenerated", "auto-generated", "automatically generated")


def is_generated_content(head: str) -> bool:
    """Heuristic over the first ~2 KB of a file for generator markers."""
    sample = head[:2048]
    if any(marker in sample for marker in _GENERATED_MARKERS_CS):
        return True
    lowered = sample.lower()
    return any(marker in lowered for marker in _GENERATED_MARKERS_CI)


_TEST_DIR_NAMES: frozenset[str] = frozenset({"tests", "test", "__tests__", "spec", "specs", "testing"})
_JS_TEST_EXTS: tuple[str, ...] = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")


def is_test_path(path: str) -> bool:
    """True when ``path`` looks like a test file by common naming conventions."""
    norm = path.replace("\\", "/")
    name = posixpath.basename(norm)
    lower = name.lower()
    stem, ext = posixpath.splitext(lower)
    dirs = norm.split("/")[:-1]
    if ext in (".py", ".pyi"):
        if lower == "conftest.py":
            return False
        return stem.startswith("test_") or stem.endswith("_test") or stem.endswith("_tests") or lower == "tests.py"
    if ext in _JS_TEST_EXTS:
        if ".test." in lower or ".spec." in lower or "__tests__" in dirs:
            return True
        return False
    if ext == ".go":
        return stem.endswith("_test")
    if ext in (".java", ".kt", ".scala", ".cs", ".php", ".swift", ".groovy"):
        base = posixpath.splitext(name)[0]
        if base.endswith(("Test", "Tests", "Spec", "IT")) or (base.startswith("Test") and len(base) > 4):
            return True
        return "src/test" in norm or (ext == ".cs" and any(d.endswith((".Tests", ".Test")) for d in dirs))
    if ext == ".rb":
        return stem.endswith("_spec") or stem.endswith("_test") or stem.startswith("test_")
    if ext == ".rs":
        return "tests" in dirs or stem.endswith("_test") or stem == "tests"
    if ext in (".c", ".cc", ".cpp", ".cxx"):
        return stem.startswith("test_") or stem.endswith("_test") or stem.endswith("_unittest")
    if ext == ".dart":
        return stem.endswith("_test")
    if ext in (".ex", ".exs"):
        return stem.endswith("_test")
    return False


def find_test_dir(path: str) -> str | None:
    """Return the outermost test directory containing ``path`` (e.g. ``tests``)."""
    parts = path.replace("\\", "/").split("/")[:-1]
    for i, part in enumerate(parts):
        if part in _TEST_DIR_NAMES:
            return "/".join(parts[: i + 1])
        if part == "src" and i + 1 < len(parts) and parts[i + 1] == "test":
            return "/".join(parts[: i + 2])
    return None
