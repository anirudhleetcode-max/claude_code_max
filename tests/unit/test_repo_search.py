from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import ai_engineer.repo.search as search_mod
from ai_engineer.repo.search import TextMatch, find_files, glob_match, search_text
from tests.unit.test_repo_discovery import init_git, write

needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

_REAL_WHICH = shutil.which

FILES = {
    "src/app.py": "import os\n\ndef Handler():\n    return os.getenv('MODE')\n",
    "src/util/strings.py": "def handler_name(x):\n    return f'handler:{x}'\n",
    "src/web/index.ts": "export const handler = () => 'a.b(c)';\n",
    "README.md": "# Project\nThe HANDLER docs.\n",
    "node_modules/dep/index.js": "module.exports = 'handler';\n",
}


def no_rg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_mod.shutil, "which", lambda name: None if name == "rg" else _REAL_WHICH(name))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = write(tmp_path / "proj", FILES)
    (root / "image.bin").write_bytes(b"\x00handler\x00" * 10)
    return root


def paths(matches: list[TextMatch]) -> list[tuple[str, int]]:
    return [(m.path, m.line) for m in matches]


EXPECTED_HANDLER = [
    ("README.md", 2),
    ("src/app.py", 3),
    ("src/util/strings.py", 1),
    ("src/util/strings.py", 2),
    ("src/web/index.ts", 1),
]


@needs_rg
def test_search_text_with_ripgrep(project: Path) -> None:
    matches = search_text(project, "handler")
    assert paths(matches) == EXPECTED_HANDLER
    assert matches[1].text == "def Handler():"


def test_search_text_python_fallback(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_mod.shutil, "which", lambda name: None)
    matches = search_text(project, "handler")
    assert paths(matches) == EXPECTED_HANDLER
    assert matches[1].text == "def Handler():"


@pytest.mark.parametrize("use_rg", [True, False])
def test_search_options(project: Path, monkeypatch: pytest.MonkeyPatch, use_rg: bool) -> None:
    if use_rg and shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    if not use_rg:
        no_rg(monkeypatch)
    assert paths(search_text(project, "Handler", case_sensitive=True)) == [("src/app.py", 3)]
    assert paths(search_text(project, "a.b(c)", regex=False)) == [("src/web/index.ts", 1)]
    assert paths(search_text(project, r"def \w+\(")) == [("src/app.py", 3), ("src/util/strings.py", 1)]
    assert paths(search_text(project, "handler", glob="*.py")) == [
        ("src/app.py", 3),
        ("src/util/strings.py", 1),
        ("src/util/strings.py", 2),
    ]
    assert paths(search_text(project, "handler", glob="src/web/*")) == [("src/web/index.ts", 1)]
    assert len(search_text(project, "handler", max_results=2)) == 2
    assert search_text(project, "no-such-text-anywhere") == []


@needs_rg
def test_ripgrep_and_fallback_agree(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with_rg = search_text(project, r"return|export", glob="**/*.py")
    no_rg(monkeypatch)
    assert search_text(project, r"return|export", glob="**/*.py") == with_rg


@pytest.mark.parametrize("use_rg", [True, False])
def test_long_lines_and_non_utf8_are_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_rg: bool) -> None:
    if use_rg and shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    root = write(tmp_path / "proj", {"long.txt": "needle " + "x" * 1000 + "\n"})
    (root / "latin1.txt").write_bytes("caf\xe9 needle\n".encode("latin-1"))
    if not use_rg:
        no_rg(monkeypatch)
    matches = search_text(root, "needle")
    assert paths(matches) == [("latin1.txt", 1), ("long.txt", 1)]
    assert matches[0].text == "caf\ufffd needle"
    assert len(matches[1].text) == 300


@needs_git
@pytest.mark.parametrize("use_rg", [True, False])
def test_search_respects_gitignore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_rg: bool) -> None:
    if use_rg and shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    root = write(tmp_path / "proj", {".gitignore": "secret_out/\n", "a.py": "token_marker\n", "secret_out/b.py": "token_marker\n"})
    init_git(root)
    if not use_rg:
        no_rg(monkeypatch)
    assert paths(search_text(root, "token_marker")) == [("a.py", 1)]


def test_search_restricted_to_files(project: Path) -> None:
    matches = search_text(project, "handler", files=["src/web/index.ts", "missing.py"])
    assert paths(matches) == [("src/web/index.ts", 1)]


@pytest.mark.parametrize("use_rg", [True, False])
def test_invalid_regex_raises_value_error(project: Path, monkeypatch: pytest.MonkeyPatch, use_rg: bool) -> None:
    if not use_rg:
        no_rg(monkeypatch)
    with pytest.raises(ValueError, match="invalid regular expression"):
        search_text(project, "foo(")
    with pytest.raises(ValueError):
        search_text(project, "")
    # the same text is fine as a literal
    assert search_text(project, "foo(", regex=False) == []


def test_find_files_patterns(project: Path) -> None:
    assert find_files(project, "*.py") == ["src/app.py", "src/util/strings.py"]
    assert find_files(project, "src/**/*.py") == ["src/app.py", "src/util/strings.py"]
    assert find_files(project, "**/index.*") == ["src/web/index.ts"]
    assert find_files(project, "src/*.py") == ["src/app.py"]
    assert find_files(project, "readme.MD") == ["README.md"]
    assert find_files(project, "*.PY") == ["src/app.py", "src/util/strings.py"]
    assert find_files(project, "strings") == ["src/util/strings.py"]
    assert find_files(project, "util/strings.py") == ["src/util/strings.py"]
    assert find_files(project, "*.py", max_results=1) == ["src/app.py"]
    assert find_files(project, "*.js") == []  # node_modules is not listed
    assert find_files(project, "") == []


def test_find_files_ranks_exact_names_first(tmp_path: Path) -> None:
    root = write(tmp_path / "p", {"a/config_loader.py": "", "b/config.py": "", "c/myconfig.yaml": ""})
    assert find_files(root, "config.py") == ["b/config.py"]
    assert find_files(root, "config") == ["a/config_loader.py", "b/config.py", "c/myconfig.yaml"]
    assert find_files(root, "x", files=["x", "dir/x", "xy"]) == ["x", "dir/x", "xy"]


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("src/a.py", "*.py", True),
        ("src/a.py", "src/*.py", True),
        ("src/deep/a.py", "src/*.py", False),
        ("src/deep/a.py", "src/**/*.py", True),
        ("src/a.py", "src/**/*.py", True),
        ("a.py", "**/*.py", True),
        ("src/a.py", "**/src/*.py", True),
        ("src/a.py", "a.?y", True),
        ("src/a.py", "[ab].py", True),
        ("src/c.py", "[!ab].py", True),
        ("src/a.py", "./src/a.py", True),
        ("src/a.py", "*.ts", False),
    ],
)
def test_glob_match(path: str, pattern: str, expected: bool) -> None:
    assert glob_match(path, pattern) is expected
