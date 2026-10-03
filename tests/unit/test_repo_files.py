from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_engineer.repo.files import (
    DEFAULT_IGNORE_DIRS,
    find_test_dir,
    is_generated_content,
    is_generated_path,
    is_git_repo,
    is_lockfile,
    is_test_path,
    language_of,
    list_files,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def write(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def init_git(root: Path) -> None:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    git(root, "config", "commit.gpgsign", "false")


def test_default_ignore_dirs_cover_common_noise() -> None:
    for name in (".git", ".agent", "node_modules", ".venv", "__pycache__", "dist", "build", "target", "vendor"):
        assert name in DEFAULT_IGNORE_DIRS


def test_list_files_non_git_skips_ignored_and_hidden_dirs(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "app.py": "print('hi')\n",
            "pkg/mod.py": "x = 1\n",
            "node_modules/lib/index.js": "module.exports = 1\n",
            ".venv/lib/site.py": "",
            "pkg/__pycache__/mod.cpython-311.pyc": "",
            ".hidden/secret.txt": "x",
            ".github/workflows/ci.yml": "on: push\n",
            ".agent/project.json": "{}",
            ".env.example": "API_URL=\n",
        },
    )
    assert not is_git_repo(tmp_path)
    files = list_files(tmp_path)
    assert files == sorted(files)
    assert files == [".env.example", ".github/workflows/ci.yml", "app.py", "pkg/mod.py"]


def test_list_files_respects_max_files(tmp_path: Path) -> None:
    write(tmp_path, {f"f{i:02d}.txt": "x" for i in range(10)})
    assert list_files(tmp_path, max_files=3) == ["f00.txt", "f01.txt", "f02.txt"]


def test_list_files_returns_posix_paths(tmp_path: Path) -> None:
    write(tmp_path, {"a/b/c.txt": "x"})
    assert list_files(tmp_path) == ["a/b/c.txt"]


@needs_git
def test_list_files_git_repo_uses_gitignore(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            ".gitignore": "ignored.txt\nout/\n",
            "src/main.py": "print(1)\n",
            "ignored.txt": "nope\n",
            "out/artifact.txt": "nope\n",
            "node_modules/dep/index.js": "x\n",
            "doomed.py": "x = 1\n",
            ".agent/state.json": "{}",
        },
    )
    init_git(tmp_path)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    # untracked-but-not-ignored files are included; tracked-but-deleted ones are not
    write(tmp_path, {"src/new_module.py": "y = 2\n"})
    (tmp_path / "doomed.py").unlink()

    assert is_git_repo(tmp_path)
    files = list_files(tmp_path)
    assert files == [".gitignore", "src/main.py", "src/new_module.py"]


@needs_git
def test_list_files_include_ignored(tmp_path: Path) -> None:
    write(tmp_path, {".gitignore": "ignored.txt\n", "a.py": "", "ignored.txt": "x"})
    init_git(tmp_path)
    files = list_files(tmp_path, include_ignored=True)
    assert "ignored.txt" in files
    assert not any(f.startswith(".git/") for f in files)


@needs_git
def test_list_files_in_git_subdirectory_is_relative_to_root(tmp_path: Path) -> None:
    write(tmp_path, {"sub/a.py": "", "other/b.py": ""})
    init_git(tmp_path)
    assert list_files(tmp_path / "sub") == ["a.py"]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_list_files_skips_symlinks_escaping_workspace(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    root = tmp_path / "ws"
    write(root, {"real.txt": "x"})
    (root / "inside_link.txt").symlink_to(root / "real.txt")
    (root / "escape_link.txt").symlink_to(outside)
    assert list_files(root) == ["inside_link.txt", "real.txt"]


def test_list_files_missing_root(tmp_path: Path) -> None:
    assert list_files(tmp_path / "missing") == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/app.py", "python"),
        ("types.pyi", "python"),
        ("web/index.js", "javascript"),
        ("web/App.jsx", "javascript"),
        ("lib/util.mjs", "javascript"),
        ("src/main.ts", "typescript"),
        ("src/App.tsx", "typescript"),
        ("types/global.d.ts", "typescript"),
        ("main.go", "go"),
        ("src/lib.rs", "rust"),
        ("A.java", "java"),
        ("build.gradle.kts", "kotlin"),
        ("Program.cs", "csharp"),
        ("x.c", "c"),
        ("x.hpp", "cpp"),
        ("x.rb", "ruby"),
        ("index.php", "php"),
        ("App.swift", "swift"),
        ("x.scala", "scala"),
        ("run.sh", "shell"),
        ("setup.ps1", "powershell"),
        ("schema.sql", "sql"),
        ("index.html", "html"),
        ("style.css", "css"),
        ("README.md", "markdown"),
        ("ci.yaml", "yaml"),
        ("pyproject.toml", "toml"),
        ("package.json", "json"),
        ("Dockerfile", "dockerfile"),
        ("docker/Dockerfile.dev", "dockerfile"),
        ("Makefile", "makefile"),
        ("Gemfile", "ruby"),
        ("C:\\proj\\src\\app.py", "python"),
        ("LICENSE", None),
        ("data.bin", None),
        ("noext", None),
    ],
)
def test_language_of(path: str, expected: str | None) -> None:
    assert language_of(path) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("package-lock.json", True),
        ("web/yarn.lock", True),
        ("pnpm-lock.yaml", True),
        ("poetry.lock", True),
        ("Cargo.lock", True),
        ("go.sum", True),
        ("static/app.min.js", True),
        ("static/app.min.css", True),
        ("proto/user_pb2.py", True),
        ("proto/user_pb2_grpc.py", True),
        ("api/user.pb.go", True),
        ("src/schema.generated.ts", True),
        ("dist/bundle.js", True),
        ("pkg/build/out.js", True),
        ("static/app.js.map", True),
        ("src/app.py", False),
        ("src/builder.py", False),
        ("docs/distribution.md", False),
    ],
)
def test_is_generated_path(path: str, expected: bool) -> None:
    assert is_generated_path(path) is expected


def test_is_lockfile() -> None:
    assert is_lockfile("a/b/poetry.lock")
    assert not is_lockfile("pyproject.toml")


def test_is_generated_content() -> None:
    assert is_generated_content("// Code generated by protoc-gen-go. DO NOT EDIT.\npackage x\n")
    assert is_generated_content("# @generated by tool\n")
    assert is_generated_content('"""This file is AUTOGENERATED."""\n')
    assert is_generated_content("/* auto-generated file */")
    assert not is_generated_content("def handler():\n    return 1\n")
    # markers past the first ~2 KB are ignored
    assert not is_generated_content("x = 1\n" * 500 + "# DO NOT EDIT\n")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_core.py", True),
        ("pkg/core_test.py", True),
        ("tests/conftest.py", False),
        ("tests/helpers.py", False),
        ("src/math.test.ts", True),
        ("src/math.spec.js", True),
        ("src/__tests__/util.js", True),
        ("src/math.ts", False),
        ("internal/store/store_test.go", True),
        ("internal/store/store.go", False),
        ("src/test/java/com/x/UserServiceTest.java", True),
        ("src/main/java/com/x/UserService.java", False),
        ("spec/models/user_spec.rb", True),
        ("tests/integration.rs", True),
        ("README.md", False),
    ],
)
def test_is_test_path(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected


def test_find_test_dir() -> None:
    assert find_test_dir("tests/unit/test_a.py") == "tests"
    assert find_test_dir("web/__tests__/a.test.js") == "web/__tests__"
    assert find_test_dir("src/test/java/A.java") == "src/test"
    assert find_test_dir("src/pkg/a.py") is None
