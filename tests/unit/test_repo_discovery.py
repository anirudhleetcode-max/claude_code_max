from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from ai_engineer.repo.discovery import CommandSuggestion, ProjectProfile, discover, load_profile, save_profile

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

# Built from pieces so the literal never appears in this source file either.
FAKE_AWS_KEY = "AKIA" + "Q7ZK3M9WX2PL5RTB"
FAKE_GH_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
    return root


def init_git(root: Path, commit: bool = True) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("config", "commit.gpgsign", "false")
    if commit:
        git("add", "-A")
        git("commit", "-q", "-m", "init")


PYTHON_PROJECT: dict[str, str] = {
    "pyproject.toml": """\
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "demo"
        version = "0.1.0"
        dependencies = ["fastapi>=0.100", "pydantic>=2"]

        [project.optional-dependencies]
        dev = ["pytest>=8", "ruff", "mypy"]

        [project.scripts]
        demo = "pkg.cli:main"

        [tool.pytest.ini_options]
        testpaths = ["tests"]

        [tool.ruff]
        line-length = 100

        [tool.mypy]
        strict = true
    """,
    "README.md": "# Demo\n\nA demo project about engines.\n",
    "LICENSE": "MIT\n",
    "src/pkg/__init__.py": "",
    "src/pkg/util.py": """\
        def helper(x: int) -> int:
            return x * 2
    """,
    "src/pkg/core.py": """\
        from .util import helper

        MAX_ITEMS = 10


        class Engine:
            def run(self, x: int) -> int:
                return helper(x)


        class EngineFactory:
            def build(self) -> Engine:
                return Engine()


        def engine() -> Engine:
            return Engine()


        def start_engine() -> Engine:
            return Engine()
    """,
    "src/pkg/api.py": """\
        from pkg.core import Engine, start_engine


        def handle_request() -> int:
            return start_engine().run(1)
    """,
    "src/pkg/cli.py": """\
        from pkg import api


        def main() -> None:
            api.handle_request()
    """,
    "src/pkg/__main__.py": """\
        from pkg.cli import main

        main()
    """,
    "tests/__init__.py": "",
    "tests/conftest.py": "",
    "tests/test_core.py": """\
        from pkg.core import Engine


        def test_run() -> None:
            assert Engine().run(2) == 4
    """,
    "tests/test_util.py": """\
        def test_placeholder() -> None:
            assert True
    """,
}

JS_PROJECT: dict[str, str] = {
    "package.json": json.dumps(
        {
            "name": "web",
            "main": "./src/index.ts",
            "scripts": {
                "test": "vitest run",
                "lint": "eslint .",
                "typecheck": "tsc --noEmit",
                "build": "tsc -p .",
                "dev": "vite",
            },
            "dependencies": {"react": "^18.0.0"},
            "devDependencies": {"vitest": "^1.0.0", "vite": "^5.0.0", "typescript": "^5.0.0"},
        },
        indent=2,
    ),
    "pnpm-lock.yaml": "lockfileVersion: '9.0'\n",
    "tsconfig.json": '{"compilerOptions": {"strict": true}}\n',
    "src/index.ts": """\
        import { add } from './math';
        import { Button } from './components/Button';
        export const VERSION = "1";
        export function total(xs: number[]): number {
          return xs.reduce(add, 0);
        }
    """,
    "src/math.ts": """\
        export function add(a: number, b: number): number {
          return a + b;
        }
    """,
    "src/components/Button/index.tsx": """\
        import React from 'react';
        export const Button = () => null;
    """,
    "src/math.test.ts": """\
        import { add } from './math';
        test('adds', () => expect(add(1, 2)).toBe(3));
    """,
}

GO_PROJECT: dict[str, str] = {
    "go.mod": "module example.com/demo\n\ngo 1.22\n\nrequire github.com/gin-gonic/gin v1.9.0\n",
    "main.go": """\
        package main

        import (
        \t"fmt"

        \t"example.com/demo/internal/store"
        )

        func main() {
        \tfmt.Println(store.New().Get("k"))
        }
    """,
    "internal/store/store.go": """\
        package store

        type Store struct{ items map[string]string }

        func New() *Store { return &Store{items: map[string]string{}} }

        func (s *Store) Get(key string) string { return s.items[key] }
    """,
    "internal/store/store_test.go": """\
        package store

        import "testing"

        func TestGet(t *testing.T) {}
    """,
    "cmd/tool/main.go": "package main\n\nfunc main() {}\n",
}


def make_python_project(root: Path) -> Path:
    return write(root, PYTHON_PROJECT)


def make_js_project(root: Path) -> Path:
    return write(root, JS_PROJECT)


def make_go_project(root: Path) -> Path:
    return write(root, GO_PROJECT)


def commands(profile: ProjectProfile, kind: str) -> list[str]:
    return [c.command for c in profile.commands[kind]]


def test_python_project_profile(tmp_path: Path) -> None:
    profile = discover(make_python_project(tmp_path))
    assert profile.root == str(tmp_path.resolve())
    assert profile.is_git is False
    assert profile.git_branch is None
    assert profile.primary_language == "python"
    assert profile.languages["python"]["files"] == 10
    assert profile.file_count == len(PYTHON_PROJECT)
    assert profile.total_bytes > 0
    assert {"fastapi", "pytest"} <= set(profile.frameworks)
    assert profile.package_managers == ["pip"]
    assert commands(profile, "test")[0] == "python -m pytest"
    assert commands(profile, "lint") == ["ruff check ."]
    assert commands(profile, "format") == ["ruff format ."]
    assert commands(profile, "typecheck") == ["mypy ."]
    assert commands(profile, "build") == ["python -m build"]
    assert commands(profile, "install")[0] == 'pip install -e ".[dev]"'
    assert "demo" in commands(profile, "run")
    assert "python -m pkg" in commands(profile, "run")
    assert set(profile.commands) == {"test", "lint", "format", "typecheck", "build", "install", "run"}
    assert profile.best_command("test") == "python -m pytest"
    assert profile.best_command("nonexistent") is None
    assert "pkg.cli:main" in profile.entry_points
    assert "src/pkg/__main__.py" in profile.entry_points
    assert profile.test_dirs == ["tests"]
    assert profile.test_files == ["tests/test_core.py", "tests/test_util.py"]
    assert "pyproject.toml" in profile.config_files
    assert profile.doc_files == ["README.md"]
    assert profile.notable_files == ["LICENSE", "README.md"]
    assert profile.generated_files == []
    assert profile.secret_findings == []
    assert "python project" in profile.summary
    assert "`python -m pytest`" in profile.summary


def test_commands_sorted_by_confidence(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "pyproject.toml": "[tool.pytest.ini_options]\n",
            "Makefile": "test:\n\tpytest\nlint:\n\truff check .\nVAR := 1\n",
            "app.py": "",
        },
    )
    profile = discover(tmp_path)
    suggestions = profile.commands["test"]
    assert [s.command for s in suggestions] == ["python -m pytest", "make test"]
    assert suggestions[0].confidence > suggestions[1].confidence
    assert suggestions[1].source == "Makefile target test"
    assert commands(profile, "lint") == ["make lint"]


def test_unittest_fallback_and_flake8_black(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "setup.py": "from setuptools import setup\nsetup(name='x', install_requires=['requests', 'black'])\n",
            "setup.cfg": "[flake8]\nmax-line-length = 100\n",
            "mypkg/__init__.py": "",
            "tests/test_thing.py": "import unittest\n",
        },
    )
    profile = discover(tmp_path)
    assert commands(profile, "test") == ["python -m unittest discover"]
    assert commands(profile, "lint") == ["flake8"]
    assert commands(profile, "format") == ["black ."]
    assert commands(profile, "typecheck") == []
    assert commands(profile, "install") == ["pip install -e ."]
    assert commands(profile, "build") == []  # no [build-system]
    assert "pytest" not in profile.frameworks


def test_no_evidence_means_no_commands(tmp_path: Path) -> None:
    write(tmp_path, {"notes.txt": "hello\n", "script.py": "print('hi')\n"})
    profile = discover(tmp_path)
    assert all(not v for v in profile.commands.values())
    assert profile.frameworks == []
    assert "No test/lint/build commands were detected." in profile.summary


def test_poetry_and_uv_installers(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "pyproject.toml": '[tool.poetry]\nname = "x"\n[tool.poetry.dependencies]\ndjango = "^5"\n',
            "poetry.lock": "",
            "manage.py": "",
        },
    )
    profile = discover(tmp_path)
    assert commands(profile, "install")[0] == "poetry install"
    assert "poetry" in profile.package_managers
    assert "django" in profile.frameworks
    assert "python manage.py runserver" in commands(profile, "run")
    assert "manage.py" in profile.entry_points
    assert "poetry.lock" in profile.generated_files

    other = tmp_path / "uvproj"
    write(other, {"pyproject.toml": '[project]\nname = "y"\ndependencies = ["flask"]\n', "uv.lock": ""})
    profile = discover(other)
    assert commands(profile, "install")[0] == "uv sync"
    assert profile.package_managers == ["uv"]
    assert profile.frameworks == ["flask"]


def test_js_project_profile(tmp_path: Path) -> None:
    profile = discover(make_js_project(tmp_path))
    assert profile.primary_language == "typescript"
    assert profile.package_managers == ["pnpm"]
    assert profile.frameworks == ["react", "vite", "vitest"]
    assert commands(profile, "test") == ["pnpm test"]
    assert profile.commands["test"][0].source == "package.json scripts.test"
    assert commands(profile, "lint") == ["pnpm lint"]
    assert commands(profile, "typecheck") == ["pnpm typecheck"]
    assert commands(profile, "build") == ["pnpm build"]
    assert commands(profile, "run") == ["pnpm dev"]
    assert commands(profile, "install") == ["pnpm install"]
    assert profile.entry_points == ["src/index.ts"]
    assert profile.test_files == ["src/math.test.ts"]
    assert "pnpm-lock.yaml" in profile.generated_files
    assert "tsconfig.json" in profile.config_files


def test_npm_default_test_script_is_ignored(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "package.json": json.dumps(
                {"scripts": {"test": 'echo "Error: no test specified" && exit 1', "start": "node index.js"}}
            ),
            "package-lock.json": "{}",
            "tsconfig.json": "{}",
            "index.js": "",
        },
    )
    profile = discover(tmp_path)
    assert commands(profile, "test") == []
    assert commands(profile, "run") == ["npm start"]
    assert commands(profile, "install") == ["npm install"]
    tsc = profile.commands["typecheck"]
    assert [c.command for c in tsc] == ["npx tsc --noEmit"]
    assert tsc[0].confidence == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("lockfile", "expected"),
    [("yarn.lock", "yarn test"), ("bun.lockb", "bun run test"), ("package-lock.json", "npm test")],
)
def test_js_package_manager_prefix(tmp_path: Path, lockfile: str, expected: str) -> None:
    write(tmp_path, {"package.json": json.dumps({"scripts": {"test": "jest", "lint": "eslint ."}}), lockfile: ""})
    profile = discover(tmp_path)
    assert commands(profile, "test") == [expected]
    lint = commands(profile, "lint")[0]
    assert lint == {"yarn": "yarn lint", "bun": "bun run lint", "npm": "npm run lint"}[expected.split()[0]]


def test_go_project_profile(tmp_path: Path) -> None:
    profile = discover(make_go_project(tmp_path))
    assert profile.primary_language == "go"
    assert profile.package_managers == ["go"]
    assert profile.frameworks == ["gin"]
    assert commands(profile, "test") == ["go test ./..."]
    assert commands(profile, "build") == ["go build ./..."]
    assert commands(profile, "lint") == ["go vet ./..."]
    assert commands(profile, "format") == ["gofmt -l ."]
    assert profile.entry_points == ["cmd/tool/main.go", "main.go"]
    assert profile.test_files == ["internal/store/store_test.go"]


def test_rust_and_java_projects(tmp_path: Path) -> None:
    rust = write(
        tmp_path / "rs",
        {"Cargo.toml": '[package]\nname = "x"\n[dependencies]\naxum = "0.7"\n', "src/main.rs": "fn main() {}\n"},
    )
    profile = discover(rust)
    assert commands(profile, "test") == ["cargo test"]
    assert commands(profile, "build") == ["cargo build"]
    assert commands(profile, "lint") == ["cargo clippy"]
    assert commands(profile, "format") == ["cargo fmt --check"]
    assert profile.frameworks == ["axum"]
    assert "src/main.rs" in profile.entry_points

    maven = write(
        tmp_path / "mvn",
        {"pom.xml": "<project><groupId>org.springframework.boot</groupId></project>", "src/main/java/A.java": ""},
    )
    profile = discover(maven)
    assert commands(profile, "test") == ["mvn -q test"]
    assert commands(profile, "build") == ["mvn -q package"]
    assert profile.frameworks == ["spring"]
    assert profile.package_managers == ["maven"]

    gradle = write(tmp_path / "gradle", {"build.gradle.kts": "plugins {}\n", "gradlew": "#!/bin/sh\n"})
    assert commands(discover(gradle), "test") == ["./gradlew test"]
    plain_gradle = write(tmp_path / "gradle2", {"build.gradle": "plugins {}\n"})
    assert commands(discover(plain_gradle), "test") == ["gradle test"]


def test_malformed_manifests_are_tolerated(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "pyproject.toml": "[project\nname = ",
            "package.json": "{not json",
            "Cargo.toml": "[[[",
            "main.py": "",
        },
    )
    profile = discover(tmp_path)
    assert profile.file_count == 4
    assert commands(profile, "test") == ["cargo test"]  # Cargo.toml presence is still evidence


def test_secret_findings_report_location_only(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "app/settings.py": f'DEBUG = True\nAWS_ACCESS_KEY_ID = "{FAKE_AWS_KEY}"\n',
            ".env": f"GITHUB_TOKEN={FAKE_GH_TOKEN}\n",
            "package-lock.json": json.dumps({"token": FAKE_GH_TOKEN}),
            "clean.py": "x = 1\n",
        },
    )
    profile = discover(tmp_path)
    assert {"path": "app/settings.py", "line": 2, "kind": "aws_access_key_id"} in profile.secret_findings
    assert {"path": ".env", "line": 1, "kind": "github_token"} in profile.secret_findings
    # lockfiles are generated and skipped
    assert all(f["path"] != "package-lock.json" for f in profile.secret_findings)
    for finding in profile.secret_findings:
        assert set(finding) == {"path", "line", "kind"}
    dumped = profile.model_dump_json()
    assert FAKE_AWS_KEY not in dumped
    assert FAKE_GH_TOKEN not in dumped
    assert FAKE_AWS_KEY[4:12] not in dumped
    assert "Potential secrets detected at" in profile.summary


def test_generated_files_detected_by_path_and_content(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "static/app.min.js": "var a=1;",
            "api/user.pb.go": "package api\n",
            "gen/models.py": "# Code generated by sqlc. DO NOT EDIT.\nclass User: ...\n",
            "src/real.py": "x = 1\n",
        },
    )
    profile = discover(tmp_path)
    assert profile.generated_files == ["api/user.pb.go", "gen/models.py", "static/app.min.js"]
    # generated files do not count toward language statistics
    assert profile.languages["python"]["files"] == 2
    assert "go" not in profile.languages


@needs_git
def test_git_branch_detection(tmp_path: Path) -> None:
    make_python_project(tmp_path)
    init_git(tmp_path)
    profile = discover(tmp_path)
    assert profile.is_git is True
    assert profile.git_branch == "main"
    assert "branch main" in profile.summary


def test_discover_accepts_explicit_file_list(tmp_path: Path) -> None:
    make_python_project(tmp_path)
    profile = discover(tmp_path, files=["src/pkg/core.py", "pyproject.toml"])
    assert profile.file_count == 2
    assert profile.test_files == []


def test_discover_is_deterministic(tmp_path: Path) -> None:
    make_python_project(tmp_path)
    make_js_project(tmp_path / "web")
    first = discover(tmp_path)
    second = discover(tmp_path)
    assert first.model_dump() == second.model_dump()
    assert first.summary == second.summary


def test_save_and_load_profile_roundtrip(tmp_path: Path) -> None:
    profile = discover(make_python_project(tmp_path / "proj"))
    target = tmp_path / ".agent" / "project.json"
    save_profile(profile, target)
    assert json.loads(target.read_text())["primary_language"] == "python"
    loaded = load_profile(target)
    assert loaded == profile
    assert isinstance(loaded.commands["test"][0], CommandSuggestion)


def test_load_profile_missing_or_corrupt(tmp_path: Path) -> None:
    assert load_profile(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert load_profile(bad) is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"root": 1}')
    assert load_profile(wrong) is None
