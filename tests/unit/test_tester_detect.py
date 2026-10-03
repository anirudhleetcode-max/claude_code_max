from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_engineer.config.settings import ValidationSettings
from ai_engineer.tester.detect import (
    executable_of,
    format_check_command,
    format_write_command,
    plan_checks,
    split_command,
    targeted_lint_command,
    targeted_test_command,
    write_format_command,
)
from ai_engineer.tester.models import CheckKind, ValidationCommand


def sugg(command: str, source: str = "manifest", confidence: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(command=command, source=source, confidence=confidence)


def profile(**commands: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        commands=commands, frameworks=[], package_managers=[], primary_language=None, test_files=[], root=None
    )


def which_all(name: str) -> str | None:
    return f"/usr/bin/{name}"


def which_only(*names: str):
    return lambda name: f"/usr/bin/{name}" if name in names else None


# --------------------------------------------------------------------------------------- planning


def test_plan_uses_best_suggestions_and_timeouts() -> None:
    p = profile(
        test=[sugg("python -m unittest discover", "test*.py files", 0.6), sugg("python -m pytest", "pyproject.toml", 0.9)],
        lint=[sugg("ruff check .", "ruff config", 0.9)],
        typecheck=[sugg("mypy .", "mypy dependency", 0.6)],
        build=[sugg("python -m build", "pyproject.toml [build-system]", 0.7)],
    )
    settings = ValidationSettings(test_timeout_s=900)
    plan = plan_checks(p, settings, which_all)
    test = plan[CheckKind.TEST]
    assert test is not None
    assert (test.command, test.source, test.confidence, test.timeout_s) == ("python -m pytest", "pyproject.toml", 0.9, 900)
    assert test.available and test.scope == "full" and test.cwd == "."
    assert plan[CheckKind.LINT].timeout_s == 300  # type: ignore[union-attr]
    assert plan[CheckKind.TYPECHECK].timeout_s == 600  # type: ignore[union-attr]
    assert plan[CheckKind.BUILD].timeout_s == 600  # type: ignore[union-attr]
    assert plan[CheckKind.FORMAT] is None
    assert set(plan) == set(CheckKind)


def test_config_overrides_win() -> None:
    p = profile(test=[sugg("python -m pytest")], lint=[sugg("ruff check .")])
    settings = ValidationSettings(test_command="make test", lint_command="", format_command="black src")
    plan = plan_checks(p, settings, which_all)
    assert plan[CheckKind.TEST].command == "make test"  # type: ignore[union-attr]
    assert plan[CheckKind.TEST].source == "config"  # type: ignore[union-attr]
    assert plan[CheckKind.LINT] is None  # an empty override disables the check
    fmt = plan[CheckKind.FORMAT]
    assert fmt is not None and fmt.command == "black --check src" and fmt.source == "config"


def test_none_profile_and_dict_shaped_suggestions() -> None:
    plan = plan_checks(None, ValidationSettings(), which_all)
    assert all(v is None for v in plan.values())
    p = SimpleNamespace(commands={"test": [{"command": "npm test", "source": "package.json", "confidence": 0.95}], "lint": ["npm run lint"]})
    plan2 = plan_checks(p, ValidationSettings(), which_all)
    assert plan2[CheckKind.TEST].command == "npm test"  # type: ignore[union-attr]
    assert plan2[CheckKind.LINT].source == "profile"  # type: ignore[union-attr]


def test_real_project_profile(tmp_path: Path) -> None:
    from ai_engineer.repo.discovery import CommandSuggestion, ProjectProfile

    prof = ProjectProfile(
        root=str(tmp_path), is_git=False, primary_language="python", package_managers=["pip"],
        commands={"test": [CommandSuggestion(command="python -m pytest", source="pyproject.toml", confidence=0.9)],
                  "format": [CommandSuggestion(command="ruff format .", source="ruff", confidence=0.85)]},
    )
    plan = plan_checks(prof, ValidationSettings(), which_all)
    assert plan[CheckKind.TEST].command == "python -m pytest"  # type: ignore[union-attr]
    assert plan[CheckKind.FORMAT].command == "ruff format --check ."  # type: ignore[union-attr]
    assert plan[CheckKind.AUDIT].command == "pip-audit"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------------- availability


def test_unavailable_executable_is_kept_but_flagged() -> None:
    plan = plan_checks(profile(lint=[sugg("ruff check .")]), ValidationSettings(), which_only("python"))
    lint = plan[CheckKind.LINT]
    assert lint is not None and lint.command == "ruff check ."
    assert lint.available is False and lint.unavailable_reason == "ruff not found on PATH"


@pytest.mark.parametrize(
    "command, needed",
    [
        ("python -m pytest", "python"),
        ("npx eslint .", "npx"),
        ("npm run lint", "npm"),
        ("pnpm test", "pnpm"),
        ("CI=1 FOO=bar yarn test", "yarn"),
        ("env DJANGO_SETTINGS_MODULE=x python manage.py test", "python"),
        ("cd web && npm test", "npm"),
    ],
)
def test_availability_checks_the_right_executable(command: str, needed: str) -> None:
    asked: list[str] = []

    def which(name: str) -> str | None:
        asked.append(name)
        return "/bin/x" if name == needed else None

    plan = plan_checks(profile(test=[sugg(command)]), ValidationSettings(), which)
    test = plan[CheckKind.TEST]
    assert test is not None and test.available, (command, asked)
    assert asked[0] == needed


def test_python_falls_back_to_python3() -> None:
    plan = plan_checks(profile(test=[sugg("python -m pytest -q")]), ValidationSettings(), which_only("python3"))
    test = plan[CheckKind.TEST]
    assert test is not None and test.available and test.command == "python3 -m pytest -q"


def test_path_executables_checked_relative_to_root(tmp_path: Path) -> None:
    p = profile(test=[sugg("./gradlew test")], build=[sugg("./mvnw -q package")])
    p.root = str(tmp_path)
    (tmp_path / "gradlew").write_text("#!/bin/sh\n")
    plan = plan_checks(p, ValidationSettings(), lambda _: None)
    assert plan[CheckKind.TEST].available  # type: ignore[union-attr]
    build = plan[CheckKind.BUILD]
    assert build is not None and not build.available and "./mvnw not found" in build.unavailable_reason


# ----------------------------------------------------------------------------------------- audit


def test_audit_python_requires_pip_audit() -> None:
    p = profile()
    p.primary_language = "python"
    assert plan_checks(p, ValidationSettings(), which_only("python"))[CheckKind.AUDIT] is None
    audit = plan_checks(p, ValidationSettings(), which_only("pip-audit"))[CheckKind.AUDIT]
    assert audit is not None and audit.command == "pip-audit" and audit.kind == CheckKind.AUDIT
    assert plan_checks(p, ValidationSettings(dependency_audit=False), which_all)[CheckKind.AUDIT] is None


def test_audit_js_projects() -> None:
    p = profile()
    p.primary_language, p.package_managers = "typescript", ["npm"]
    audit = plan_checks(p, ValidationSettings(), which_all)[CheckKind.AUDIT]
    assert audit is not None and audit.command == "npm audit --audit-level=high"
    p.package_managers = ["pnpm"]
    pn = plan_checks(p, ValidationSettings(), lambda _: None)[CheckKind.AUDIT]
    assert pn is not None and pn.command.startswith("pnpm audit") and not pn.available
    p.package_managers, p.primary_language = ["cargo"], "rust"
    assert plan_checks(p, ValidationSettings(), which_all)[CheckKind.AUDIT] is None


# --------------------------------------------------------------------------------------- formatters


@pytest.mark.parametrize(
    "command, expected",
    [
        ("ruff format .", "ruff format --check ."),
        ("ruff format --check .", "ruff format --check ."),
        ("python -m ruff format src", "python -m ruff format --check src"),
        ("black .", "black --check ."),
        ("poetry run black src tests", "poetry run black --check src tests"),
        ("prettier --write src", "prettier --check src"),
        ("npx prettier --write .", "npx prettier --check ."),
        ("npx prettier .", "npx prettier --check ."),
        ("gofmt -l .", "gofmt -l ."),
        ("gofmt -w .", "gofmt -l ."),
        ("cargo fmt", "cargo fmt --check"),
        ("cargo fmt --check", "cargo fmt --check"),
        ("isort .", "isort --check-only ."),
        ("npm run format", None),
        ("pnpm format:check", "pnpm format:check"),
        ("make format", None),
    ],
)
def test_format_check_conversion(command: str, expected: str | None) -> None:
    assert format_check_command(command) == expected


def test_npm_format_uses_check_script_when_suggested() -> None:
    p = profile(format=[sugg("npm run format", confidence=0.9), sugg("npm run format:check", confidence=0.5)])
    fmt = plan_checks(p, ValidationSettings(), which_all)[CheckKind.FORMAT]
    assert fmt is not None and fmt.command == "npm run format:check"
    only_write = profile(format=[sugg("npm run format")])
    assert plan_checks(only_write, ValidationSettings(), which_all)[CheckKind.FORMAT] is None


@pytest.mark.parametrize(
    "command, expected",
    [
        ("ruff format --check .", "ruff format ."),
        ("black --check src", "black src"),
        ("npx prettier --check .", "npx prettier --write ."),
        ("gofmt -l .", "gofmt -w ."),
        ("cargo fmt --check", "cargo fmt"),
        ("npm run format", "npm run format"),
        ("npm run format:check", None),
    ],
)
def test_format_write_conversion(command: str, expected: str | None) -> None:
    assert format_write_command(command) == expected


def test_write_format_command_targets_files() -> None:
    p = profile(format=[sugg("ruff format .")])
    s = ValidationSettings()
    assert write_format_command(p, s) == "ruff format ."
    assert write_format_command(p, s, ["src/a.py", "README.md"]) == "ruff format src/a.py"
    assert write_format_command(p, s, ["README.md"]) is None
    assert write_format_command(profile(format=[sugg("cargo fmt --check")]), s, ["src/lib.rs"]) is None
    assert write_format_command(profile(), ValidationSettings(format_command="black --check .")) == "black ."
    assert write_format_command(None, s) is None


# -------------------------------------------------------------------------------------- targeting


def vc(command: str, kind: CheckKind = CheckKind.TEST) -> ValidationCommand:
    return ValidationCommand(kind=kind, command=command, source="test", timeout_s=123)


@pytest.mark.parametrize(
    "base, expected",
    [
        ("python -m pytest", "python -m pytest tests/test_a.py tests/test_b.py"),
        ("python -m pytest -q tests", "python -m pytest -q tests/test_a.py tests/test_b.py"),
        ("pytest -x --tb=short -k 'not slow' tests/", "pytest -x --tb=short -k 'not slow' tests/test_a.py tests/test_b.py"),
        ("uv run pytest -p no:cacheprovider --maxfail 2", "uv run pytest -p no:cacheprovider --maxfail 2 tests/test_a.py tests/test_b.py"),
        ("python -m unittest discover -s tests -p 'test_*.py'", "python -m unittest tests.test_a tests.test_b"),
        ("python -m unittest -v", "python -m unittest -v tests.test_a tests.test_b"),
    ],
)
def test_targeted_python_tests(base: str, expected: str) -> None:
    t = targeted_test_command(vc(base), ["tests/test_a.py", "tests/test_b.py"])
    assert t is not None
    assert t.command == expected
    assert t.scope == "targeted" and t.timeout_s == 123 and t.kind == CheckKind.TEST


def test_targeted_quotes_paths_and_handles_windows_separators() -> None:
    t = targeted_test_command(vc("pytest"), ["tests/my test.py", "tests\\win_test.py"])
    assert t is not None and t.command == "pytest 'tests/my test.py' tests/win_test.py"


def test_targeted_django() -> None:
    t = targeted_test_command(vc("python manage.py test"), ["app/tests/test_views.py"])
    assert t is not None and t.command == "python manage.py test app.tests.test_views"


@pytest.mark.parametrize(
    "base, frameworks, expected",
    [
        ("npx jest", [], "npx jest src/a.test.ts"),
        ("npx vitest run", [], "npx vitest run src/a.test.ts"),
        ("npm test", ["jest"], "npm test -- src/a.test.ts"),
        ("npm run test", ["vitest", "vite"], "npm run test -- src/a.test.ts"),
        ("pnpm test", ["vitest"], "pnpm test src/a.test.ts"),
        ("yarn test", ["jest"], "yarn test src/a.test.ts"),
        ("npm test", ["mocha"], None),
        ("npm test", [], None),
    ],
)
def test_targeted_js_tests(base: str, frameworks: list[str], expected: str | None) -> None:
    t = targeted_test_command(vc(base), ["src/a.test.ts"], frameworks)
    assert (t.command if t else None) == expected


def test_targeted_go_and_unsupported() -> None:
    t = targeted_test_command(vc("go test -race ./..."), ["pkg/a/a_test.go", "pkg/a/b_test.go", "main_test.go"])
    assert t is not None and t.command == "go test -race ./pkg/a ."
    assert targeted_test_command(vc("cargo test"), ["tests/it.rs"]) is None
    assert targeted_test_command(vc("mvn -q test"), ["src/test/java/AppTest.java"]) is None
    assert targeted_test_command(vc("./gradlew test"), ["x"]) is None
    assert targeted_test_command(vc("make test"), ["tests/test_a.py"]) is None
    assert targeted_test_command(vc("tox"), ["tests/test_a.py"]) is None
    assert targeted_test_command(vc("cd api && pytest"), ["tests/test_a.py"]) is None
    assert targeted_test_command(vc("pytest"), []) is None


@pytest.mark.parametrize(
    "base, files, expected",
    [
        ("ruff check .", ["src/a.py", "web/x.ts"], "ruff check src/a.py"),
        ("ruff check --select E,F src tests", ["src/a.py"], "ruff check --select E,F src/a.py"),
        ("flake8 --max-line-length=100", ["a.py"], "flake8 --max-line-length=100 a.py"),
        ("pylint src", ["src/a.py"], "pylint src/a.py"),
        ("mypy .", ["src/a.py", "src/b.pyi"], "mypy src/a.py src/b.pyi"),
        ("python -m mypy --config-file mypy.ini src", ["src/a.py"], "python -m mypy --config-file mypy.ini src/a.py"),
        ("npx eslint .", ["src/a.tsx", "README.md"], "npx eslint src/a.tsx"),
        ("npx eslint --ext .ts,.tsx src", ["src/a.ts"], "npx eslint --ext .ts,.tsx src/a.ts"),
        ("ruff format --check .", ["src/a.py"], "ruff format --check src/a.py"),
        ("npx prettier --check .", ["src/a.css"], "npx prettier --check src/a.css"),
        ("ruff check .", ["README.md"], None),
        ("npm run lint", ["src/a.ts"], None),
        ("go vet ./...", ["main.go"], None),
        ("cargo clippy", ["src/lib.rs"], None),
        ("npx tsc --noEmit", ["src/a.ts"], None),
    ],
)
def test_targeted_lint(base: str, files: list[str], expected: str | None) -> None:
    t = targeted_lint_command(vc(base, CheckKind.LINT), files)
    assert (t.command if t else None) == expected
    if t is not None:
        assert t.scope == "targeted" and t.kind == CheckKind.LINT


# ------------------------------------------------------------------------------------- tokenising


def test_tokenising_never_crashes() -> None:
    assert split_command('pytest -k "unterminated') == ["pytest", "-k", '"unterminated']
    win = split_command(r"C:\venv\Scripts\python.exe -m pytest tests\unit")
    assert win[0] == r"C:\venv\Scripts\python.exe" and win[-1] == r"tests\unit"
    assert executable_of(r'"C:\Program Files\nodejs\npm.cmd" test') == r"C:\Program Files\nodejs\npm.cmd"
    assert executable_of("FOO=1 BAR=2") is None
    assert format_check_command(r"C:\venv\Scripts\black.exe --check .") is not None
    assert targeted_test_command(vc(r"C:\venv\Scripts\python.exe -m pytest"), ["tests/test_a.py"]) is None
