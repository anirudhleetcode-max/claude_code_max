from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_engineer.core.errors import PathViolation
from ai_engineer.security.audit import AuditLog
from ai_engineer.security.paths import PathGuard, glob_match
from ai_engineer.security.secrets import REDACTED, Redactor, is_secret_env_name, scan_text, secret_env_values
from ai_engineer.security.static_rules import added_lines_by_file, scan_diff

# Assemble fake credentials at runtime so this file itself does not trip secret scanners.
FAKE_GH = "ghp_" + "Ab3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3fG5"
FAKE_AWS = "AKIA" + "IOSFODNN7EXAMPLQ"


def test_scan_text_reports_location_without_value() -> None:
    text = f"line1\ntoken = '{FAKE_GH}'\n"
    findings = scan_text(text)
    assert len(findings) == 1
    f = findings[0]
    assert f.line == 2 and f.kind == "github_token"
    assert FAKE_GH not in f.preview


@pytest.mark.parametrize(
    "text",
    [
        'password = os.environ["DB_PASSWORD"]',
        'API_KEY = "your-api-key"',
        "secret: ${SECRET_VALUE}",
        "def get_token(self): return self.tokens.lookup(name)",
        "token_count = 4096",
        "api_key = settings.api_key",
    ],
)
def test_scan_text_ignores_placeholders_and_code(text: str) -> None:
    assert scan_text(text) == []


def test_redactor_handles_env_values_patterns_and_nesting() -> None:
    r = Redactor(environ={"SERVICE_TOKEN": "tok-ABCDEFGH1234", "GIT_AUTHOR_EMAIL": "dev@example.com", "HOME": "/home/u"})
    out = r.redact({"a": ["x tok-ABCDEFGH1234 y", f"key {FAKE_AWS}"], "b": "dev@example.com", "n": 3})
    assert out["a"][0] == f"x {REDACTED} y"
    assert FAKE_AWS not in out["a"][1]
    assert out["b"] == "dev@example.com"  # author emails are not secrets
    assert out["n"] == 3


def test_secret_env_name_heuristics() -> None:
    assert is_secret_env_name("OPENAI_API_KEY")
    assert is_secret_env_name("DB_PASSWORD")
    assert is_secret_env_name("GITHUB_TOKEN")
    assert not is_secret_env_name("PATH")
    assert not is_secret_env_name("GIT_AUTHOR_NAME")
    assert not is_secret_env_name("MAX_THINKING_TOKENS")
    assert not is_secret_env_name("SSH_KEY_PATH")
    assert not is_secret_env_name("PWD") and not is_secret_env_name("OLDPWD")
    assert is_secret_env_name("DB_PWD")
    assert secret_env_values({"API_TOKEN_LOCATION": "/home/user/project"}) == set()
    assert secret_env_values({"X_TOKEN": "short"}) == set()


@pytest.mark.parametrize(
    ("rel", "pattern", "expected"),
    [
        (".env", ".env", True),
        ("config/.env", ".env", True),
        (".env.production", ".env.*", True),
        (".git/HEAD", ".git/**", True),
        ("src/.git_helpers.py", ".git/**", False),
        ("certs/server.pem", "**/*.pem", True),
        ("server.pem", "**/*.pem", True),
        ("src/app.py", "*.pem", False),
    ],
)
def test_glob_match(rel: str, pattern: str, expected: bool) -> None:
    assert glob_match(rel, pattern) is expected


def test_path_guard_blocks_escape_and_protected(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "src").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    guard = PathGuard(ws, protected=[".git/**", ".env"], secret_files=[".env"])
    assert guard.resolve("src/a.py", for_write=True) == (ws / "src" / "a.py").resolve()
    with pytest.raises(PathViolation):
        guard.resolve("../outside.txt")
    with pytest.raises(PathViolation):
        guard.resolve(str(outside))
    with pytest.raises(PathViolation):
        guard.resolve("~/.ssh/id_rsa")
    with pytest.raises(PathViolation):
        guard.resolve(".git/config", for_write=True)
    with pytest.raises(PathViolation):
        guard.resolve(".env", for_write=True)
    assert guard.resolve(".env").name == ".env"  # reading is allowed (content is redacted by tools)
    assert guard.is_secret_file(".env")


def test_default_protected_paths_cover_nested_git_and_ignore_case(tmp_path: Path) -> None:
    # Audit regression: only the top-level .git was protected (a submodule's hooks were writable),
    # and matching was case-sensitive although ".GIT/hooks" or ".ENV" are the same files on macOS/Windows.
    from ai_engineer.config.settings import PermissionsSettings

    ws = tmp_path / "ws"
    (ws / "vendor" / "lib" / ".git" / "hooks").mkdir(parents=True)
    perms = PermissionsSettings()
    guard = PathGuard(ws, protected=perms.protected_paths, secret_files=perms.secret_files)
    for rel in [".git/hooks/pre-commit", "vendor/lib/.git/hooks/pre-commit", "vendor/lib/.git", ".GIT/hooks/x",
                ".ENV", ".Env.Local", "certs/Server.PEM", ".Agent/config.toml", "home/.SSH/config"]:
        with pytest.raises(PathViolation):
            guard.resolve(rel, for_write=True)
    for rel in [".gitignore", ".github/workflows/ci.yml", "src/.git_helpers.py", "src/app.py"]:
        guard.resolve(rel, for_write=True)
    assert guard.is_secret_file(".ENV") and guard.is_secret_file("conf/Credentials.json")
    assert not glob_match(".ENV", ".env")  # plain glob_match stays case-sensitive (used for search)
    assert glob_match(".ENV", ".env", ignore_case=True)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_path_guard_blocks_symlink_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "data.txt").write_text("s")
    (ws / "link").symlink_to(secret_dir)
    guard = PathGuard(ws)
    with pytest.raises(PathViolation):
        guard.resolve("link/data.txt")
    with pytest.raises(PathViolation):
        guard.resolve("link/new.txt", for_write=True)


def test_audit_log_is_redacted_and_rotates(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", Redactor(environ={}), max_bytes=200)
    for _ in range(5):
        log.record("tool", summary=f"used {FAKE_GH}")
    content = (tmp_path / "audit.jsonl").read_text()
    assert FAKE_GH not in content
    assert (tmp_path / "audit.jsonl.1").exists()
    assert log.tail(2)[-1]["action"] == "tool"


DIFF = """diff --git a/app/db.py b/app/db.py
--- a/app/db.py
+++ b/app/db.py
@@ -10,3 +10,6 @@ def get(user_id):
     conn = connect()
-    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
+    cur = conn.execute(f"SELECT * FROM users WHERE id = {user_id}")
+    data = yaml.load(open("x.yml"))
+    requests.get(url, verify=False)
     return cur
diff --git a/web/app.js b/web/app.js
--- /dev/null
+++ b/web/app.js
@@ -0,0 +1,2 @@
+el.innerHTML = userInput;
+const ok = yaml.load(x, Loader=yaml.SafeLoader)
"""


def test_added_lines_parser_tracks_new_line_numbers() -> None:
    added = added_lines_by_file(DIFF)
    assert [n for n, _ in added["app/db.py"]] == [11, 12, 13]
    assert added["web/app.js"][0] == (1, "el.innerHTML = userInput;")


def test_scan_diff_finds_issues_only_in_added_lines() -> None:
    rules = {(f.rule, f.path, f.line) for f in scan_diff(DIFF)}
    assert ("py-sql-format", "app/db.py", 11) in rules
    assert ("py-yaml-load", "app/db.py", 12) in rules
    assert ("py-verify-false", "app/db.py", 13) in rules
    assert ("js-innerhtml", "web/app.js", 1) in rules
    assert not any(r == "py-yaml-load" and p == "web/app.js" for r, p, _ in rules)


def test_scan_diff_flags_hardcoded_secret() -> None:
    diff = f"+++ b/config.py\n@@ -0,0 +1 @@\n+GITHUB = '{FAKE_GH}'\n"
    findings = scan_diff(diff)
    assert any(f.rule == "hardcoded-secret" for f in findings)
    assert all(FAKE_GH not in f.snippet for f in findings)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('DB_PASSWORD = "hunter2-Xq9!zL7w"', True),
        ("DB_PASSWORD=S3cr3t-pass99", True),
        ("api_key = 'AbC123xyz987QWE'", True),
        ("SECRET_KEY=django-insecure-8a7sd6f87asd6f", True),
        # code, not credentials
        ("token: CancellationToken", False),
        ("tokens = program_tokens(command)", False),
        ("tokens = tokens[1:]", False),
        ('MAX_TOKENS = "max_tokens"', False),
        ("tokenize = 'unicode61'", False),
        ('"api_key_env": "OPENAI_COMPATIBLE_API_KEY"', False),
        ("password = getpass()", False),
        ("input_tokens=resp.usage.input_tokens", False),
    ],
)
def test_credential_assignment_separates_literals_from_code(text: str, expected: bool) -> None:
    assert bool(scan_text(text)) is expected


def test_secret_file_redaction_keeps_structure_hides_values() -> None:
    from ai_engineer.security.secrets import redact_secret_file_text

    text = (
        "DATABASE_URL=postgres://app:pw@db/app\n# a comment\n# OLD_KEY=abc\nexport TOKEN='x'\n\n"
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADAN\n-----END PRIVATE KEY-----\n"
        '{\n  "private_key": "k",\n  "client_email": "a@b"\n}\n[default]\naws_secret_access_key = abc\n'
    )
    out = redact_secret_file_text(text)
    for value in ("postgres://", "abc", "'x'", "MIIEvQ", '"k"', "a@b"):
        assert value not in out
    for kept in ("DATABASE_URL=", "# a comment", "# OLD_KEY=", "export TOKEN=", "-----BEGIN PRIVATE KEY-----", '"private_key": ', "[default]"):
        assert kept in out
    assert out.count("\n") == text.count("\n")


def test_diff_redaction_tracks_hunks_and_file_sections() -> None:
    from ai_engineer.security.secrets import redact_secret_files_in_diff

    git_diff = (
        "diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n@@ -1,2 +1,3 @@\n KEEP=old\n-PASSWORD=hunter2\n"
        "+PASSWORD=hunter3\n+--- not a header\ndiff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        "@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    )
    out = redact_secret_files_in_diff(git_diff, lambda p: p == ".env")
    assert "hunter" not in out and "KEEP=[REDACTED]" in out and "not a header" not in out
    assert "-x = 1\n+x = 2\n" in out  # other files untouched
    plain = "--- a/creds.txt\n+++ b/creds.txt\n@@ -1 +1,2 @@\n-sekrit1\n+sekrit2\n+--- a/fake\n--- a/ok.py\n+++ b/ok.py\n@@ -1 +1 @@\n-a\n+b\n"
    out = redact_secret_files_in_diff(plain, lambda p: p.startswith("creds"))
    assert "sekrit" not in out and "-a\n+b\n" in out
