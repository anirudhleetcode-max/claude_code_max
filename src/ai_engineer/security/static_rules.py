"""Lightweight static security checks over changed code.

These are high-signal heuristics, not a full SAST engine. They run on the
*added* lines of a diff so pre-existing issues are not attributed to the agent.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .secrets import scan_text


@dataclass(frozen=True)
class SecurityFinding:
    rule: str
    severity: str  # "high" | "medium" | "low"
    path: str
    line: int
    message: str
    snippet: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Rule:
    rule: str
    severity: str
    languages: tuple[str, ...]  # file suffixes; empty = all
    regex: re.Pattern[str]
    message: str
    unless: re.Pattern[str] | None = None


_PY = (".py",)
_JS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
_GO = (".go",)

RULES: tuple[_Rule, ...] = (
    _Rule("py-eval", "high", _PY, re.compile(r"(?<![\w.])(eval|exec)\s*\("), "eval/exec of dynamic code"),
    _Rule("py-shell-true", "high", _PY, re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True"), "subprocess with shell=True"),
    _Rule("py-os-system", "medium", _PY, re.compile(r"\bos\.(system|popen)\s*\("), "os.system/os.popen runs a shell"),
    _Rule("py-pickle", "high", _PY, re.compile(r"\b(pickle|cPickle|dill|marshal)\.loads?\s*\("), "deserializing untrusted data with pickle/marshal"),
    _Rule(
        "py-yaml-load", "high", _PY, re.compile(r"\byaml\.(load|load_all)\s*\("), "yaml.load without SafeLoader",
        unless=re.compile(r"Loader\s*=\s*(yaml\.)?(Safe|CSafe)Loader"),
    ),
    _Rule("py-verify-false", "high", _PY, re.compile(r"verify\s*=\s*False"), "TLS certificate verification disabled"),
    _Rule("py-unverified-ssl", "high", _PY, re.compile(r"ssl\._create_unverified_context|CERT_NONE"), "TLS verification disabled"),
    _Rule(
        "py-sql-format", "high", _PY,
        re.compile(r"\.(execute|executemany|raw)\s*\(\s*(f[\"']|[\"'][^\"']*[\"']\s*(%|\.format\(|\+))", re.I),
        "SQL built with string formatting (injection risk); use parameters",
    ),
    _Rule("py-weak-hash", "low", _PY, re.compile(r"hashlib\.(md5|sha1)\s*\("), "weak hash (fine for checksums, not for passwords)", unless=re.compile(r"usedforsecurity\s*=\s*False")),
    _Rule("py-mktemp", "medium", _PY, re.compile(r"tempfile\.mktemp\s*\("), "insecure tempfile.mktemp"),
    _Rule("py-jwt-noverify", "high", _PY, re.compile(r"verify_signature[\"']?\s*:\s*False|jwt\.decode\([^)]*verify\s*=\s*False"), "JWT signature verification disabled"),
    _Rule("py-debug-true", "medium", _PY, re.compile(r"^\s*DEBUG\s*=\s*True\b"), "DEBUG enabled"),
    _Rule("py-bind-all", "low", _PY, re.compile(r"[\"']0\.0\.0\.0[\"']"), "binds to all interfaces"),
    _Rule("js-eval", "high", _JS, re.compile(r"(?<![\w.])(eval\s*\(|new\s+Function\s*\()"), "eval/new Function of dynamic code"),
    _Rule("js-innerhtml", "medium", _JS, re.compile(r"\.innerHTML\s*=|dangerouslySetInnerHTML|document\.write\s*\("), "unsanitized HTML injection sink"),
    _Rule("js-child-exec", "high", _JS, re.compile(r"\bexec(Sync)?\s*\(\s*`[^`]*\$\{"), "shell command built from template literal"),
    _Rule("js-tls-off", "high", _JS, re.compile(r"rejectUnauthorized\s*:\s*false|NODE_TLS_REJECT_UNAUTHORIZED"), "TLS verification disabled"),
    _Rule("js-sql-template", "high", _JS, re.compile(r"\.(query|execute|raw)\s*\(\s*`[^`]*\$\{"), "SQL built with template literal (injection risk)"),
    _Rule("go-tls-skip", "high", _GO, re.compile(r"InsecureSkipVerify\s*:\s*true"), "TLS verification disabled"),
    _Rule("go-sql-sprintf", "high", _GO, re.compile(r"\.(Query|Exec|QueryRow)(Context)?\([^)]*fmt\.Sprintf"), "SQL built with Sprintf (injection risk)"),
    _Rule("go-shell", "medium", _GO, re.compile(r"exec\.Command\(\s*\"(sh|bash)\"\s*,\s*\"-c\""), "shell command execution"),
    _Rule("any-chmod-777", "medium", (), re.compile(r"chmod\s+(-R\s+)?777|0o?777\b"), "world-writable permissions"),
    _Rule("any-cors-wildcard", "low", (), re.compile(r"Access-Control-Allow-Origin[\"']?\s*[:,]\s*[\"']\*"), "CORS allows any origin"),
)


def scan_lines(path: str, lines: list[tuple[int, str]]) -> list[SecurityFinding]:
    """Scan ``(line_number, text)`` pairs from ``path``."""
    findings: list[SecurityFinding] = []
    lower = path.lower()
    for rule in RULES:
        if rule.languages and not lower.endswith(rule.languages):
            continue
        for number, text in lines:
            stripped = text.strip()
            if stripped.startswith(("#", "//", "*")):
                continue
            if rule.regex.search(text) and not (rule.unless and rule.unless.search(text)):
                findings.append(SecurityFinding(rule.rule, rule.severity, path, number, rule.message, stripped[:200]))
    joined = "\n".join(t for _, t in lines)
    if joined:
        numbers = [n for n, _ in lines]
        for secret in scan_text(joined):
            idx = min(secret.line - 1, len(numbers) - 1)
            findings.append(
                SecurityFinding("hardcoded-secret", "high", path, numbers[idx], f"possible hardcoded {secret.kind}", secret.preview)
            )
    return findings


def added_lines_by_file(diff: str) -> dict[str, list[tuple[int, str]]]:
    """Parse a unified diff into ``{path: [(new_line_number, text), ...]}`` for added lines."""
    result: dict[str, list[tuple[int, str]]] = {}
    current: str | None = None
    line_no = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            current = None if target == "/dev/null" else re.sub(r"^b/", "", target)
            if current is not None:
                result.setdefault(current, [])
            continue
        if raw.startswith("--- "):
            continue
        hunk = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if hunk:
            line_no = int(hunk.group(1))
            continue
        if current is None:
            continue
        if raw.startswith("+"):
            result[current].append((line_no, raw[1:]))
            line_no += 1
        elif raw.startswith("-"):
            continue
        elif raw.startswith("\\"):
            continue
        else:
            line_no += 1
    return result


def scan_diff(diff: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for path, lines in added_lines_by_file(diff).items():
        if lines:
            findings.extend(scan_lines(path, lines))
    return findings
