"""Secret detection and redaction.

Used on tool output before a model sees it, on logs/events/memory/reports, on
staged diffs before agent commits, and by repository discovery (which reports
locations only, never values).
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class _Pattern:
    kind: str
    regex: re.Pattern[str]
    group: int = 0  # which group holds the secret value (0 = whole match)
    check_entropy: bool = False
    assignment: bool = False  # generic key=value rule: apply code-vs-literal heuristics


_PATTERNS: tuple[_Pattern, ...] = (
    _Pattern(
        "private_key",
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY(?: BLOCK)?-----[\s\S]*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY(?: BLOCK)?-----"),
    ),
    _Pattern("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    _Pattern("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    _Pattern("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    _Pattern("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}")),
    _Pattern("openai_api_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}")),
    _Pattern("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    _Pattern("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    _Pattern("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    _Pattern("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b")),
    _Pattern("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    _Pattern("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    _Pattern("url_credentials", re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s/:@'\"]+:([^\s/@'\"]{3,})@"), group=1),
    _Pattern("bearer_token", re.compile(r"(?i)\b(?:bearer|token)\s+([A-Za-z0-9\-._~+/]{20,}=*)"), group=1, check_entropy=True),
    _Pattern(
        "credential_assignment",
        re.compile(
            r"""(?ix)
            \b[A-Z0-9_.\-]*(?:password|passwd|pwd|secret|token|api[_\-]?key|access[_\-]?key|
            private[_\-]?key|client[_\-]?secret|auth[_\-]?key|credentials?)[A-Z0-9_.\-]*
            ["']?\s*[:=]\s*
            (["'`]?)([^\s"'`,;#}{)(<>\[]{8,})
            """
        ),
        group=2,
        check_entropy=True,
        assignment=True,
    ),
)

_PLACEHOLDER = re.compile(
    r"""(?ix)^(
        x{3,}|\*{3,}|\.{3,}|<[^>]*>|\$\{?[A-Z_][A-Z0-9_]*\}?|%\([^)]*\)s|\{\{.*\}\}|
        changeme|change_me|password|secret|example\S*|your[_\-]?\S*|placeholder\S*|dummy\S*|test\S*|
        none|null|true|false|redacted|\[redacted\]|todo|fixme|
        os\.environ.*|process\.env.*|env\(.*|getenv.*|settings\..*|config\..*|self\..*|
        [a-z_]+\.[a-z_]+(\(.*\))?|[a-z_]+\(.*\)?
    )$"""
)

_SECRET_ENV_NAME = re.compile(
    r"(?i)(api[_\-]?key|_key$|^key$|token|secret|passw|pwd|credential|private|cookie|session[_\-]?id|"
    r"auth[_\-]?(?:token|key|header)|access[_\-]?key)"
)
_SAFE_ENV_NAMES = re.compile(r"(?i)(author|committer|_file$|_path$|_dir$|_url$|_host$|_port$|max_.*tokens|tokens?_limit)")


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _char_classes(value: str) -> int:
    return sum(
        bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9_]")
    )


def _looks_secret(value: str) -> bool:
    if _PLACEHOLDER.match(value):
        return False
    if value.isdigit() or len(set(value)) < 5:
        return False
    return shannon_entropy(value) >= 3.0


_CODE_LIKE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _assignment_value_is_secret(match: re.Match[str], text: str) -> bool:
    """Distinguish credential literals from code such as ``token: CancellationToken`` or ``tokens = f(x)``."""
    quote, value = match.group(1), match.group(2)
    if not _looks_secret(value):
        return False
    if quote:
        # a quoted literal: require some randomness (several character classes, not an identifier/constant name)
        identifier_name = bool(_CODE_LIKE.match(value)) and not re.search(r"[0-9]", value)
        return len(value) >= 10 and _char_classes(value) >= 2 and not identifier_name
    following = text[match.end(2) : match.end(2) + 1]
    if following in ("(", "["):
        return False  # a call or subscript expression
    if _CODE_LIKE.match(value):
        return False  # an identifier or attribute access
    return bool(re.search(r"[0-9]", value)) or bool(re.search(r"[^A-Za-z0-9_.]", value))


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    line: int
    start: int
    end: int
    preview: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "line": self.line, "preview": self.preview}


def mask(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…({len(value)} chars)"


def scan_text(text: str, max_findings: int = 100) -> list[SecretFinding]:
    """Return locations of likely secrets. Values are masked in ``preview``."""
    findings: list[SecretFinding] = []
    covered: list[tuple[int, int]] = []
    for pattern in _PATTERNS:
        for match in pattern.regex.finditer(text):
            start, end = match.span(pattern.group)
            if any(s <= start < e for s, e in covered):
                continue
            value = match.group(pattern.group)
            if pattern.assignment:
                if not _assignment_value_is_secret(match, text):
                    continue
            elif pattern.check_entropy and not _looks_secret(value):
                continue
            covered.append((start, end))
            line = text.count("\n", 0, start) + 1
            findings.append(SecretFinding(pattern.kind, line, start, end, mask(value)))
            if len(findings) >= max_findings:
                return findings
    findings.sort(key=lambda f: f.start)
    return findings


def secret_env_values(environ: Mapping[str, str] | None = None) -> set[str]:
    """Values of environment variables whose names suggest secrets."""
    env = os.environ if environ is None else environ
    values = set()
    for name, value in env.items():
        if not value or len(value) < 8 or value.isdigit() or value.lower() in ("true", "false"):
            continue
        if _SECRET_ENV_NAME.search(name) and not _SAFE_ENV_NAMES.search(name):
            values.add(value)
    return values


def is_secret_env_name(name: str) -> bool:
    return bool(_SECRET_ENV_NAME.search(name)) and not _SAFE_ENV_NAMES.search(name)


class Redactor:
    """Replaces secrets in strings and nested structures with ``[REDACTED]``."""

    def __init__(self, extra_values: Iterable[str] = (), environ: Mapping[str, str] | None = None) -> None:
        self._values: set[str] = {v for v in extra_values if v and len(v) >= 6}
        self._values |= secret_env_values(environ)
        self._compiled: re.Pattern[str] | None = None
        self._rebuild()

    def _rebuild(self) -> None:
        if self._values:
            ordered = sorted(self._values, key=len, reverse=True)
            self._compiled = re.compile("|".join(re.escape(v) for v in ordered))
        else:
            self._compiled = None

    def add_value(self, value: str) -> None:
        if value and len(value) >= 6 and value not in self._values:
            self._values.add(value)
            self._rebuild()

    def redact_text(self, text: str) -> str:
        if not text:
            return text
        if self._compiled is not None:
            text = self._compiled.sub(REDACTED, text)
        findings = scan_text(text, max_findings=1000)
        if not findings:
            return text
        out = []
        pos = 0
        for f in findings:
            if f.start < pos:
                continue
            out.append(text[pos : f.start])
            out.append(REDACTED)
            pos = f.end
        out.append(text[pos:])
        return "".join(out)

    def __call__(self, obj: Any) -> Any:
        return self.redact(obj)

    def redact(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {k: self.redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact(v) for v in obj)
        return obj
