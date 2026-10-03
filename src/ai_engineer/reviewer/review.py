"""Independent review: deterministic diff checks + a structured model review."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import AllModelsFailedError, MalformedOutputError
from ..core.types import Message
from ..core.util import truncate_middle
from ..models.base import ModelRequest
from ..security.static_rules import SecurityFinding, added_lines_by_file, scan_diff

Severity = Literal["blocker", "major", "minor", "nit"]


class ReviewIssue(BaseModel):
    severity: Severity
    category: str = Field(description="correctness, requirements, edge-cases, error-handling, security, performance, maintainability, readability, types, dependencies, api, database, tests, regression-risk")
    file: str | None = None
    line: int | None = None
    description: str
    suggestion: str = ""


class CriterionStatus(BaseModel):
    criterion: str
    status: Literal["met", "unmet", "unverified"]
    evidence: str = ""


class ModelReview(BaseModel):
    """Schema the reviewer model must return."""

    verdict: Literal["approve", "request_changes"]
    summary: str
    issues: list[ReviewIssue] = Field(default_factory=list)
    requirements: list[CriterionStatus] = Field(default_factory=list)


class ReviewResult(BaseModel):
    verdict: Literal["approve", "request_changes"]
    summary: str
    issues: list[ReviewIssue] = Field(default_factory=list)
    requirements: list[CriterionStatus] = Field(default_factory=list)
    source: str = "deterministic"
    model: str = ""

    def blocking(self) -> list[ReviewIssue]:
        return [i for i in self.issues if i.severity in ("blocker", "major")]

    def render(self) -> str:
        lines = [f"Verdict: {self.verdict} — {self.summary}"]
        for i in self.issues:
            loc = f" ({i.file}:{i.line})" if i.file and i.line else (f" ({i.file})" if i.file else "")
            lines.append(f"- [{i.severity}/{i.category}]{loc} {i.description}" + (f" → {i.suggestion}" if i.suggestion else ""))
        for c in self.requirements:
            lines.append(f"- criterion [{c.status}]: {c.criterion}" + (f" — {c.evidence}" if c.evidence else ""))
        return "\n".join(lines)


_TEST_FILE = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+\.py$|_test\.(py|go)$|\.(test|spec)\.[jt]sx?$")
_DEBUG = re.compile(r"(?<![\w.])(print\(|console\.log\(|debugger;|breakpoint\(\)|pdb\.set_trace\(|binding\.pry|var_dump\()")
_SKIP_TEST = re.compile(r"(@pytest\.mark\.skip|@unittest\.skip|\bit\.skip\(|\bdescribe\.skip\(|\btest\.skip\(|\bxit\(|\bxdescribe\(|t\.Skip\(|#\[ignore\])")
_CONFLICT = re.compile(r"^(<<<<<<< |=======$|>>>>>>> )")
_MANIFESTS = ("package.json", "pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "Gemfile", "composer.json", "setup.py", "setup.cfg")


def _security_issue(f: SecurityFinding) -> ReviewIssue:
    sev: Severity = "blocker" if f.severity == "high" else ("major" if f.severity == "medium" else "minor")
    return ReviewIssue(severity=sev, category="security", file=f.path, line=f.line, description=f"{f.message} [{f.rule}]", suggestion="")


def deterministic_review(diff: str, deleted_files: list[str] | None = None, repo_has_tests: bool = True) -> tuple[list[ReviewIssue], list[SecurityFinding]]:
    """Cheap, reliable checks over the added lines of a unified diff."""
    issues: list[ReviewIssue] = []
    findings = scan_diff(diff)
    issues.extend(_security_issue(f) for f in findings)
    added = added_lines_by_file(diff)
    touched_tests = any(_TEST_FILE.search(p) for p in added)
    touched_source = [p for p in added if not _TEST_FILE.search(p) and p.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php", ".cs", ".kt"))]
    for path, lines in added.items():
        is_test = bool(_TEST_FILE.search(path))
        for number, text in lines:
            if _CONFLICT.match(text):
                issues.append(ReviewIssue(severity="blocker", category="correctness", file=path, line=number, description="merge conflict marker committed"))
            if not is_test and _DEBUG.search(text) and not path.endswith((".md", ".txt")) and "/cli" not in path and not path.endswith(("cli.py", "__main__.py")):
                issues.append(ReviewIssue(severity="minor", category="maintainability", file=path, line=number, description=f"debug output left in code: {text.strip()[:100]}"))
            if _SKIP_TEST.search(text):
                issues.append(ReviewIssue(severity="major", category="tests", file=path, line=number, description="a test is being skipped; tests must not be disabled to make validation pass"))
            if re.search(r"\b(TODO|FIXME|XXX)\b", text):
                issues.append(ReviewIssue(severity="nit", category="maintainability", file=path, line=number, description=f"unresolved marker added: {text.strip()[:100]}"))
        if path.endswith(_MANIFESTS) or path.rsplit("/", 1)[-1] in _MANIFESTS:
            issues.append(ReviewIssue(severity="minor", category="dependencies", file=path, description="dependency manifest changed; confirm new dependencies are necessary, pinned appropriately and installed"))
    for path in deleted_files or []:
        if _TEST_FILE.search(path):
            issues.append(ReviewIssue(severity="major", category="tests", file=path, description="a test file was deleted"))
    if touched_source and repo_has_tests and not touched_tests:
        issues.append(ReviewIssue(severity="minor", category="tests", description="source code changed but no tests were added or updated"))
    return issues, findings


FOCUS_TEXT = {
    "security": "security (in depth): authentication and authorization checks, secret handling, input validation, "
    "injection (SQL, shell, template, path traversal), unsafe deserialization, TLS verification, error messages leaking data",
    "performance": "performance (in depth): algorithmic complexity on realistic input sizes, N+1 queries, unnecessary I/O or "
    "allocations in hot paths, missing caching or batching, blocking calls in async code",
}

REVIEW_SYSTEM = """You are an independent senior code reviewer. You did not write this change. Review it
skeptically and precisely against the task requirements. Check: correctness, requirements coverage,
edge cases, error handling, security, performance, maintainability, readability, type safety,
dependency correctness, API correctness, database correctness, test coverage and regression risk.

Rules:
- Only report real, specific problems visible in the diff or implied by the evidence. No speculation,
  no style preferences presented as defects.
- severity: blocker (wrong/broken/insecure), major (likely bug or missing requirement), minor
  (should fix), nit (optional polish).
- For every acceptance criterion state met / unmet / unverified with concrete evidence (file:line,
  test name). "unverified" when the diff and test results do not show it.
- verdict "approve" only when no blocker/major issue exists and no criterion is unmet."""


async def model_review(
    model: Any,
    *,
    task: str,
    criteria: list[str],
    diff: str,
    validation_summary: str,
    deterministic: list[ReviewIssue],
    project_brief: str = "",
    max_diff_chars: int = 60000,
    cancel: Any = None,
    focus: list[str] | None = None,
) -> ReviewResult:
    """Ask the reviewer model for a structured review. Raises on model failure."""
    det = "\n".join(f"- [{i.severity}] {i.file or ''}:{i.line or ''} {i.description}" for i in deterministic) or "(none)"
    crit = "\n".join(f"- {c}" for c in criteria) or "- (none stated: infer from the task)"
    content = (
        f"## Task\n{task}\n\n## Acceptance criteria\n{crit}\n\n"
        + (f"## Project\n{project_brief}\n\n" if project_brief else "")
        + (("## Focus areas requested by triage\n" + "\n".join(f"- {FOCUS_TEXT.get(f, f)}" for f in focus) + "\n\n") if focus else "")
        + f"## Validation results (actually executed)\n{validation_summary or '(none)'}\n\n"
        f"## Automated findings\n{det}\n\n"
        f"## Diff\n```diff\n{truncate_middle(diff, max_diff_chars)}\n```"
    )
    req = ModelRequest(messages=[Message.user(content)], system=REVIEW_SYSTEM, metadata={"role": "reviewer", "stage": "review"})
    result = await model.structured_output(req, ModelReview, cancel=cancel)
    review: ModelReview = result.value
    model_ref = str(getattr(model, "last_ref", "") or "")
    return ReviewResult(
        verdict=review.verdict, summary=review.summary, issues=review.issues, requirements=review.requirements,
        source="model", model=model_ref,
    )


def combine(deterministic: list[ReviewIssue], model_result: ReviewResult | None, note: str = "") -> ReviewResult:
    issues = list(deterministic)
    requirements: list[CriterionStatus] = []
    summary_parts = []
    source = "deterministic"
    model_ref = ""
    if model_result is not None:
        existing = {(i.file, i.line, i.description) for i in issues}
        issues += [i for i in model_result.issues if (i.file, i.line, i.description) not in existing]
        requirements = model_result.requirements
        summary_parts.append(model_result.summary)
        source = "model+deterministic"
        model_ref = model_result.model
    if note:
        summary_parts.append(note)
    blocking = [i for i in issues if i.severity in ("blocker", "major")]
    unmet = [c for c in requirements if c.status == "unmet"]
    # the reviewer's explicit request for changes stands even when it reports no blocking issue
    model_rejects = model_result is not None and model_result.verdict == "request_changes"
    verdict: Literal["approve", "request_changes"] = "request_changes" if blocking or unmet or model_rejects else "approve"
    if not summary_parts:
        summary_parts.append("automated checks only")
    return ReviewResult(verdict=verdict, summary=" ".join(summary_parts), issues=issues, requirements=requirements, source=source, model=model_ref)


async def review_change(
    model: Any,
    *,
    task: str,
    criteria: list[str],
    diff: str,
    validation_summary: str,
    deleted_files: list[str] | None = None,
    repo_has_tests: bool = True,
    use_model: bool = True,
    project_brief: str = "",
    cancel: Any = None,
    focus: list[str] | None = None,
) -> tuple[ReviewResult, list[SecurityFinding]]:
    deterministic, findings = deterministic_review(diff, deleted_files, repo_has_tests)
    model_result: ReviewResult | None = None
    note = ""
    if use_model and model is not None and diff.strip():
        try:
            model_result = await model_review(
                model, task=task, criteria=criteria, diff=diff, validation_summary=validation_summary,
                deterministic=deterministic, project_brief=project_brief, cancel=cancel, focus=focus,
            )
        except (AllModelsFailedError, MalformedOutputError) as exc:
            note = f"model review unavailable ({type(exc).__name__}); deterministic checks only"
    result = combine(deterministic, model_result, note)
    if use_model and model_result is None:
        result.source = "deterministic-only"
    return result, findings
