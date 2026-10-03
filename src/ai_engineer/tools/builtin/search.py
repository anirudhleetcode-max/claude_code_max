"""Search and repository-intelligence tools."""

from __future__ import annotations

import asyncio

from pydantic import Field

from ...core.errors import ToolError
from ..base import Tool, ToolContext, ToolInput, ToolResult


class FindFilesInput(ToolInput):
    pattern: str = Field(description="Glob pattern such as '**/*.py', 'src/**/auth*', or a file name")
    max_results: int = Field(default=200, ge=1, le=2000)


class FindFilesTool(Tool):
    name = "find_files"
    description = "Find files by glob pattern (matches the relative path or the file name). Ignores dependency/build directories."
    Input = FindFilesInput

    def summarize(self, args: FindFilesInput) -> str:
        return f"Finding files matching {args.pattern}"

    async def run(self, args: FindFilesInput, ctx: ToolContext) -> ToolResult:
        from ...repo.search import find_files

        matches = await asyncio.to_thread(find_files, ctx.workspace, args.pattern, args.max_results)
        if not matches:
            return ToolResult(content=f"no files match {args.pattern!r}", data={"matches": 0})
        suffix = f"\n[limited to {args.max_results} results]" if len(matches) >= args.max_results else ""
        return ToolResult(content="\n".join(matches) + suffix, data={"matches": len(matches)})


class SearchTextInput(ToolInput):
    pattern: str = Field(description="Regular expression (or literal text when regex=false)")
    regex: bool = True
    case_sensitive: bool = False
    glob: str | None = Field(default=None, description="Restrict to files matching this glob, e.g. '*.py'")
    max_results: int = Field(default=100, ge=1, le=1000)


class SearchTextTool(Tool):
    name = "search_text"
    description = "Search file contents (ripgrep when available). Returns path:line: text for each match."
    Input = SearchTextInput
    timeout_s = 60.0

    def summarize(self, args: SearchTextInput) -> str:
        return f"Searching for {args.pattern!r}" + (f" in {args.glob}" if args.glob else "")

    async def run(self, args: SearchTextInput, ctx: ToolContext) -> ToolResult:
        from ...repo.search import search_text

        try:
            matches = await asyncio.to_thread(
                search_text, ctx.workspace, args.pattern,
                regex=args.regex, case_sensitive=args.case_sensitive, glob=args.glob, max_results=args.max_results,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        if not matches:
            return ToolResult(content=f"no matches for {args.pattern!r}", data={"matches": 0})
        lines = [f"{m.path}:{m.line}: {m.text}" for m in matches]
        if len(matches) >= args.max_results:
            lines.append(f"[limited to {args.max_results} matches; narrow the pattern or glob]")
        return ToolResult(content="\n".join(lines), data={"matches": len(matches)})


def _index(ctx: ToolContext):  # type: ignore[no-untyped-def]
    if ctx.repo_index is None:
        raise ToolError("the repository index is not available")
    return ctx.repo_index


class RepoOverviewInput(ToolInput):
    pass


class RepoOverviewTool(Tool):
    name = "repo_overview"
    description = "Summarize the repository: languages, frameworks, package managers, entry points, test/lint/build commands, key files."
    Input = RepoOverviewInput

    async def run(self, args: RepoOverviewInput, ctx: ToolContext) -> ToolResult:
        profile = ctx.profile
        if profile is None:
            from ...repo.discovery import discover

            profile = await asyncio.to_thread(discover, ctx.workspace)
            ctx.profile = profile
        lines = [profile.summary, ""]
        if profile.languages:
            langs = sorted(profile.languages.items(), key=lambda kv: -kv[1].get("files", 0))[:8]
            lines.append("Languages: " + ", ".join(f"{k} ({v.get('files', 0)} files)" for k, v in langs))
        for label, values in (
            ("Frameworks", profile.frameworks), ("Package managers", profile.package_managers),
            ("Entry points", profile.entry_points[:15]), ("Test dirs", profile.test_dirs[:10]),
            ("Config files", profile.config_files[:20]), ("Docs", profile.doc_files[:15]), ("Notable", profile.notable_files[:20]),
        ):
            if values:
                lines.append(f"{label}: {', '.join(values)}")
        for kind, suggestions in profile.commands.items():
            if suggestions:
                best = suggestions[0]
                lines.append(f"{kind} command: {best.command}  (from {best.source})")
        if profile.secret_findings:
            lines.append(f"Potential secrets detected in {len({f['path'] for f in profile.secret_findings})} file(s) (values hidden).")
        return ToolResult(content="\n".join(lines))


class FindSymbolInput(ToolInput):
    name: str = Field(description="Symbol name (class, function, method, type) — exact, prefix or substring")
    kind: str | None = Field(default=None, description="Optional kind filter: class, function, method, interface, type, struct, enum")


class FindSymbolTool(Tool):
    name = "find_symbol"
    description = "Locate definitions of a symbol across the repository (path:line, kind, parent, signature)."
    Input = FindSymbolInput

    async def run(self, args: FindSymbolInput, ctx: ToolContext) -> ToolResult:
        results = await asyncio.to_thread(_index(ctx).find_symbol, args.name, args.kind)
        if not results:
            return ToolResult(content=f"no symbol matching {args.name!r}; try search_text")
        lines = [
            f"{r['path']}:{r['line']} {r['kind']} {(r.get('parent') + '.') if r.get('parent') else ''}{r['name']} {r.get('signature', '')}".rstrip()
            for r in results
        ]
        return ToolResult(content="\n".join(lines), data={"matches": len(results)})


class PathInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root")


class FindDependentsTool(Tool):
    name = "find_dependents"
    description = "List files that import the given file, directly or transitively (what could break if it changes)."
    Input = PathInput

    async def run(self, args: PathInput, ctx: ToolContext) -> ToolResult:
        rel = ctx.guard.relative(ctx.guard.resolve(args.path))
        index = _index(ctx)
        direct = await asyncio.to_thread(index.dependents, rel, False)
        transitive = await asyncio.to_thread(index.dependents, rel, True)
        indirect = [p for p in transitive if p not in direct]
        lines = [f"Direct dependents of {rel} ({len(direct)}):", *direct]
        if indirect:
            lines += [f"Indirect dependents ({len(indirect)}):", *indirect[:200]]
        return ToolResult(content="\n".join(lines), data={"paths": transitive})


class RelatedTestsTool(Tool):
    name = "related_tests"
    description = "Find test files that cover the given file (by imports and naming conventions)."
    Input = PathInput

    async def run(self, args: PathInput, ctx: ToolContext) -> ToolResult:
        rel = ctx.guard.relative(ctx.guard.resolve(args.path))
        tests = await asyncio.to_thread(_index(ctx).related_tests, rel)
        if not tests:
            return ToolResult(content=f"no tests found for {rel}")
        return ToolResult(content="\n".join(tests), data={"paths": tests})


class CodeSearchInput(ToolInput):
    query: str = Field(description="Natural-language or keyword query, e.g. 'password reset token expiry'")
    limit: int = Field(default=10, ge=1, le=50)


class CodeSearchTool(Tool):
    name = "code_search"
    description = "Ranked keyword (BM25) search over code chunks; good for 'where is X implemented?' questions."
    Input = CodeSearchInput

    async def run(self, args: CodeSearchInput, ctx: ToolContext) -> ToolResult:
        hits = await asyncio.to_thread(_index(ctx).search, args.query, args.limit)
        if not hits:
            return ToolResult(content="no results")
        parts = [f"{h['path']}:{h['start_line']}-{h['end_line']} (score {h['score']:.2f})\n{h['snippet']}" for h in hits]
        return ToolResult(content="\n\n".join(parts), data={"matches": len(hits)})


SEARCH_TOOLS: list[type[Tool]] = [
    FindFilesTool, SearchTextTool, RepoOverviewTool, FindSymbolTool, FindDependentsTool, RelatedTestsTool, CodeSearchTool,
]
