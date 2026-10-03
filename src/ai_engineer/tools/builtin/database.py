"""Database tools (SQLite built in). Read-only unless the statement needs writes, which raises the permission level."""

from __future__ import annotations

import asyncio
import re
import sqlite3
from pathlib import Path

from pydantic import Field

from ...config.settings import PermissionLevel
from ...core.errors import ToolError
from ...security.command_risk import Risk
from ..base import ActionAssessment, SideEffect, Tool, ToolContext, ToolInput, ToolResult

_READ_SQL = re.compile(r"^\s*(select|with|explain|pragma\s+(table_info|index_list|foreign_key_list|table_xinfo)|values)\b", re.I)
_DESTRUCTIVE_SQL = re.compile(r"\b(drop\s+(table|index|view|trigger|database)|truncate|delete\s+from\s+\w+\s*;?\s*$|alter\s+table\s+\w+\s+drop)\b", re.I)


def classify_sql(sql: str) -> tuple[Risk, bool]:
    """Returns (risk, read_only)."""
    statements = [s for s in sql.split(";") if s.strip()]
    if not statements:
        return Risk.LOW, True
    if all(_READ_SQL.match(s) for s in statements):
        return Risk.LOW, True
    if any(_DESTRUCTIVE_SQL.search(s.strip()) for s in statements):
        return Risk.HIGH, False
    return Risk.MEDIUM, False


def _connect(path: Path, read_only: bool) -> sqlite3.Connection:
    if read_only:
        return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10)
    return sqlite3.connect(str(path), timeout=10)


class DbSchemaInput(ToolInput):
    database: str = Field(description="Path to a SQLite database file in the workspace")


class DbSchemaTool(Tool):
    name = "db_schema"
    description = "Show tables, columns, indexes and foreign keys of a SQLite database file."
    Input = DbSchemaInput

    async def run(self, args: DbSchemaInput, ctx: ToolContext) -> ToolResult:
        path = ctx.guard.resolve(args.database)
        if not path.is_file():
            raise ToolError(f"database not found: {args.database}")

        def work() -> str:
            con = _connect(path, read_only=True)
            try:
                rows = con.execute("SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
                lines = []
                for kind, name, sql in rows:
                    lines.append(f"-- {kind} {name}\n{sql or ''};")
                return "\n".join(lines) or "(empty database)"
            finally:
                con.close()

        try:
            return ToolResult(content=await asyncio.to_thread(work))
        except sqlite3.DatabaseError as exc:
            raise ToolError(f"cannot read database: {exc}") from exc


class DbQueryInput(ToolInput):
    database: str = Field(description="Path to a SQLite database file in the workspace")
    sql: str = Field(description="SQL statement(s). Prefer parameters over string formatting.")
    params: list[str | int | float | None] = Field(default_factory=list)
    max_rows: int = Field(default=200, ge=1, le=5000)


class DbQueryTool(Tool):
    name = "db_query"
    description = (
        "Run SQL against a SQLite database file. SELECT/EXPLAIN run read-only; data or schema changes need "
        "development permission and destructive statements (DROP, TRUNCATE, DELETE without WHERE) need approval."
    )
    Input = DbQueryInput
    side_effect = SideEffect.WRITE

    def assess(self, args: DbQueryInput, ctx: ToolContext) -> ActionAssessment:
        risk, read_only = classify_sql(args.sql)
        level = PermissionLevel.READ_ONLY if read_only else PermissionLevel.DEVELOPMENT
        return ActionAssessment(level=level, summary=f"SQL on {args.database}: {' '.join(args.sql.split())[:120]}", risk=risk, read_only=read_only)

    @property
    def concurrency_safe(self) -> bool:
        return False

    async def run(self, args: DbQueryInput, ctx: ToolContext) -> ToolResult:
        _, read_only = classify_sql(args.sql)
        path = ctx.guard.resolve(args.database, for_write=not read_only)
        if not path.is_file():
            raise ToolError(f"database not found: {args.database}")
        if not read_only:
            ctx.files.before_write(path)

        def work() -> tuple[list[str], list[tuple], int]:
            con = _connect(path, read_only=read_only)
            try:
                if read_only:
                    cur = con.execute(args.sql, args.params)
                    cols = [d[0] for d in cur.description or []]
                    rows = cur.fetchmany(args.max_rows + 1)
                    return cols, rows, -1
                if args.params:
                    cur = con.execute(args.sql, args.params)
                else:
                    cur = con.executescript(args.sql) if ";" in args.sql.strip().rstrip(";") else con.execute(args.sql)
                con.commit()
                return [], [], cur.rowcount
            finally:
                con.close()

        try:
            cols, rows, changed = await asyncio.to_thread(work)
        except sqlite3.Error as exc:
            raise ToolError(f"SQL error: {exc}") from exc
        if not read_only:
            ctx.files.after_write(path)
            return ToolResult(content=f"statement executed; rows affected: {changed if changed >= 0 else 'unknown'}")
        more = len(rows) > args.max_rows
        rows = rows[: args.max_rows]
        lines = [" | ".join(cols)] if cols else []
        lines += [" | ".join("NULL" if v is None else str(v) for v in row) for row in rows]
        if more:
            lines.append(f"[more than {args.max_rows} rows; refine the query]")
        return ToolResult(content="\n".join(lines) or "(no rows)", data={"rows": len(rows)})


DB_TOOLS: list[type[Tool]] = [DbSchemaTool, DbQueryTool]
