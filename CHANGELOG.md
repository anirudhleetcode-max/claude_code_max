# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-03

### Added
- Model-agnostic model layer: provider-neutral messages, `ModelProvider` interface
  (generate, stream, structured output with validation and repair, tool calls,
  embeddings, health checks), prompted tool calling for models without native tools.
- Providers: Anthropic (official SDK), OpenAI and OpenAI-compatible servers,
  Google Gemini, Ollama, and a deterministic scripted provider for tests and demos.
- Role-based router with fallback chains, retries with exponential backoff and
  jitter, circuit breakers and refusal fallback.
- Tool system with schema validation, permission levels and operating modes,
  approval brokers, timeouts, cancellation, redaction, truncation and audit log;
  built-in file, search, repository, terminal, process, git, test/lint/typecheck/
  build/format, SQLite, web, browser, environment and memory tools.
- Security: command risk classifier, workspace path guard, secret detection and
  redaction, secret-free child environments, static security rules for diffs,
  stale-write protection, optional Docker command sandbox.
- Repository intelligence: discovery, incremental index, symbols, import graph,
  dependents, related tests, BM25 code search.
- Pipeline: understanding with deterministic routing, planning with validated
  subtask DAGs, agent loop with budgets and loop detection, baseline checks,
  validation engine with output parsers, structured failure records and bounded
  repair, deterministic and model review, quality gates, final QA, reports and
  retrospectives.
- Persistence and recovery: SQLite state with leases, crash detection, resumable
  stages, git snapshot and file-backup checkpoints with reversible restores,
  queues, schedules and a daemon.
- Layered memory with versioning, confidence, staleness detection and FTS search.
- Observability: typed events, redacted traces, metrics, terminal renderer, web
  dashboard (token-protected, localhost by default) with live activity, plan and
  subtasks, gates, approvals, questions, checkpoint diffs and reports.
- Benchmark suite (10 categories, hidden verification, harness and model suites),
  adversarial tests, cross-platform CI, install scripts, Dockerfile, documentation.

### Fixed during development (found by end-to-end and CI testing)
- Snapshot checkpoints could miss a same-size edit made within the same second
  (git's racy-clean check was defeated by copying the index without its mtime).
- Stale Python bytecode could be reused after a same-second, same-size edit;
  agent-run commands now set `PYTHONDONTWRITEBYTECODE=1`.
- Generic credential detection flagged code (type annotations, calls) as secrets;
  it now only flags literal values.
- `PWD` was treated as a password variable, redacting paths in output.
- Test-failure paths kept Windows separators; the repository index kept stale data
  when its database file could not be deleted (Windows file locking).
- A `.agent/.gitignore` that un-ignored its config made fresh repositories dirty.
- On Windows, targeted test and lint runs always fell back to the full command,
  because any backslash in the command disabled targeting. Commands are now
  rebuilt with `cmd.exe` quoting on Windows.

### Security and robustness fixes (found by the final audit)
- `db_query` could create or overwrite SQLite files outside the workspace with
  `ATTACH DATABASE` or `VACUUM INTO` without approval. Attaching is now disabled
  on every connection.
- SQL classification missed unbounded `DELETE`/`ALTER … DROP` on quoted or
  schema-qualified table names. It also rated `WITH … DELETE` as read-only, but
  the read-only connection still refused that write. `UPDATE` without `WHERE`
  now needs approval like `DELETE` without `WHERE`.
- `git_commit` also committed changes the user had staged separately (unscanned).
  A broad path such as `.` could commit key or credential files. Commits are now
  limited to the given paths, and secret and protected files are refused. The
  pipeline's auto-commit uses the same checks.
- Only the top-level `.git` was write-protected, so hooks in nested repositories
  and submodules were writable. Protected and secret-file patterns were also
  case-sensitive, although `.GIT/hooks` or `.ENV` are the same files on macOS
  and Windows. Both are fixed.
- `web_fetch` followed redirects automatically and only checked the final URL,
  so a public page could make the agent send a request to a local or private
  address. Every hop is now checked before it is requested. The browser tool
  refuses cloud metadata endpoints.
- `read_file` redacted the values in `.env`, key and credential files, but
  `search_text`, `git_diff` and checkpoint diffs (sent to the reviewer model and
  shown in reports, the CLI and the dashboard) did not. All of them now apply the
  same redaction.
- Stopping a task did not interrupt a model request already in flight; the router
  now cancels it immediately.
- Task events are returned in emission order (events emitted within the same
  millisecond could appear out of order).
