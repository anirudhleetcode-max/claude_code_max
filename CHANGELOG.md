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
  dashboard with live activity, approvals and diffs.
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
