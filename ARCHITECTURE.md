# AI Engineer — Architecture

AI Engineer is a self-hosted, model-agnostic autonomous software-engineering agent.
It is a *harness*: the intelligence comes from whichever language model you connect;
everything else — tools, safety, verification, persistence, recovery, observability —
is deterministic code in this repository.

This document describes the system as implemented. Where something is a design
limitation or unverified, it says so.

---

## 1. Design principles

1. **Evidence beats confidence.** A task is only `COMPLETED` when configured
   quality gates *actually ran and passed*. Anything that could not be checked is
   reported as `UNVERIFIED`, never as passed.
2. **AI for decisions, programs for deterministic work.** Repository discovery,
   test-command detection, output parsing, secret scanning, risk classification,
   diffing and checkpointing are plain code. Models are used for understanding,
   planning, editing, diagnosing and reviewing.
3. **The model is replaceable.** No provider-specific logic exists outside
   `ai_engineer/providers/`. The agent talks to a canonical message format and a
   `ModelProvider` interface.
4. **Safe by default.** Workspace path jail, permission levels, command risk
   classification, approvals, secret redaction, audit log, reversible changes.
5. **Append-only conversations.** Agent conversations are never edited in place.
   When context grows too large the agent starts a *new* conversation seeded with
   a structured hand-off summary. This keeps provider features that depend on an
   unmodified history (prompt caching, preserved reasoning blocks) valid.
6. **Survive interruption.** Every stage transition is persisted. After a crash
   the agent never assumes an interrupted step completed; it re-inspects the
   repository and re-verifies.

---

## 2. High-level structure

```
                       ┌──────────────────────────────────────────┐
  CLI (aie) ──────────▶│                Runtime                   │◀────── Web dashboard
  Web UI (SSE) ◀───────│  config · event bus · stores · services  │        (approvals, stop)
                       └──────────────┬───────────────────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │      Orchestrator       │  pipeline state machine
                         │ understand→plan→execute │  (persisted per stage)
                         │ →validate→repair→review │
                         │ →gates→checkpoint→QA    │
                         └──┬─────────┬─────────┬──┘
             ┌──────────────┘         │         └──────────────┐
     ┌───────▼───────┐        ┌───────▼──────┐        ┌────────▼────────┐
     │ Agent loop    │        │ Validation   │        │ Reviewer        │
     │ (executor)    │        │ engine       │        │ deterministic + │
     │ model ⇄ tools │        │ tests/lint/  │        │ model review    │
     └───┬───────┬───┘        │ type/build   │        └─────────────────┘
         │       │            └──────────────┘
 ┌───────▼──┐ ┌──▼──────────────────┐
 │ Model    │ │ Tool executor       │ validation · permission policy · approval
 │ router   │ │                     │ timeout · cancellation · redaction · audit
 │ retry +  │ └──┬──────────────────┘
 │ fallback │    │  fs · search · repo · terminal · git · tests · web · browser · db · env · memory
 └───┬──────┘    │
     │           ▼
 ┌───▼─────────────────────┐   ┌───────────────────┐   ┌────────────────────┐
 │ Providers               │   │ Repo intelligence │   │ Persistence        │
 │ anthropic · openai(+    │   │ discovery · index │   │ .agent/state.db    │
 │ compatible) · google ·  │   │ symbols · imports │   │ tasks · checkpoints│
 │ ollama · scripted       │   │ BM25 search       │   │ memory · failures  │
 └─────────────────────────┘   └───────────────────┘   │ tests · metrics    │
                                                       └────────────────────┘
```

### Package layout (`src/ai_engineer/`)

| Package | Responsibility |
|---|---|
| `core/` | Canonical types (messages, content blocks, tool specs), errors, event bus, ids, cancellation tokens, small utilities |
| `config/` | Typed settings (pydantic), layered loading (defaults → global → project → env → CLI), `.env` loader, platform paths |
| `models/` | `ModelProvider` interface, request/response types, provider registry, role router, retry/backoff, circuit breaker, fallback chain, structured output (parse + validate + repair), prompted tool-calling adapter, token estimation |
| `providers/` | Adapters: `anthropic` (official SDK), `openai` (Chat Completions; also any OpenAI-compatible server), `google` (Gemini REST), `ollama` (native REST), `scripted` (deterministic, for tests/offline/benchmarks) |
| `security/` | Secret detection and redaction, command risk classifier, workspace path guard, audit log, lightweight static security rules |
| `tools/` | Tool base class + registry, executor, permission policy, approval brokers, process manager, built-in tools |
| `repo/` | Repository discovery, incremental file index, symbol extraction (Python AST; regex for JS/TS/Go/Rust/Java), import graph, text/BM25 search, related-test lookup |
| `memory/` | Layered memory store (session, project, engineering, command, decision) with FTS search, versioning, confidence, source and staleness tracking; project context files |
| `context/` | Context builder: selects relevant files/memory within a token budget; hand-off summaries for conversation reset |
| `planner/` | Triage (decides which stages are needed), planner (subtask DAG with acceptance criteria), plan validation |
| `executor/` | The agent loop: model ⇄ tools, loop detection, change tracking, step/time budgets |
| `tester/` | Validation-command detection, runners, output parsers (pytest, unittest, jest/vitest, go, cargo, tsc, ruff/flake8, mypy, eslint), failure classification, test history |
| `debugger/` | Structured failure records (error, context, causes, evidence, hypothesis, change, result), attempted-fix fingerprints, loop detection |
| `reviewer/` | Deterministic diff checks + independent model review with structured verdicts |
| `gates/` | Configurable quality gates and verdict computation (`PASSED / FAILED / SKIPPED / UNVERIFIED`) |
| `tasks/` | SQLite task store, state machine, leases/heartbeats, queue, schedules, crash recovery |
| `git/` | Git wrapper, non-destructive snapshot checkpoints (temporary index + `commit-tree` + private refs), branch/commit helpers |
| `orchestrator/` | The end-to-end pipeline and final engineering report |
| `parallel/` | Dependency-graph job runner with worker limit, timeouts, cancellation, resource locks |
| `observability/` | Structured JSONL tracing, metrics, redacting log handler, debug mode |
| `improvement/` | Post-task retrospective and improvement suggestions (never self-modifying) |
| `benchmark/` | Benchmark runner, fixtures, metrics, report |
| `ui/` | CLI (`aie`) and web dashboard (Starlette + Server-Sent Events) |
| `prompts/` | Role system prompts as editable Markdown files |

---

## 3. Model abstraction

### 3.1 Canonical format

All conversations use provider-neutral types (`core/types.py`):

- `Message(role: user|assistant, content: list[Block])`
- Blocks: `TextBlock`, `ToolUseBlock(id, name, input)`, `ToolResultBlock(tool_use_id, name, content, is_error)`,
  `OpaqueBlock(provider, model, payload)`.
- `OpaqueBlock` carries provider-specific data that must be echoed back unchanged to the
  *same* provider and model (e.g. Anthropic thinking blocks). Adapters drop opaque
  blocks produced by a different provider/model. Blocks may also carry a `meta` dict
  for per-provider round-trip data (e.g. Gemini `thoughtSignature`).

### 3.2 `ModelProvider` interface (`models/base.py`)

```python
class ModelProvider(ABC):
    async def generate(req: ModelRequest) -> ModelResponse
    async def stream(req) -> AsyncIterator[StreamEvent]        # default: wraps generate
    async def structured_output(req, schema) -> StructuredResult  # default: prompt + validate + repair
    async def tool_call(req) -> ModelResponse                  # generate with tools required
    async def embeddings(texts, model=None) -> list[list[float]]  # CapabilityNotSupported if absent
    async def health_check() -> HealthStatus
```

Providers raise a small error taxonomy (`ProviderError` →
`RetryableProviderError`, `RateLimitError`, `AuthenticationError`,
`ContextLengthError`, `InvalidRequestError`, `ProviderUnavailableError`, `RefusalError`),
so retry and fallback logic is uniform.

Models without native tool calling are wrapped by `PromptedToolCalling`, which
describes tools in the system prompt and parses `<tool_call>` JSON blocks.

### 3.3 Routing and fallback (`models/router.py`)

Config maps *roles* to ordered model chains:

```toml
[models.roles]
planner    = ["anthropic:<model-id>", "openai:<model-id>"]
coder      = ["anthropic:<model-id>", "ollama:<model-id>"]
reviewer   = ["openai:<model-id>"]          # independent reviewer where possible
fast       = ["ollama:<model-id>"]
classifier = ["ollama:<model-id>"]
```

`ModelRouter.for_role(role)` returns a `FallbackModel` which, per request:
1. tries each candidate whose circuit breaker is closed;
2. retries transient failures with exponential backoff + jitter (bounded);
3. on non-recoverable failure or refusal, emits `MODEL_FALLBACK` and moves on;
4. raises `AllModelsFailedError` when the chain is exhausted — the orchestrator
   then persists state as `BLOCKED` (resumable) rather than losing work.

Because history is canonical, switching provider mid-task needs no state
translation beyond what each adapter already does per request.

No model identifiers are hard-coded; you configure them. `aie providers models`
lists models reported by each provider's own model-listing endpoint.

---

## 4. Tool system and safety

### 4.1 Tools

Each tool declares: name, description, pydantic input model (→ JSON schema),
minimum `PermissionLevel`, side-effect class (`none/read/write/execute/network`),
default timeout. The `ToolExecutor` performs, in order:

1. schema validation (errors returned to the model as tool errors);
2. policy evaluation → `ALLOW`, `ASK` or `DENY` (mode, permission ceiling, command
   risk, protected paths, user allow/deny patterns);
3. approval via an `ApprovalBroker` (interactive CLI prompt, web queue, or
   non-interactive policy) when `ASK`;
4. execution with timeout and cooperative cancellation;
5. output truncation (head + tail) and **secret redaction**;
6. audit log entry + `TOOL_CALLED`/`TOOL_RESULT` events + metrics.

Built-in tools: `read_file`, `write_file`, `edit_file`, `delete_file`,
`list_directory`, `find_files`, `search_text`, `code_search`, `repo_overview`,
`find_symbol`, `find_dependents`, `related_tests`, `run_command`, `process_list`,
`process_output`, `process_stop`, `run_tests`, `run_linter`, `run_formatter`,
`run_build`, `run_typecheck`, `git_status`, `git_diff`, `git_log`, `git_branch`,
`git_commit`, `git_checkout`, `web_search`, `web_fetch`, `browser` (Playwright,
optional), `db_schema`, `db_query`, `environment_info`, `memory_search`,
`memory_record`, plus the stage-control tools `submit_work` and `submit_answer`.
The full generated reference is in [docs/TOOL_GUIDE.md](docs/TOOL_GUIDE.md).
During pipeline runs the orchestrator owns branches and commits, so the model is
not offered `git_commit`, `git_branch` or `git_checkout`.

### 4.2 Permission levels and operating modes

| Level | Allows |
|---|---|
| `READ_ONLY` | reading, searching, read-only git, read-only commands |
| `SAFE_WRITE` | + file writes inside the workspace |
| `DEVELOPMENT` | + low/medium-risk commands, tests, builds, package installs, commits, branches |
| `PRIVILEGED` | + high-risk commands (always with approval) |

| Mode | Ceiling | Behaviour |
|---|---|---|
| `safe` | READ_ONLY | analysis and recommendations only |
| `assisted` | DEVELOPMENT | user approves every write and command beyond read-only |
| `developer` | DEVELOPMENT | normal development actions automatic; high risk asks |
| `autonomous` | DEVELOPMENT | runs a plan with minimal intervention; high risk asks if a human is attached, otherwise denied |

`CRITICAL` commands (disk formatting, recursive deletion of `/` or home, fork bombs,
privilege escalation tricks, …) are always denied. The agent never attempts to
bypass operating-system security.

### 4.3 Command risk classification (`security/command_risk.py`)

Shell commands are tokenised (POSIX `shlex`; Windows-aware heuristics), split on
`&& || ; |` and command substitutions, and each segment is classified; the highest
risk wins. Examples: `ls`, `git status`, `pytest` → LOW; `pip install`, migrations,
unknown programs, redirection into the workspace → MEDIUM; `rm -r`,
`git reset --hard`, `git push`, `curl … | sh`, `DROP TABLE`, `sudo` → HIGH;
`rm -rf /`, `mkfs`, `dd of=/dev/…` → CRITICAL.

### 4.4 Other controls

- **Path guard:** every path is resolved (symlinks included) and must stay inside
  the workspace; `.git/` internals and `.agent/` state are not writable by tools;
  secret-like files (`.env`, keys) are protected.
- **Secret handling:** pattern + entropy detection; values of secret-like
  environment variables are redacted verbatim; redaction is applied to tool
  output before the model sees it, to logs, events, memory, and reports;
  staged diffs are scanned before any agent commit.
- **Child-process environment:** provider API keys and secret-like variables are
  stripped from command environments unless explicitly allowed.
- **Stale-write protection:** overwriting a file requires that the agent has read
  its current version; if it changed on disk since, the write is refused.
- **Optional container sandbox:** commands can run inside a Docker container with
  the workspace mounted and networking disabled (`[terminal] sandbox = "docker"`).
- **Audit log:** `.agent/logs/audit.jsonl`, redacted, append-only.

---

## 5. Repository intelligence (`repo/`)

- **Discovery** (deterministic, cached in `.agent/project.json`): languages by
  file count/bytes, frameworks (from manifests and dependencies), package
  managers (from lockfiles), entry points, test layout and commands, lint /
  format / typecheck / build commands, config files, docs, generated files,
  potential secrets (paths only).
- **File listing** uses `git ls-files` when available (respects `.gitignore`),
  otherwise a walker with standard ignore rules.
- **Incremental index** (`.agent/indexes/index.db`): per-file mtime/size/hash,
  symbols, imports; only changed files are re-parsed.
- **Symbols:** Python via `ast`; JS/TS, Go, Rust, Java via conservative regexes.
- **Import graph:** module → file resolution for Python and relative JS/TS
  imports, Go packages; answers "what depends on X" (transitively) and
  "which tests cover X" (import edges + naming conventions).
- **Search:** ripgrep when installed (Python fallback), filename globbing, and
  BM25 (SQLite FTS5) over file chunks for natural-language retrieval.

---

## 6. Agent pipeline (`orchestrator/`)

```
TASK
 └─ UNDERSTAND   triage: question vs change, complexity, requirements, acceptance
 │               criteria, which stages are needed, blocking questions only
 └─ INSPECT      deterministic discovery + environment inspection
 └─ PLAN         planner model → validated subtask DAG (skipped for trivial tasks)
 └─ for each subtask (topological order, sequential):
 │    CHECKPOINT (pre)
 │    IMPLEMENT  agent loop (implementer role) → submit_work
 │    VALIDATE   targeted tests → affected suite; lint/typecheck changed files
 │    REPAIR     debugger role with structured failure records (bounded loop,
 │               duplicate-fix and repeated-failure detection)
 │    REVIEW     deterministic checks + independent reviewer model → FIX loop
 │    GATES      per-subtask verdict
 │    CHECKPOINT (post) [+ commit on agent branch when configured & safe]
 └─ FINAL QA     full validation, security scan of total diff, final review
 │               against original requirements, docs check, git-state check
 └─ REPORT       engineering report (Markdown + JSON) in .agent/reports/
 └─ RETROSPECT   improvement suggestions (advisory only)
```

**Routing:** triage decides which stages run. A question ("where is auth?") runs a
read-only research loop with cited evidence and skips planning, validation and
review. Trivial and small changes skip the planner. Security and performance flags
(set by the model's triage and by deterministic keyword detection — which can only
add stages, never remove them) give the independent review focused, in-depth
instructions; a dependency audit runs only when dependency manifests changed; the
docs stage runs only when the change affects documented behaviour.

**Loops are bounded:** `max_steps` per agent loop, `max_repair_iterations`,
`max_review_iterations`, wall-clock budgets, and a loop detector for repeated
identical tool calls, repeated identical fixes, and repeated failure signatures.

**Quality gates** (`gates/`): `requirements`, `implementation`, `typecheck`,
`lint`, `tests`, `build`, `security`, `review`, `docs`, `git_state`. Each is
`required`, `if_available` or `disabled`. Final status:

- `COMPLETED` — every required gate `PASSED` (if-available gates passed or skipped);
- `COMPLETED_UNVERIFIED` — work done but some required gate could not run;
- `FAILED`, `BLOCKED` (needs a human/model), `CANCELLED`, `INTERRUPTED`.

---

## 7. Persistence, checkpoints and recovery

```
<project>/.agent/
  config.toml          project configuration (user-editable)
  .gitignore           "*": keeps all agent state out of your repository
  state.db             SQLite (WAL): tasks, pipeline state, events, checkpoints,
                       test runs, failure records, metrics, schedules, control
  memory.db            project memory (FTS5)
  project.json         repository profile (generated)
  context.json         project context snapshot (generated from memory)
  tasks.json           task snapshot (generated)
  decisions.json       decision log snapshot (generated)
  checkpoints/         file-backup checkpoints (non-git workspaces)
  indexes/index.db     repository index
  logs/                trace-<session>.jsonl, audit.jsonl, agent.log
  reports/             <task>.md / .json reports, <task>.improvements.json
  artifacts/           browser screenshots
```

- **Checkpoints (git):** the working tree (tracked + untracked, honouring
  `.gitignore`) is snapshotted through a *temporary index* and `git commit-tree`,
  stored under `refs/ai-engineer/checkpoints/<id>`. The user's index, branch and
  working tree are untouched. Restoring first snapshots the current state, so a
  restore is itself reversible.
- **Checkpoints (no git):** original contents of every file the agent touches are
  backed up before the first write; restore writes them back and removes
  agent-created files.
- **Leases:** a running task holds a lease (pid, host, heartbeat). On start-up,
  tasks whose lease expired are marked `INTERRUPTED`.
- **Resume:** reload pipeline state → diff working tree against the last verified
  checkpoint → re-run validation for the in-flight subtask (never assume it
  completed) → continue, or roll back on request.

---

## 8. Memory (`memory/`)

Layers: `session`, `project`, `engineering` (global, cross-project), `command`,
`decision`. Each item has content, kind, source, confidence, version history,
tags, file references with content hashes, and timestamps. Search uses SQLite
FTS5 (BM25) with layer filters. When a referenced file's hash changes, the item
is flagged **stale** and presented as such. Memory is always presented to models
as *hints*; current repository state wins.

---

## 9. Observability

Every significant action emits a typed event (`TASK_STARTED`, `PLAN_CREATED`,
`TOOL_CALLED`, `TOOL_RESULT`, `FILE_CHANGED`, `TEST_STARTED`, `TEST_FAILED`,
`FIX_ATTEMPTED`, `TEST_PASSED`, `REVIEW_STARTED`, `CHECKPOINT_CREATED`,
`MODEL_FALLBACK`, `APPROVAL_REQUIRED`, `TASK_COMPLETED`, …) on an in-process bus.
Subscribers: redacting JSONL trace writer, SQLite event log, metrics aggregator,
CLI renderer, web SSE stream. Metrics include task and stage duration, model
latency and token usage (when providers report it), tool latency, retries,
fallbacks, failures, test iterations. `--debug` adds request/response summaries
(still redacted). Hidden model reasoning is never displayed; the UI shows action
summaries and evidence.

---

## 10. Concurrency

Async I/O throughout (`asyncio`). The job-graph runner (`parallel/graph.py`)
executes independent jobs with dependencies, a concurrency limit, per-job
timeouts, cancellation and named exclusive resources. Concretely:

- validation: lint and type checks run concurrently; tests and builds share an
  exclusive `workspace-writers` resource, so they never overlap;
- the model's independent read-only tool calls within one turn run concurrently
  (`ToolExecutor.execute_many`); calls with side effects run in order;
- repository indexing reads and parses files in a thread pool, with
  single-threaded database writes.

Subtasks that modify the workspace run sequentially — concurrent writers to one
working tree are unsafe.

---

## 11. Offline operation

Works offline: repository analysis, indexing, file editing, terminal, git,
tests, memory, checkpoints, reports, the web dashboard, and local models
(Ollama or any OpenAI-compatible local server). Requires network: hosted model
providers, `web_search`/`web_fetch`/`browser` on remote sites, dependency
installation, vulnerability databases.

---

## 12. Cross-platform

Paths via `pathlib`; shell selection per OS (`/bin/sh -c` or `cmd.exe /c`);
process-tree termination (`killpg` on POSIX, `taskkill /T` on Windows);
platform-appropriate config/data directories; CRLF preservation on edits.
CI runs the test suite on Linux, macOS and Windows.

---

## 13. Known limitations

- Agent capability is bounded by the connected model. The harness verifies and
  constrains; it cannot make a weak model strong.
- Provider adapters are verified against documented wire formats with mocked
  HTTP; not every live service has been exercised (`aie providers test`).
- Subtasks run sequentially on one working tree (no worktree-parallel execution).
- Symbol extraction for non-Python languages is regex-based, not full AST;
  import resolution covers Python, JS/TS, Go, Rust, Java/Kotlin and C includes.
- Isolation beyond the path guard and command policy requires Docker
  (`[terminal] sandbox = "docker"`), which is implemented but less tested than
  the host runner.
- Web search needs a configured backend (SearXNG, Brave or Tavily).
- The task budget, step budget and context management are heuristics tuned for
  typical repositories; very large changes should be split into several tasks.
