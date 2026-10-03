# AI Engineer

A self-hosted, model-agnostic autonomous software-engineering agent.

AI Engineer takes a task ("fix the pagination bug", "add rate limiting with tests",
"where is authentication handled?"), understands it, inspects the repository,
plans, edits code, runs the project's own tests and linters, diagnoses and repairs
failures, has the change independently reviewed, and only reports **COMPLETED**
when configured quality gates actually ran and passed. Everything it does is
checkpointed, reversible, logged and resumable.

It is a *harness*, not a model. You connect the intelligence — Anthropic, OpenAI,
Google Gemini, Ollama, or any OpenAI-compatible server (vLLM, llama.cpp, LM Studio,
hosted gateways) — and can switch, combine or fall back between them without
changing the agent. It does not contain or replicate any model's weights; its
results depend on the model you connect.

```
$ aie run "Fix the add function in calc so the tests pass"
  ✎ change · small: Fix add() so it returns the sum
  ✘ baseline pytest: 1 failed, 1 passed (0.3s)
  ◆ Subtask s1: Fix add() so it returns the sum
    → Reading calc/__init__.py
    ✏ edited calc/__init__.py
  ✘ pytest: 1 failed, 1 passed (0.3s)
  ⚠ Diagnosing: pytest: 1 failed, 1 passed
  🔧 Fix attempt 1: root cause: add used the wrong operator
  ✔ pytest: 2 passed (0.4s)
  🔍 Review: approve (0 blocking)
  ⎇ Committed s1 as 17f4c519fe
  ☑ Quality gates: COMPLETED
  📄 Report written to .agent/reports/task_….md
```

## Highlights

- **Verification, not claims.** Baseline checks before any change; targeted then
  full test runs after each step; structured failure records and a bounded repair
  loop; deterministic + independent model review; configurable quality gates with
  `PASSED / FAILED / SKIPPED / UNVERIFIED / PRE-EXISTING` results. Anything that
  could not be checked is reported as unverified.
- **Model-agnostic routing.** Per-role fallback chains (planner, coder, debugger,
  reviewer, fast, classifier) with retries, exponential backoff and circuit
  breakers. Prompt-based tool calling for local models without native tool support.
- **Safety by default.** Workspace path jail, permission levels and operating
  modes, a shell-command risk classifier (compound commands, substitutions,
  wrappers, Windows syntax), approvals, secret detection and redaction everywhere,
  secret-free child environments, stale-write protection, append-only audit log,
  optional Docker sandbox.
- **Repository intelligence.** Deterministic discovery (languages, frameworks,
  package managers, entry points, test/lint/typecheck/build commands, generated
  files, secrets), incremental index with symbols, import graph, dependents,
  related tests and BM25 code search.
- **Survives interruption.** SQLite state with leases; every stage is persisted;
  crashes are detected; `aie resume` re-verifies interrupted work instead of
  assuming it finished. Git snapshot checkpoints (your index and branch are never
  touched) or file backups; every restore is itself reversible.
- **Memory.** Session, project, cross-project engineering, command and decision
  memory with full-text search, versioning, confidence and staleness detection —
  presented to models as hints; the repository always wins.
- **Observability.** Typed events, redacted JSONL traces, metrics (latency, tokens,
  retries, fallbacks, tool use, test iterations), engineering reports, terminal UI
  and a web dashboard with live activity, approvals and diffs.
- **Benchmarks.** Ten task categories with hidden verification, runnable offline
  (harness suite) or against your models (model suite).

## Install

```bash
git clone <this repository> ai-engineer && cd ai-engineer
scripts/install.sh            # Windows: powershell -File scripts\install.ps1
```

Requires Python 3.11+ (tested on 3.11–3.13) and, for git features, `git`.
Details: [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Use

```bash
cd your/project
aie init
export AIE_MODEL=anthropic:<model-id>     # list ids: aie providers models
aie doctor
aie run "Add input validation to the signup endpoint, with tests"
aie ask "Which modules depend on the payment client?"
aie ui                                    # web dashboard (pip install 'ai-engineer[web]')
```

Modes: `--mode safe | assisted | developer | autonomous`. See
[docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md).

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | design, components, data flow |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | development and production setup, upgrades |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | every setting, modes, gates, providers |
| [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) | using the agent, stages, statuses, commands |
| [docs/MODEL_PROVIDERS.md](docs/MODEL_PROVIDERS.md) | providers, routing, fallback, adding a provider |
| [docs/TOOL_GUIDE.md](docs/TOOL_GUIDE.md) | tools, permissions, adding a tool |
| [docs/MEMORY_SYSTEM.md](docs/MEMORY_SYSTEM.md) | memory layers, staleness, editing |
| [docs/TESTING.md](docs/TESTING.md) | the validation engine and this project's test suite, benchmarks |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | contributing to AI Engineer itself |
| [SECURITY.md](SECURITY.md) | threat model and controls |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | servers, containers, CI usage, backup |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | common problems |
| [CHANGELOG.md](CHANGELOG.md) | releases |

## Status and limitations

Version 0.1.0. The harness is covered by unit, integration, adversarial and
benchmark tests on Linux, macOS and Windows (CI). Provider adapters are tested
against their documented wire formats with mocked HTTP; they have **not** yet
been exercised against every live service — run `aie providers test` with your
credentials. Real-world task success depends on the connected model; measure it
with `aie bench run --suite model --model <provider:id>`. See *Known limitations*
in [ARCHITECTURE.md](ARCHITECTURE.md).
