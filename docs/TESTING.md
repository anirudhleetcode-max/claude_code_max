# Testing

This document covers (1) how the agent validates *your* project, and (2) how
AI Engineer itself is tested and benchmarked.

## 1. The validation engine (testing your project)

`src/ai_engineer/tester/` detects, runs and parses validation commands.

**Detection** (`detect.py`) uses evidence from the repository profile: pytest
configuration or test files → `python -m pytest`; `package.json` scripts with the
detected package manager (`npm`/`pnpm`/`yarn`/`bun`); `go test ./...`;
`cargo test`; Maven/Gradle; Makefile targets; ruff/flake8/eslint/mypy/pyright/tsc
configuration. A command is only suggested when evidence exists; unavailable
executables are reported (`aie doctor`), never assumed. Format checks always use
check mode (`ruff format --check`, `black --check`, `prettier --check`, ...).
Override anything in `[validation]`.

**Scopes:** after a subtask the orchestrator runs *targeted* tests (tests related
to the changed files via the import graph and naming conventions, plus changed
test files), then the full suite, lint on changed files and the type checker.
Final QA runs the full suite, lint, type check and build. Baselines are taken
before the first change.

**Parsing** (`parsers.py`): pytest, unittest, jest, vitest, go test, cargo test,
mypy, tsc, ruff, flake8, pylint, eslint, gofmt, black, prettier, and generic
output. Results are structured (`CheckResult`: status, counts, failures with
test id/file/line/message/type, diagnostics, classification, summary, output
tail) and fingerprinted (`signature()`) so repeated identical failures are
detected. Classifications: `test_failure`, `collection_error`, `import_error`,
`missing_dependency`, `syntax_error`, `type_errors`, `lint_errors`,
`build_error`, `timeout`, `no_tests`, `command_not_found`, `environment`,
`vulnerabilities`.

**Repair loop:** each failing validation produces a failure record (error,
context, likely causes, evidence, hypothesis, change, result) stored in the
state database. The debugger model sees the record and every previous attempt;
identical fixes, repeated identical failures and oscillation stop the loop.
Environmental failures (missing tools) are reported, not "fixed" in code.

Test history is kept in `.agent/state.db` (`test_runs` table) and shown in reports
and the dashboard.

## 2. Testing AI Engineer itself

```bash
pip install -e ".[dev,all]"
pytest -q                     # unit + integration + adversarial
ruff check src tests && mypy  # lint and types
python scripts/gen_docs.py --check
```

| Suite | Location | What it covers |
|---|---|---|
| Unit | `tests/unit/` | config, events, router (retry/backoff/breaker/fallback/refusal/cancellation), structured output repair, prompted tool calling, provider wire formats (mocked HTTP, incl. streaming and errors), command risk classifier (100+ cases), secrets and redaction, path guard, static rules, tool executor and every built-in tool, git snapshots, state store, checkpoints, gates, reviewer, planner, debugger, agent loop, memory, job graph, repository intelligence, validation engine and parsers, web dashboard, generated docs |
| Integration | `tests/integration/` | full pipeline runs on real git repositories with real pytest runs: bug fix with repair, unfixable bug, question answering, provider fallback, all models down then resume, interrupted subtask re-verification, benchmark negative controls, audit regressions (one test per bug found by the final audits, each shown to fail on the pre-fix code) |
| Adversarial | `tests/adversarial/` | compromised model (destructive commands, exfiltration, protected writes), malformed model output, hanging tests, concurrent human edits, network loss, crashed process recovery, huge repositories, failing commands |

Models are simulated with `ScriptedProvider` (role-aware transcripts). These
tests verify the harness; they cannot say anything about a real model's quality.

CI (`.github/workflows/ci.yml`) runs lint, type checks and the full suite on
Linux, macOS and Windows with Python 3.11 and 3.12, plus the harness benchmark.

## 3. Benchmarks

```bash
aie bench list
aie bench run --suite harness                     # offline, deterministic
aie bench run --suite model --model anthropic:<id> # real model
```

Ten cases, one per category: repository exploration, bug fixing, feature
implementation, refactoring, test generation (with a mutation check), debugging,
documentation, dependency/API upgrade, security remediation, multi-file
architecture change. Each case builds a fixture repository, runs a full agent
task in `autonomous` mode, then runs **hidden verification** the agent never saw.

Metrics: completion rate, verified completion rate, test success, retries,
fallbacks, failure recoveries, human interventions, duration, model and tool
calls, tokens. Results go to `benchmark-results/<suite>-results.{json,md}`.

The `harness` suite replays fixed transcripts and must always verify 10/10 —
it is a regression test of the infrastructure, **not** a capability score.
Negative controls in `tests/integration/test_benchmark.py` prove that false
claims, wrong answers and insecure fixes are scored as unverified. Only the
`model` suite measures an actual model.
