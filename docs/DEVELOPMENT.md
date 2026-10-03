# Development

Working on AI Engineer itself.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,all]"
pytest -q && ruff check src tests && mypy && python scripts/gen_docs.py --check
```

## Layout

See the package table in [ARCHITECTURE.md](../ARCHITECTURE.md). Entry points:

- `ui/cli.py` — the `aie` command; every command goes through `runtime.Runtime`.
- `runtime.py` — builds all services for a workspace from configuration.
- `orchestrator/pipeline.py` — the end-to-end stage machine.
- `executor/loop.py` — the model ⇄ tools loop.
- `tools/executor.py` — validation, policy, approval, execution, redaction, audit.
- `models/router.py` — role routing, retries, circuit breakers, fallback.

## Conventions

- Python 3.11+, `from __future__ import annotations`, full type hints; mypy and
  ruff must be clean (`pyproject.toml` holds the configuration).
- No provider-specific logic outside `providers/`. No model ids in the code.
- Deterministic work (parsing, detection, classification, diffing) is code, not
  model calls. Add a test for every rule.
- Never log or return secrets: use `ctx.redactor` / `runtime.redactor` for
  anything that may contain tool output.
- Persisted formats carry a schema version; refuse newer versions, migrate older ones.
- Tools raise `ToolError` for expected failures; the message is shown to the model,
  so make it actionable.
- Keep conversations append-only (never edit earlier turns); reset with a new
  conversation instead.

## Testing approach

- Unit tests for every module (`tests/unit/`), provider adapters with
  `httpx.MockTransport` (`httpx2.MockTransport` for the Anthropic SDK).
- End-to-end pipeline tests with `ScriptedProvider` on real git repositories and
  real test commands (`tests/integration/helpers.py`).
- Adversarial tests for hostile models and environments (`tests/adversarial/`).
- When fixing a bug, add a regression test and confirm it fails without the fix.
- `python scripts/gen_docs.py` regenerates reference sections of the docs; a test
  fails when they drift from the code.

## Changing the agent safely

The retrospective (`aie improve`) produces suggestions; it never modifies the
agent. Changes to prompts (`src/ai_engineer/prompts/*.md`), policies or stages go
through the same process as any code change: tests, review, CI, and
`aie bench run --suite harness` (must stay 10/10). Measure behavioural changes
with `aie bench run --suite model` before and after.

## Releasing

1. Update `CHANGELOG.md` and the version in `pyproject.toml` and
   `src/ai_engineer/__init__.py`.
2. Run the full checks and the harness benchmark.
3. Tag the release; build with `python -m build` if publishing a wheel.
