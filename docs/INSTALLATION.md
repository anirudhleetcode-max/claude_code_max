# Installation

## Requirements

| Component | Required | Notes |
|---|---|---|
| Python | 3.11 or newer | CI runs 3.11 and 3.12 on Linux, macOS and Windows; 3.13 has been used for installs |
| git | strongly recommended | branches, snapshot checkpoints and commits; without it, file-backup checkpoints are used |
| ripgrep (`rg`) | optional | faster text search (a Python fallback is built in) |
| Docker | optional | only for `[terminal] sandbox = "docker"` |
| A model | yes, to run tasks | hosted API key, or a local Ollama / OpenAI-compatible server |

Optional extras: `anthropic` (official Anthropic SDK), `web` (dashboard:
starlette + uvicorn), `browser` (Playwright), `all`, `dev` (tests and linters).

## Production / personal install

```bash
git clone <repository-url> ai-engineer
cd ai-engineer
scripts/install.sh                 # creates ~/.ai-engineer/venv and links ~/.local/bin/aie
# options: --prefix DIR  --extras all|anthropic,web|none  --bin-dir DIR  --source PATH_OR_URL
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1   # -Prefix, -Extras, -Source
```

Manual alternative:

```bash
python -m venv ~/.ai-engineer/venv
~/.ai-engineer/venv/bin/pip install "/path/to/ai-engineer[all]"
```

For the browser tool also run `playwright install chromium`.

## Development install

```bash
git clone <repository-url> ai-engineer && cd ai-engineer
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"
pytest -q && ruff check src tests && mypy
```

## Configure a model

Put keys in the environment or a project `.env` file (see `.env.example`), then
choose models:

```bash
export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY / GEMINI_API_KEY
aie providers models                    # lists model ids each provider reports
export AIE_MODEL=anthropic:<model-id>,ollama:<model-id>   # primary, then fallback
aie providers test                      # one real request through the default role
```

Fully offline: run Ollama (`ollama serve`, `ollama pull <model>`) or another
local server, set `AIE_MODEL=ollama:<model>` (or `OPENAI_COMPATIBLE_BASE_URL` +
`AIE_MODEL=local:<model>`). For local models without native tool calling set
`native_tools = false` for that model (see CONFIGURATION.md).

## Verify the installation

```bash
aie --version
cd some/project && aie init && aie doctor
aie bench run --suite harness            # offline self-test of the whole pipeline
```

## Upgrading

```bash
cd ai-engineer && git pull
~/.ai-engineer/venv/bin/pip install --upgrade ".[all]"      # or re-run scripts/install.sh
```

State databases carry a schema version. Opening state written by a *newer*
version is refused with a clear error rather than risking corruption; older state
is used as-is (the repository index rebuilds itself automatically when its schema
changes). Back up `.agent/` and the global directory before major upgrades
(see DEPLOYMENT.md).

## Uninstall

Remove `~/.ai-engineer` (or your `--prefix`), the `aie` link, the global data
directory (`~/.local/share/ai-engineer` / `~/.config/ai-engineer` /
`%APPDATA%\ai-engineer`), and any project `.agent/` directories you no longer need.
