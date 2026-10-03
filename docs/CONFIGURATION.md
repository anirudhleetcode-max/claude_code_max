# Configuration

Configuration is layered; later layers override earlier ones:

1. Built-in defaults (below).
2. Global file: `~/.config/ai-engineer/config.toml` on Linux,
   `~/Library/Application Support/ai-engineer/config.toml` on macOS,
   `%APPDATA%\ai-engineer\config.toml` on Windows (or `$AIE_HOME/config.toml`).
3. Project file: `<project>/.agent/config.toml` (created by `aie init`).
4. Environment variables: `AIE__SECTION__KEY=value` (JSON values are parsed), plus
   the shortcuts `AIE_MODEL`, `AIE_REVIEW_MODEL`, `AIE_FAST_MODEL`, `AIE_MODE`.
5. Command-line flags (`--mode`, `--model`, `--review-model`, `--max-steps`, `--debug`).

`aie config show` prints the effective configuration; `aie config path` shows the files.
A `.env` file in the project root is loaded at startup (existing environment
variables win). **Credentials never go in config files**: providers name the
environment variable that holds their key (`api_key_env`).

## Models

```toml
[models.providers.anthropic]          # auto-registered when ANTHROPIC_API_KEY is set
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.providers.local]              # any OpenAI-compatible server
type = "openai_compatible"
base_url = "http://localhost:8000/v1"

[models.providers.ollama]             # always registered (default http://localhost:11434)
type = "ollama"
defaults = { params = { num_ctx = 32768 } }

[models.roles]                        # ordered fallback chains of provider:model
default    = ["anthropic:<model-id>", "local:<model-id>"]
reviewer   = ["openai:<model-id>"]    # an independent reviewer where possible
classifier = ["ollama:<model-id>"]    # cheap triage
```

Roles: `default`, `planner`, `coder`, `debugger`, `reviewer`, `fast`,
`classifier`, `summarizer`, `embeddings`. Unset roles inherit:
`planner`, `debugger`, `reviewer` → `coder` → `default`; `classifier`,
`summarizer` → `fast` → `default`. Model ids are never hard-coded; list the ids
your providers offer with `aie providers models`.

Per-model options (`[models.providers.<name>.models."<model-id>"]` or
`defaults = {...}` for the whole provider):

| Option | Default | Meaning |
|---|---|---|
| `max_output_tokens` | 16000 | output cap per request |
| `context_window` | 128000 | used to size the agent's context budget |
| `temperature` | unset | only sent when set (some models reject sampling parameters) |
| `native_tools` | true | false = describe tools in the prompt and parse `<tool_call>` blocks (for local models without tool calling) |
| `structured_output` | `"prompt"` | `"native"` uses the provider's JSON-schema mode where supported |
| `params` | `{}` | provider-specific pass-through (e.g. Anthropic `effort`, `thinking`; Ollama `num_ctx`) |

Provider `options`:
- `anthropic`: `server_side_fallback` (bool, default false; enables the API's
  server-side refusal fallback and disables itself for models that reject it),
  `prompt_caching` (default true), `stream_threshold_tokens` (requests with a
  larger `max_tokens` are streamed; default 16000).
- `openai` / `openai_compatible`: `max_tokens_field` (`max_completion_tokens` for
  OpenAI, `max_tokens` for compatible servers), `system_role`, `embedding_model`.
- `ollama`, `google`: `embedding_model`.
- `scripted`: `script_file` (JSON transcript; for demos and tests only).

## Operating modes and permissions

| Mode | Behaviour |
|---|---|
| `safe` | read-only analysis and answers; nothing is written or executed beyond read-only commands |
| `assisted` | every write and command beyond read-only needs your approval; multi-step plans need approval |
| `developer` (default) | normal development actions are automatic; high-risk actions ask |
| `autonomous` | runs the plan with minimal intervention: clarifying questions become recorded assumptions; high-risk actions still ask when a human is attached (terminal or web dashboard) and are denied otherwise — pre-approve routine ones with `allow_commands` |

```toml
[permissions]
mode = "developer"
max_level = "PRIVILEGED"             # set "DEVELOPMENT" to deny high-risk actions outright
allow_commands = ["npm run e2e*"]    # fnmatch patterns auto-approved (never CRITICAL)
deny_commands = ["*prod*"]           # always denied
protected_paths = [".git/**", ".agent/**", ".env", ...]
```

## Quality gates

Each gate is `required`, `if_available` or `disabled`. A task is `COMPLETED` only
when every required gate passed; a required gate that could not run makes the task
`COMPLETED_UNVERIFIED`; any gate that ran and failed makes it `FAILED`.
Failures already present before the agent started are reported as
`PRE-EXISTING` (and the task as `COMPLETED_UNVERIFIED`) — unless they are in tests
related to the files the agent changed, which keeps bug-fix tasks honest.

## Validation commands

Detected automatically from the repository (`aie inspect` shows what was found).
Override when detection is wrong or your environment differs:

```toml
[validation]
test_command = "python -m pytest -q"
lint_command = "ruff check ."
typecheck_command = ""        # empty string disables the check
```

## Defaults (generated from the code)

<!-- BEGIN GENERATED: settings -->
```toml
[permissions]
mode = "developer"
max_level = "PRIVILEGED"
allow_commands = []
deny_commands = []
protected_paths = [".git/**", ".agent/**", ".env", ".env.*", "**/*.pem", "**/*.key", "**/id_rsa*", "**/id_ed25519*", "**/.ssh/**"]
secret_files = [".env", ".env.*", "**/*.pem", "**/*.key", "**/id_rsa*", "**/credentials*"]
approval_timeout_s = 900.0

[terminal]
shell = "" # unset
default_timeout_s = 300.0
max_timeout_s = 3600.0
max_output_chars = 30000
strip_secret_env = true
env_allow = []
sandbox = "none"
docker_image = "python:3.11-slim"
docker_network = "none"

[agent]
max_steps = 60
max_repair_iterations = 4
max_review_iterations = 2
max_subtasks = 12
task_time_budget_s = 14400.0
on_questions = "ask"
stop_on_subtask_failure = false
context_fill_ratio = 0.6
tool_result_max_chars = 20000
repeated_call_limit = 3

[gates]
requirements = "required"
implementation = "required"
typecheck = "if_available"
lint = "if_available"
tests = "required"
build = "if_available"
security = "required"
review = "required"
docs = "if_available"
git_state = "required"

[git]
checkpoints = true
auto_commit = "if_clean"
branch_prefix = "aie/"
commit_prefix = "aie: "
author_name = "" # unset
author_email = "" # unset

[validation]
test_command = "" # unset
lint_command = "" # unset
format_command = "" # unset
typecheck_command = "" # unset
build_command = "" # unset
test_timeout_s = 1200.0
baseline = true
full_suite_per_subtask = true
run_full_suite_at_end = true
dependency_audit = true

[memory]
global_dir = "" # unset
max_items_in_context = 8

[web]
search_backend = "none"
searxng_url = "" # unset
api_key_env = "" # unset
fetch_max_bytes = 2000000
allow_domains = []
block_domains = []
timeout_s = 30.0
browser_executable = "" # unset

[ui]
host = "127.0.0.1"
port = 8765

[observability]
debug = false
trace = true
log_level = "WARNING"

[models]
request_timeout_s = 600.0

[models.retry]
max_attempts = 3
base_delay_s = 1.0
max_delay_s = 30.0
jitter = 0.25

[models.circuit_breaker]
failure_threshold = 3
reset_after_s = 60.0

```
<!-- END GENERATED: settings -->
