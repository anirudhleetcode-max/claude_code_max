# Troubleshooting

Start with `aie doctor`: environment, configuration, detected validation commands
(and whether their executables exist), and provider health.

| Symptom | Cause and fix |
|---|---|
| `Configuration error: no model configured for role 'default'` | Set `AIE_MODEL=provider:model` or `[models.roles] default = [...]`. `aie providers models` lists ids. |
| `environment variable X is not set` / provider shows `AuthenticationError` | Export the key or put it in the project `.env`; check `api_key_env` in config. |
| `cannot connect` / `ProviderUnavailableError` | Network or local server down. For Ollama: `ollama serve`; check `OLLAMA_HOST`. Configure a fallback model in the role chain. |
| Task ends `BLOCKED: no model available ...` | Every model in the chain failed. Fix the provider, then `aie resume <id>` — progress is kept. |
| Task ends `COMPLETED_UNVERIFIED` | Something required could not be checked. The report's *Unverified items* section says what: no tests, pre-existing failures, a missing linter, or the review model was unavailable. Add tests or configure commands in `[validation]`. |
| Gate `tests: UNVERIFIED — no test command detected` | Set `[validation] test_command`. `aie inspect` shows what was detected. |
| Tests fail with `No module named pytest` (or similar) | The validation command runs in your environment. Install the project's dev dependencies, or point `test_command` at the right interpreter (e.g. `.venv/bin/python -m pytest`). |
| `permission denied: requires PRIVILEGED ...` or `not approved` | The action is high risk. Approve it interactively, run with a human attached, or add a precise `allow_commands` pattern. In `safe` mode nothing beyond reading is allowed. |
| `... has not been read in this session; read it before overwriting` | Stale-write protection working as intended; the agent re-reads and retries. |
| `... changed on disk since it was last read` | Someone (you, an editor, a formatter) changed the file during the task; the agent re-reads it. |
| Agent repeats itself / `LOOP_DETECTED` | The model is stuck. Try a stronger model for `coder`/`debugger`, a more specific task description, or smaller tasks. |
| `context budget reached; continuing in a fresh conversation` | Normal for long tasks: the agent continues with a progress summary. Frequent resets suggest splitting the task or a larger `context_window`. |
| `task ... is being run by another process` | Another `aie` process holds the task lease. If that process is gone, the lease expires after 60 s (same host: immediately on next start). |
| Agent left a branch `aie/...` | Expected when the tree was clean at start: review and merge it, or delete it. Your original branch was never modified. |
| Want to undo the agent's work | `aie checkpoints list --task <id>` then `aie checkpoints restore <id>` (it snapshots first, so it is reversible), or delete the task branch. |
| Web UI says unauthorized | Open the exact URL printed by `aie ui` (with `?token=`), or set `AIE_WEB_TOKEN`. |
| Windows: commands behave differently | Commands run through `cmd.exe`; configure `[terminal] shell` (e.g. a PowerShell or Git Bash path) and use explicit `[validation]` commands. |
| Slow on huge repositories | The first index build reads every file once; later refreshes are incremental. Exclude build output via `.gitignore`. Disable `validation.full_suite_per_subtask` if the full suite is very slow. |
| Something else | Re-run with `--debug -v`; inspect `.agent/logs/trace-<session>.jsonl`, `.agent/logs/agent.log` and `aie tasks events <id>`. All logs are redacted, so they are safe to share. |
