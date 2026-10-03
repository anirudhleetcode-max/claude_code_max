# Agent guide

How to work with AI Engineer day to day, and what it does on your behalf.

## Quick start

```bash
cd your/project
aie init                      # creates .agent/, detects the project, builds the index
export AIE_MODEL=anthropic:<model-id>      # or openai:/google:/ollama:/local:<model-id>
aie doctor                    # environment, validation commands, provider health
aie run "Add rate limiting to the /login endpoint with tests"
```

Questions about the code (read-only):

```bash
aie ask "Where is authentication handled and which tests cover it?"
```

Plan without executing, then run the plan later:

```bash
aie plan "Migrate the settings module to pydantic v2"
aie resume <task-id>
```

## What happens during `aie run`

| Stage | What the agent does | What you see |
|---|---|---|
| Understand | Restates the task; extracts requirements, acceptance criteria and assumptions; decides which extra reviews are needed; asks *only* genuinely blocking questions | `✎ change · medium: ...` |
| Inspect | Deterministic repository discovery, incremental index, **baseline checks** (tests/lint/typecheck before any change) | `· Running baseline checks` |
| Plan | For medium/large tasks: a validated subtask DAG with acceptance criteria (trivial/small tasks skip the planner) | `☰ Plan with N subtask(s)` |
| Git setup | If the working tree is clean: a task branch `aie/<slug>-<id>`; otherwise changes stay uncommitted so they never mix with yours. A start checkpoint is taken either way | `⚑ checkpoint ...` |
| Per subtask | checkpoint → implement (agent loop) → targeted tests → full suite → lint/typecheck → repair loop on new failures → independent review → fix loop → gates → checkpoint (+ commit on the task branch) | `→ Reading ...`, `✘ pytest: 1 failed`, `🔧 Fix attempt 1`, `🔍 Review: approve` |
| Final QA | Full validation (tests, lint, typecheck, build), security scan of the whole diff, dependency audit when manifests changed, final review against the original requirements, docs check, git-state check | `☑ Quality gates: COMPLETED` |
| Report | `.agent/reports/<task>.md` and `.json`: requirements coverage, changes, commands actually run, gates, review findings, failure records, checkpoints, metrics, unverified items | `📄 Report written to ...` |

The agent's own claims ("tests pass", "fixed") are recorded but never trusted:
changed files come from checkpoint diffs and results from commands the
orchestrator runs itself.

### Final statuses

| Status | Meaning | Exit code |
|---|---|---|
| `COMPLETED` | every required gate ran and passed | 0 |
| `COMPLETED_UNVERIFIED` | work done, but something required could not be verified (no tests, pre-existing failures, model review unavailable, ...) | 2 |
| `FAILED` | a gate ran and failed | 1 |
| `BLOCKED` | needs outside help (no model reachable, plan rejected, merge in progress, time budget) — resumable | 3 |
| `INTERRUPTED` | stopped (Ctrl+C, `aie tasks stop`, crash) — resumable | 3 |
| `CANCELLED` | cancelled on purpose | 130 |

## Stopping, resuming and rolling back

- **Ctrl+C** stops after the current step; state is saved. A second Ctrl+C exits immediately.
- `aie tasks stop <id>` / `aie tasks cancel <id>` from another terminal (or the dashboard).
- `aie resume [<id>]` continues from the last persisted stage. An interrupted
  subtask whose changes are already on disk is **re-validated**, never assumed done.
- `aie checkpoints list --task <id>` / `aie checkpoints diff <id>` /
  `aie checkpoints restore <id>`. Every restore first snapshots the current state,
  so it can be undone with the checkpoint id it prints.
- If the agent process crashes, the next `aie` command notices the dead lease and
  marks the task `INTERRUPTED`.

## Approvals and questions

In `developer` mode the agent works without prompts until an action is high risk
(e.g. `git push`, `rm -r`, system package installs, destructive SQL). You then see:

```
⚠ Approval required (HIGH): Running: git push origin aie/fix-login
  Reason: high-risk action requires approval
  Approve? [y]es / [n]o / [a]lways this session:
```

Declining can include a note, which is passed to the agent ("use a dry run
instead"). Non-interactive runs (`--no-interactive`, CI, daemon) deny anything
that needs approval. Pre-approve routine commands with `[permissions] allow_commands`.

## Queues, schedules and the daemon

```bash
aie tasks add "Update dependencies and fix deprecations" --priority 10
aie schedule add "Run the dependency audit and fix advisories" --every 1d
aie daemon               # runs due schedules and queued tasks, one at a time
aie queue                # process the queue once and exit
```

## Web dashboard

```bash
pip install 'ai-engineer[web]'
aie ui                   # prints http://127.0.0.1:8765/?token=...
```

Shows tasks, live activity, plan and subtasks, tests, failures, gates,
checkpoints with diffs, approvals and questions (answerable in the browser),
reports, and stop/cancel/resume buttons. It binds to localhost and requires the
printed token.

## Memory

`aie memory search <query>`, `aie memory add "We use pnpm, never npm" --layer project --kind convention`,
`aie memory forget <id>`, `aie memory history <id>`. See [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md).

## Improving results

- Write task descriptions with concrete acceptance criteria.
- Put project conventions in `AGENTS.md` or `CLAUDE.md` at the repository root:
  they are included in every model request.
- Configure validation commands if detection is wrong (`aie doctor` shows them).
- Use a strong model for `coder`/`planner` and, ideally, a different model for `reviewer`.
- Read `aie improve` for retrospective suggestions after tasks.

## Command reference (generated from the code)

<!-- BEGIN GENERATED: cli -->
| Command | Description |
|---|---|
| `aie init` | create .agent/, detect the project and build the index |
| `aie run` | run an engineering task end to end |
| `aie ask` | answer a question about the repository (read-only) |
| `aie plan` | understand and plan a task without executing it |
| `aie resume` | resume an interrupted, blocked or planned task |
| `aie status` | recent tasks |
| `aie tasks` | manage tasks and the queue |
| `aie queue` | run queued tasks (and due schedules) once |
| `aie daemon` | continuously run queued tasks and recurring schedules |
| `aie report` | print a task's engineering report |
| `aie checkpoints` | list, diff or restore checkpoints |
| `aie memory` | search and edit agent memory |
| `aie providers` | list providers, their models, or test a role |
| `aie doctor` | check environment, configuration and provider health |
| `aie inspect` | show the detected repository profile |
| `aie index` | build or refresh the repository index |
| `aie config` | show effective configuration |
| `aie improve` | show retrospective improvement suggestions |
| `aie schedule` | recurring maintenance tasks |
| `aie bench` | run the benchmark suite |
| `aie ui` | start the web dashboard |
<!-- END GENERATED: cli -->
