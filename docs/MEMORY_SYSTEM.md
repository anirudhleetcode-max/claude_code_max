# Memory system

Memory helps the agent carry knowledge between steps, tasks and projects. It is
always presented to models as **hints**: "verify against the repository;
repository state wins". It never overrides what the agent can observe.

## Stores and layers

| Layer | Store | Contents | Written by |
|---|---|---|---|
| `session` | project | short-lived notes for the current session | agent |
| `project` | project | facts and conventions about this repository; research notes from `web_fetch` (with source URL and retrieval time) | agent (`memory_record`), you (`aie memory add`) |
| `decision` | project | architecture decisions (ADR-style: decision, rationale, alternatives, task) | planner, agent |
| `command` | project | validation commands with success/failure counts, last outcome and duration | orchestrator (every validation run) |
| `engineering` | **global** | lessons that apply across projects | agent (`memory_record kind=lesson`), you |

The project store is `<project>/.agent/memory.db`; the global store is
`engineering-memory.db` in the global data directory (`$AIE_HOME` or the platform
default). `aie` writes a readable snapshot to `.agent/context.json` and
`.agent/decisions.json` after each task.

## Item model

Each item has: `layer`, `kind`, optional `key`, `content`, `source`, `confidence`
(0–1), `tags`, `file_refs` (file → content hash at write time), `meta`, `version`,
`lineage`, timestamps, and `active`.

- **Versioned:** updating an item (or adding one with the same `layer/kind/key`)
  creates a new version in the same lineage; `aie memory history <id>` shows all.
  Adding identical content only refreshes it.
- **Source- and confidence-aware:** search ranks by relevance, then confidence,
  then recency. Command memories gain confidence with each success.
- **Staleness:** if any referenced file's current hash differs from the stored one
  (or the file is gone), the item is marked `STALE` when read and ranked after
  fresh items; prompts show the stale files explicitly.
- **Redacted:** content, source, key, tags and metadata pass through secret
  redaction before storage.
- **Searchable:** SQLite FTS5 (BM25) with a LIKE fallback when FTS5 is unavailable.

## How the agent uses it

- The context builder adds the most relevant memory items to subtask and planning
  prompts, marked as hints.
- Tools: `memory_search` (read-only) and `memory_record` (kinds: fact, decision,
  lesson, convention, known_issue; optional file references for staleness).
- The orchestrator records every validation command outcome, and plan decisions.

## Editing memory

```bash
aie memory list --layer project
aie memory search "test command"
aie memory add "Integration tests need docker compose up db first" --kind known_issue
aie memory forget <id>            # deactivates the whole lineage
aie memory history <id>
```

Project conventions that should *always* apply are better placed in `AGENTS.md`
or `CLAUDE.md` at the repository root; those files are included in every request.
