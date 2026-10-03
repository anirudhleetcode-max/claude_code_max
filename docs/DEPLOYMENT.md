# Deployment

AI Engineer is a local tool; "deployment" means choosing where the agent runs and
how it is isolated.

## Options

| Setup | When | Notes |
|---|---|---|
| Your workstation (`scripts/install.sh`) | interactive development | `developer` or `assisted` mode; approvals in the terminal or dashboard |
| Agent in a container (`Dockerfile`) | untrusted repositories or models, shared machines | the container only sees the mounted project; add `--network none` for offline local models reachable via a socket/host network as needed |
| Docker command sandbox (`[terminal] sandbox = "docker"`) | agent on the host, commands isolated | each command runs in `docker_image` with the workspace mounted at `/workspace`, `--network=none` by default, as your uid |
| Server / daemon (`aie daemon`) | queued and recurring maintenance tasks | runs one task at a time; non-interactive, so anything needing approval is denied unless pre-approved with `allow_commands` |
| CI job | scheduled maintenance, benchmarks | use `--no-interactive`, `--mode autonomous`, secrets from the CI secret store; exit code reflects the final status |

### Container

```bash
docker build -t ai-engineer .
docker run --rm -it -v "$PWD:/workspace" \
  -e ANTHROPIC_API_KEY -e AIE_MODEL ai-engineer run "Fix the failing tests"
# dashboard
docker run --rm -it -p 8765:8765 -v "$PWD:/workspace" -e AIE_WEB_TOKEN=change-me \
  -e AIE_MODEL -e ANTHROPIC_API_KEY ai-engineer ui --host 0.0.0.0
```

The image contains Python, git and ripgrep only; add your project's toolchain
(Node, Go, JDK, ...) in a derived image so the agent can run your tests.

### Daemon as a service (systemd example)

```ini
[Unit]
Description=AI Engineer daemon for /srv/project

[Service]
WorkingDirectory=/srv/project
EnvironmentFile=/etc/ai-engineer.env
ExecStart=/home/agent/.ai-engineer/venv/bin/aie daemon --mode autonomous
Restart=on-failure
User=agent

[Install]
WantedBy=multi-user.target
```

If the daemon crashes, the next start detects the dead lease, marks the task
`INTERRUPTED`; resume it with `aie resume <id>`.

## Web dashboard exposure

The dashboard can approve actions, so treat its token like a password. It binds to
127.0.0.1 by default; when exposing it, put it behind TLS (a reverse proxy) and set
a strong `AIE_WEB_TOKEN`.

## Backup and restore

| What | Where | Contains |
|---|---|---|
| Project state | `<project>/.agent/` | config, `state.db` (tasks, events, checkpoints metadata, test history, failures, schedules, metrics), `memory.db`, reports, logs, file-backup checkpoints, index (rebuildable) |
| Git checkpoints | `refs/ai-engineer/checkpoints/*` in the project's `.git` | snapshot commits |
| Global state | `$AIE_HOME` or the platform data directory | `engineering-memory.db`, global `config.toml` |

Back up with the agent stopped (SQLite WAL files must be copied with their
databases), e.g. `tar czf agent-backup.tgz .agent ~/.local/share/ai-engineer`.
Git checkpoint refs travel with the repository's `.git` (they are not pushed by
default). The index can always be deleted and rebuilt with `aie index`.
To prune old checkpoint refs: `git for-each-ref refs/ai-engineer/checkpoints --format='%(refname)'`
and `git update-ref -d <ref>`.
