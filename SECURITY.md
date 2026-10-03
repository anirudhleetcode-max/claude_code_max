# Security

AI Engineer runs model-generated actions on your machine. Its controls assume the
model may be wrong, confused, or manipulated by prompt injection (instructions
hidden in code, issues, web pages or tool output). The harness enforces limits
**no matter what the model asks for**; `tests/adversarial/` contains a scripted
"compromised model" that tries destructive commands, secret exfiltration and
writes outside the workspace, and verifies they are blocked.

## Threat model

| Threat | Controls |
|---|---|
| Destructive commands (`rm -rf ~`, `mkfs`, `git push --force`, `DROP TABLE`) | command risk classifier; CRITICAL always denied; HIGH needs approval (denied when no human is attached); `deny_commands`; optional Docker sandbox |
| Writes outside the project / to git internals / to secrets | path guard resolves symlinks and confines writes to the workspace; protected paths (`.git` at any depth, `.agent/**`, `.env*`, keys; matched case-insensitively) |
| Secret exfiltration via model context, logs or commits | secrets redacted from tool output before the model sees it, from events, traces, audit log, memory and reports; secret files (`.env`, keys, credentials) appear only in redacted form in reads, search results and diffs; the agent never commits secret or protected files and scans staged diffs first; provider keys and secret-like variables stripped from child processes |
| Exfiltration via network | commands that upload data (`curl -d @file`, `-X POST`, `scp`, `ssh`, `nc`) are HIGH risk; `web_fetch` blocks private addresses, cloud metadata endpoints and URLs with embedded credentials (every redirect hop is checked before it is requested); the browser never opens metadata endpoints; domain allow/block lists |
| Clobbering human work | stale-write protection (must read the current version before writing; refuses if the file changed since); auto-commit only on a fresh task branch when the tree was clean; never switches branches with uncommitted changes; checkpoints before every subtask; restores are reversible |
| Runaway loops / cost | step, time, repair and review budgets; repeated-call and repeated-failure detection; bounded retries with backoff; circuit breakers |
| Web dashboard abuse | binds to 127.0.0.1 by default; random access token (or `AIE_WEB_TOKEN`), HttpOnly SameSite=Strict cookie; cross-origin POSTs rejected |

The agent never attempts to bypass operating-system security; privilege
escalation (`sudo`, `su`, `doas`, `runas`) is HIGH risk and requires approval.

## Command risk levels

| Level | Examples | Policy |
|---|---|---|
| LOW (read-only) | `ls`, `cat`, `git status`, `rg` | allowed in every mode, including `safe` |
| LOW (executes code) | `pytest`, `npm test`, `cargo build`, `ruff check` | allowed from `developer` level |
| MEDIUM | package installs, migrations, unknown programs, writes into the workspace, `python script.py` | allowed in `developer`/`autonomous`; asks in `assisted` |
| HIGH | `rm -r`, `git reset --hard`, `git push`, `curl … \| sh`, system package managers, destructive SQL, cloud/cluster changes, credential tools | approval required |
| CRITICAL | `rm -rf /` or `~`, `mkfs`, `dd of=/dev/…`, fork bombs, `kill -9 -1`, recursive permission changes on system directories | never allowed |

Compound commands (`&&`, `;`, `|`), command substitution (`$(…)`, backticks),
wrappers (`sudo`, `env`, `timeout`, `xargs`, `sh -c`), redirections and Windows
syntax are analysed; the highest risk of any part wins; anything unparseable is
treated as at least MEDIUM. The classifier is a policy aid, **not a sandbox**: for
untrusted repositories or models use `[terminal] sandbox = "docker"` and/or run
the agent itself in a container (see DEPLOYMENT.md).

## Secret handling

- Detection: provider keys (Anthropic, OpenAI, Google, AWS, GitHub, GitLab, Slack,
  Stripe, npm), private keys, JWTs, credentials in URLs, bearer tokens, and
  `password/secret/token/api_key = <literal>` assignments (literals only — code such
  as `token = get_token()` is not flagged).
- Values of environment variables with secret-like names are redacted verbatim
  wherever they appear.
- Repository discovery reports *locations* of potential secrets (path, line, kind),
  never values.
- Credentials are only read from the environment (`api_key_env`), never stored in
  configuration.

## Audit trail

`.agent/logs/audit.jsonl` records every tool call and denial (redacted, rotated at
10 MB). `.agent/logs/trace-<session>.jsonl` records every event. Reports list the
commands that were actually executed.

## Limitations

- Without the Docker sandbox, allowed commands (including the project's own tests)
  run with your user's permissions and network access.
- Static security rules and secret patterns are heuristics; they reduce, not
  eliminate, risk.
- Prompt injection can still mislead the model into producing wrong code; the
  review and gates catch many such cases but not all. Review agent branches before
  merging.

## Reporting vulnerabilities

Please report security issues privately to the repository owner rather than in
public issues. Include steps to reproduce and the affected version.
