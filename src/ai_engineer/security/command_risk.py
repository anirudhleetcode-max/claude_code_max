"""Shell command risk classification.

A command string is split into segments on unquoted ``; && || | &`` and newlines;
command substitutions are classified recursively; each segment is classified by
program and arguments. The highest risk wins. Unknown programs are MEDIUM.

This is a policy aid, not a sandbox: it errs towards higher risk when unsure.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import PurePosixPath, PureWindowsPath


class Risk(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class CommandAssessment:
    command: str
    risk: Risk = Risk.LOW
    read_only: bool = True
    reasons: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    workspace: str | None = field(default=None, repr=False)

    def bump(self, risk: Risk, reason: str, read_only: bool = False) -> None:
        if risk > self.risk:
            self.risk = risk
        if not read_only:
            self.read_only = False
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def merge(self, other: CommandAssessment) -> None:
        self.risk = max(self.risk, other.risk)
        self.read_only = self.read_only and other.read_only
        for r in other.reasons:
            if r not in self.reasons:
                self.reasons.append(r)
        self.programs.extend(p for p in other.programs if p not in self.programs)

    def summary(self) -> str:
        why = "; ".join(self.reasons) if self.reasons else "no risk indicators"
        return f"{self.risk.name} ({why})"


# ---- program tables --------------------------------------------------------------------

READ_ONLY_PROGRAMS = {
    "ls", "dir", "pwd", "cd", "echo", "printf", "cat", "head", "tail", "less", "more", "wc", "grep", "egrep",
    "fgrep", "rg", "ag", "ack", "fd", "tree", "stat", "file", "which", "where", "whereis", "type", "uname",
    "hostname", "whoami", "id", "date", "df", "du", "ps", "diff", "cmp", "sort", "uniq", "cut", "tr", "jq",
    "yq", "basename", "dirname", "realpath", "readlink", "true", "false", "test", "[", "sleep", "seq",
    "md5sum", "sha1sum", "sha256sum", "shasum", "xxd", "od", "hexdump", "column", "nl", "comm", "paste",
    "fold", "fmt", "expand", "unexpand", "rev", "strings", "nproc", "free", "uptime", "lscpu", "lsb_release",
    "arch", "getconf", "locale", "tput", "clear", "history", "findstr", "ver", "systeminfo", "get-childitem",
    "get-content", "get-location", "select-string", "awk", "gawk", "sed", "find", "lsof", "pgrep", "top",
    "htop", "man", "help", "info", "cloc", "tokei", "wc.exe",
}

ANALYZERS = {
    "pytest", "py.test", "tox", "nox", "jest", "vitest", "mocha", "ava", "karma", "playwright", "cypress",
    "ruff", "flake8", "pylint", "mypy", "pyright", "pyre", "bandit", "black", "isort", "autopep8", "yapf",
    "eslint", "prettier", "tsc", "tslint", "stylelint", "biome", "golangci-lint", "staticcheck", "gofmt",
    "goimports", "rustfmt", "clippy-driver", "shellcheck", "hadolint", "markdownlint", "yamllint",
    "pip-audit", "safety", "semgrep", "govulncheck", "phpunit", "rspec", "rubocop", "ctest", "coverage",
}

BUILD_TOOLS = {"make", "cmake", "ninja", "bazel", "buck", "mvn", "gradle", "gradlew", "./gradlew", "ant", "dotnet", "msbuild", "meson", "scons", "just", "task"}

SYSTEM_PACKAGE_MANAGERS = {"apt", "apt-get", "aptitude", "dpkg", "yum", "dnf", "rpm", "pacman", "brew", "port", "snap", "flatpak", "choco", "winget", "scoop", "apk", "zypper", "emerge", "nix-env"}

PRIVILEGE = {"sudo", "doas", "su", "runas", "pkexec"}

CRITICAL_PROGRAMS = {
    "mkfs", "fdisk", "sfdisk", "cfdisk", "parted", "wipefs", "diskpart", "format", "mkswap", "fsck",
    "format-volume", "clear-disk", "initialize-disk", "remove-partition",
}

HIGH_PROGRAMS = {
    "shutdown", "reboot", "halt", "poweroff", "mount", "umount", "useradd", "userdel", "usermod", "groupadd",
    "groupdel", "passwd", "chpasswd", "visudo", "ssh-keygen", "ssh-add", "ssh-copy-id", "shred", "srm",
    "iptables", "ufw", "firewall-cmd", "netsh", "nc", "ncat", "netcat", "telnet", "ftp", "sftp", "scp",
    "rsync", "ssh", "launchctl", "crontab", "at", "insmod", "rmmod", "modprobe", "sysctl", "setenforce",
    "security", "keychain", "certutil", "update-alternatives", "chattr", "eval", "set-executionpolicy",
    "reg", "bcdedit", "takeown", "icacls", "cipher", "vssadmin", "wmic",
    "dropdb", "dropuser", "swapoff", "swapon", "scutil", "nvram", "csrutil", "spctl", "pmset", "networksetup",
    "setx", "stop-computer", "restart-computer", "set-itemproperty", "remove-itemproperty",
}

NETWORK_PROGRAMS = {"curl", "wget", "http", "https", "aria2c", "invoke-webrequest", "iwr", "invoke-restmethod", "irm"}

INTERPRETERS = {"python", "python3", "python2", "py", "node", "deno", "bun", "ruby", "perl", "php", "lua", "julia", "rscript"}
SHELLS = {"sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh", "pwsh", "powershell", "cmd"}
WRAPPERS = {"time", "nice", "nohup", "command", "exec", "builtin", "stdbuf", "timeout", "env", "xargs", "ionice", "unbuffer", "busybox", "toybox"}

# Credential / secret files: deleting or overwriting them from the shell needs approval.
_SENSITIVE_NAME = re.compile(
    r"(?i)(^|[/\\])(\.env(\.(?!example$|sample$|template$|dist$)[^/\\]+)?|[^/\\]*\.(pem|key|p12|pfx|keystore|jks)|id_(rsa|dsa|ecdsa|ed25519)[^/\\]*|credentials[^/\\]*|\.netrc|\.pgpass|\.npmrc|\.pypirc)$"
)


def _sensitive(path: str) -> bool:
    return bool(_SENSITIVE_NAME.search(path.strip().strip("\"'")))

SAFE_SCRIPT_NAMES = {"test", "tests", "lint", "build", "typecheck", "type-check", "check", "format", "fmt", "compile", "coverage", "ci"}

_FORK_BOMB = re.compile(r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:\s*&")
_DANGEROUS_SQL = re.compile(
    r"(?i)\b(drop\s+(database|schema|table)|truncate\s+(table\s+)?\w|delete\s+from\s+\w+\s*(;|$|\"|')|flushall|flushdb|dropdatabase\s*\(|\.drop\s*\(\s*\))"
)
_SYSTEM_DIRS = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/var", "/opt", "/sys", "/proc", "/dev", "/root", "/home", "/System", "/Library", "/Applications")


def _norm_program(token: str) -> str:
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def split_segments(command: str) -> tuple[list[str], list[str]]:
    """Split on unquoted separators. Returns (segments, substitutions)."""
    segments: list[str] = []
    subs: list[str] = []
    buf: list[str] = []
    i, n = 0, len(command)
    quote: str | None = None
    while i < n:
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i : i + 2])
                i += 2
                continue
            if quote == '"' and command.startswith("$(", i):
                depth, j = 1, i + 2
                while j < n and depth:
                    depth += {"(": 1, ")": -1}.get(command[j], 0)
                    j += 1
                subs.append(command[i + 2 : j - 1])
                buf.append(command[i:j])
                i = j
                continue
            if ch == quote:
                quote = None
            buf.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(command[i : i + 2])
            i += 2
            continue
        if command.startswith("$(", i):
            depth, j = 1, i + 2
            while j < n and depth:
                depth += {"(": 1, ")": -1}.get(command[j], 0)
                j += 1
            subs.append(command[i + 2 : j - 1])
            buf.append(command[i:j])
            i = j
            continue
        if ch == "`":
            j = command.find("`", i + 1)
            j = n if j == -1 else j
            subs.append(command[i + 1 : j])
            buf.append(command[i : j + 1])
            i = j + 1
            continue
        two = command[i : i + 2]
        if two in ("&&", "||"):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in (";", "|", "\n") or (ch == "&" and not (i > 0 and command[i - 1] in "<>") and command[i + 1 : i + 2] != ">"):
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()], subs


def _tokenize(segment: str, windows: bool) -> list[str] | None:
    try:
        return shlex.split(segment, posix=not windows)
    except ValueError:
        return None


class _Context:
    def __init__(self, workspace: str | None, windows: bool) -> None:
        self.workspace = os.path.normpath(workspace) if workspace else None
        self.windows = windows
        self.home = os.path.expanduser("~")

    def outside_workspace(self, path: str) -> bool:
        if not path or path.startswith("-") or path in ("/dev/null", "nul", "NUL"):
            return False
        expanded = path.replace("$HOME", self.home).replace("${HOME}", self.home)
        if expanded.startswith("~"):
            return True
        is_abs = PureWindowsPath(expanded).is_absolute() if self.windows else PurePosixPath(expanded).is_absolute()
        if not is_abs:
            # relative paths escaping with ../ are treated as outside
            parts = expanded.replace("\\", "/").split("/")
            depth = 0
            for part in parts:
                if part == "..":
                    depth -= 1
                    if depth < 0:
                        return True
                elif part not in ("", "."):
                    depth += 1
            return False
        if self.workspace is None:
            return True
        norm = os.path.normpath(expanded)
        try:
            return os.path.commonpath([norm, self.workspace]) != self.workspace
        except ValueError:
            return True

    def catastrophic_target(self, path: str) -> bool:
        p = path.strip().rstrip("/\\") or "/"
        p = p.replace("${HOME}", "$HOME")
        if p in ("/", "/*", "~", "~/*", "$HOME", "$HOME/*", "*", ".", "..", "C:", "C:\\*", "c:", "/.", "./*" ):
            return p not in (".", "./*", "*") or self.workspace is None
        if self.windows and re.fullmatch(r"[A-Za-z]:\\?\*?", p):
            return True
        return any(p == d or p == d + "/*" for d in _SYSTEM_DIRS)


def _redirect_targets(tokens: list[str]) -> list[str]:
    targets: list[str] = []
    for i, tok in enumerate(tokens):
        m = re.fullmatch(r"(\d|&)?>>?(.*)", tok)
        if m:
            target = m.group(2)
            if not target and i + 1 < len(tokens):
                target = tokens[i + 1]
            if target and not target.startswith("&"):
                targets.append(target)
    return targets


def _strip_redirects(tokens: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        m = re.fullmatch(r"(\d|&)?(>>?|<)(.*)", tok)
        if m:
            if not m.group(3):
                skip = True
            continue
        out.append(tok)
    return out


def _skip_options(args: list[str], value_opts: set[str]) -> list[str]:
    """Drop a wrapper's own leading options (and their values), keep the wrapped command verbatim."""
    i = 0
    while i < len(args) and args[i].startswith("-") and args[i] != "-":
        if args[i] == "--":
            return args[i + 1 :]
        i += 2 if args[i] in value_opts else 1
    return args[i:]


_WRAPPER_VALUE_OPTS: dict[str, set[str]] = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U"},
    "doas": {"-u", "-C"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p"},
    "xargs": {"-n", "-I", "-L", "-P", "-d", "-E", "-s", "-a", "--max-args", "--max-procs", "--delimiter"},
    "env": {"-u", "--unset", "-C", "--chdir", "-S"},
    "stdbuf": {"-i", "-o", "-e"},
}


def _classify_git(args: list[str], a: CommandAssessment) -> None:
    head = args[: next((i for i, x in enumerate(args) if not x.startswith("-")), len(args))]
    if any(x in ("-c", "--config-env") or x.startswith(("--exec-path", "--config-env=")) for x in head):
        # -c core.pager=..., diff.external=..., --exec-path=...: even read-only commands can run code
        a.bump(Risk.MEDIUM, "git configuration override can run arbitrary programs")
    if any(x == "--output" or x.startswith("--output=") for x in args):
        a.bump(Risk.MEDIUM, "git writes its output to a file")
    sub = next((x for x in args if not x.startswith("-") and x not in _git_option_values(args)), "")
    rest = args[args.index(sub) + 1 :] if sub in args else []
    flags = set(rest)
    read_only = {
        "status", "diff", "log", "show", "blame", "rev-parse", "ls-files", "ls-tree", "describe", "shortlog",
        "grep", "reflog", "cat-file", "rev-list", "name-rev", "whatchanged", "count-objects", "check-ignore",
        "merge-base", "show-ref", "for-each-ref", "help", "version", "--version",
    }
    if sub in read_only and not (sub == "reflog" and rest[:1] in (["expire"], ["delete"])):
        return
    if sub == "branch" and (not rest or flags <= {"-a", "-r", "--list", "-v", "-vv", "--all", "--show-current", "--merged", "--no-merged", "--contains"}):
        return
    if sub == "remote" and (not rest or rest[:1] in (["-v"], ["show"], ["get-url"])):
        return
    if sub == "tag" and (not rest or rest[:1] in (["-l"], ["--list"])):
        return
    if sub == "stash" and rest[:1] in (["list"], ["show"]):
        return
    if sub == "config" and (flags & {"--get", "--list", "-l", "--get-all", "--get-regexp"}) and not any("credential" in r for r in rest):
        return
    if sub == "push":
        reason = "force-pushes to a remote" if flags & {"-f", "--force", "--force-with-lease", "--mirror", "--delete", "-d"} else "pushes to a remote (outward-facing)"
        a.bump(Risk.HIGH, reason)
        return
    if sub == "reset" and "--hard" in flags:
        a.bump(Risk.HIGH, "git reset --hard discards uncommitted work")
        return
    if sub == "clean" and any(f.startswith("-") and "f" in f for f in rest):
        a.bump(Risk.HIGH, "git clean deletes untracked files")
        return
    if sub in ("checkout", "restore") and (
        (("." in rest or "--" in rest or flags & {"-f", "--force"}) and "--staged" not in flags) or flags & {"--worktree", "-W"}
    ):
        a.bump(Risk.HIGH, f"git {sub} can discard working-tree changes")
        return
    if sub == "tag" and flags & {"-d", "--delete", "-f", "--force"}:
        a.bump(Risk.HIGH, "deletes or moves a tag")
        return
    if sub in ("submodule", "worktree") and (flags & {"-f", "--force"} or rest[:1] in (["deinit"], ["remove"])):
        a.bump(Risk.HIGH, f"git {sub} can discard checked-out work")
        return
    if sub == "rm" and "--cached" not in flags:
        a.bump(Risk.HIGH if flags & {"-r", "-rf", "-fr"} else Risk.MEDIUM, "git rm deletes files")
        return
    if sub == "rebase" and "--abort" not in flags:
        a.bump(Risk.HIGH, "git rebase rewrites commit history")
        return
    if sub == "branch" and flags & {"-D", "--delete", "-d", "-M", "-f", "--force"}:
        a.bump(Risk.HIGH, "deletes or force-moves a branch")
        return
    if sub == "stash" and rest[:1] in (["drop"], ["clear"]):
        a.bump(Risk.HIGH, "discards stashed work")
        return
    if sub in ("filter-branch", "filter-repo", "replace", "update-ref", "gc", "prune") or (sub == "reflog" and rest):
        a.bump(Risk.HIGH, f"git {sub} rewrites or deletes history")
        return
    if sub in ("credential", "credential-store", "credential-cache") or (sub == "config" and any("credential" in r for r in rest)):
        a.bump(Risk.HIGH, "touches git credentials")
        return
    if sub in ("merge", "cherry-pick", "revert", "am", "pull") and "--abort" not in flags:
        a.bump(Risk.MEDIUM, f"git {sub} modifies history/working tree")
        return
    a.bump(Risk.MEDIUM, f"git {sub or 'command'} modifies repository state")


def _git_option_values(args: list[str]) -> set[str]:
    """Values of git's global options (``-c name=value``, ``-C dir``) so they are not taken for the subcommand."""
    values = set()
    for i, x in enumerate(args[:-1]):
        if x in ("-c", "-C", "--git-dir", "--work-tree", "--namespace", "--config-env"):
            values.add(args[i + 1])
        elif not x.startswith("-"):
            break
    return values


# Variables that make an otherwise harmless program load libraries, run hooks or pick another binary.
_CODE_LOADING_VARS = re.compile(
    r"^(LD_PRELOAD|LD_LIBRARY_PATH|LD_AUDIT|DYLD_[A-Z_]+|PYTHONSTARTUP|PYTHONPATH|PYTHONHOME|NODE_OPTIONS|NODE_PATH|PERL5OPT|PERL5LIB|"
    r"RUBYOPT|RUBYLIB|BASH_ENV|ENV|PROMPT_COMMAND|PS4|IFS|PAGER|GIT_PAGER|MANPAGER|EDITOR|VISUAL|GIT_EDITOR|GIT_EXTERNAL_DIFF|GIT_SSH|"
    r"GIT_SSH_COMMAND|GIT_ASKPASS|SSH_ASKPASS|GIT_EXEC_PATH|GIT_CONFIG_[A-Z0-9_]+|GIT_CONFIG|GIT_DIR|GIT_WORK_TREE|PATH)="
)


def _check_assignment(assignment: str, a: CommandAssessment) -> None:
    if _CODE_LOADING_VARS.match(assignment):
        a.bump(Risk.MEDIUM, f"sets {assignment.split('=', 1)[0]}, which can make the command run other code")


def _read_only_escape(prog: str, args: list[str], segment: str) -> str:
    """Why a normally read-only program would write files or run commands here ("" if it would not)."""
    if prog == "sed":
        scripts = [x for x in args if not x.startswith("-")][:1] + [args[i + 1] for i, x in enumerate(args[:-1]) if x in ("-e", "--expression")]
        for script in scripts:
            if re.search(r"(^|[;{}\n]|\d|\$|/)\s*[ewWrR]\b|/[gpIiMm0-9]*e[gpIiMm0-9]*\s*($|[;}])", script):
                return "sed script writes files or runs commands"
        return ""
    if prog in ("sort",) and any(x in ("-o",) or x.startswith(("--output", "-o")) for x in args):
        return "sort writes its output to a file"
    if prog == "uniq" and len([x for x in args if not x.startswith("-")]) >= 2:
        return "uniq writes its output to a file"
    if prog in ("awk", "gawk") and re.search(r"system\s*\(|\|\s*\"|\bgetline\b|print[^;}]*>", segment):
        return "awk script writes files or runs commands"
    if prog == "find" and any(x in ("-fprint", "-fprint0", "-fprintf", "-fls", "-exec", "-execdir", "-ok", "-okdir", "-delete") for x in args):
        return "find writes files or runs commands"
    if prog in ("less", "more", "man") and any(x.startswith("+!") or x.startswith("-o") for x in args):
        return f"{prog} writes files or runs commands"
    return ""


def _classify_package_manager(prog: str, args: list[str], a: CommandAssessment) -> bool:
    sub = next((x for x in args if not x.startswith("-")), "")
    if prog in ("npm", "pnpm", "yarn", "bun"):
        if sub in ("publish", "unpublish", "deprecate", "owner", "adduser", "login", "logout", "token"):
            a.bump(Risk.HIGH, f"{prog} {sub} is outward-facing or touches credentials")
        elif sub in ("install", "i", "add", "ci", "remove", "rm", "uninstall", "update", "upgrade", "up", "link", "dedupe", "prune"):
            a.bump(Risk.MEDIUM, f"{prog} {sub} changes dependencies")
        elif sub in ("test", "t") or (sub in ("run", "run-script") and len(args) > 1 and args[args.index(sub) + 1] in SAFE_SCRIPT_NAMES):
            a.bump(Risk.LOW, "")
            a.read_only = False
        elif sub in SAFE_SCRIPT_NAMES and prog != "npm":
            a.bump(Risk.LOW, "")
            a.read_only = False
        elif sub in ("ls", "list", "outdated", "view", "info", "why", "audit", "config", "-v", "--version", "help"):
            return True
        else:
            a.bump(Risk.MEDIUM, f"{prog} {sub or ''} runs project scripts".strip())
        return True
    if prog == "npx" or prog == "pnpx" or (prog == "pnpm" and sub == "dlx") or prog == "bunx":
        a.bump(Risk.MEDIUM, "downloads and executes a package")
        return True
    if prog in ("pip", "pip3", "uv", "poetry", "pipenv", "conda", "mamba", "pdm", "hatch", "rye", "pipx"):
        if sub in ("publish", "upload") or prog == "twine":
            a.bump(Risk.HIGH, f"{prog} {sub} publishes a package")
        elif sub in ("list", "show", "freeze", "check", "--version", "-V", "help", "search", "info", "env") and prog != "uv":
            return True
        elif prog == "uv" and sub == "run":
            rest = args[args.index(sub) + 1 :]
            if rest:
                a.merge(classify_command(shlex.join(rest), a.workspace))
            return True
        else:
            a.bump(Risk.MEDIUM, f"{prog} {sub} changes the environment".strip())
        return True
    if prog == "twine":
        a.bump(Risk.HIGH, "uploads a package")
        return True
    if prog == "cargo":
        if sub in ("publish", "yank", "login", "owner"):
            a.bump(Risk.HIGH, f"cargo {sub} is outward-facing")
        elif sub in ("install", "add", "remove", "update", "uninstall"):
            a.bump(Risk.MEDIUM, f"cargo {sub} changes dependencies")
        else:
            a.bump(Risk.LOW, "")
            a.read_only = False
        return True
    if prog == "go":
        if sub in ("get", "install", "mod", "work", "generate", "run"):
            a.bump(Risk.MEDIUM, f"go {sub} changes dependencies or runs code")
        elif sub in ("version", "env", "list", "doc", "help"):
            return True
        else:
            a.bump(Risk.LOW, "")
            a.read_only = False
        return True
    if prog in ("gem", "bundle", "bundler", "composer"):
        if sub in ("push", "publish", "yank", "signin"):
            a.bump(Risk.HIGH, f"{prog} {sub} is outward-facing")
        else:
            a.bump(Risk.MEDIUM, f"{prog} {sub} changes dependencies".strip())
        return True
    return False


def _classify_db(prog: str, segment: str, a: CommandAssessment) -> None:
    if _DANGEROUS_SQL.search(segment):
        a.bump(Risk.HIGH, "destructive database statement")
    else:
        a.bump(Risk.MEDIUM, f"{prog} accesses a database")


def _classify_segment(tokens: list[str], segment: str, ctx: _Context, a: CommandAssessment, depth: int) -> None:
    for target in _redirect_targets(tokens):
        if re.match(r"/dev/(sd|hd|nvme|disk|mmcblk|xvd|vd)", target):
            a.bump(Risk.CRITICAL, f"writes directly to block device {target}")
        elif target not in ("/dev/null", "nul", "NUL", "/dev/stdout", "/dev/stderr"):
            if _sensitive(target):
                a.bump(Risk.HIGH, f"overwrites a credential/secret file ({target})")
            elif ctx.outside_workspace(target):
                a.bump(Risk.HIGH, f"redirects output outside the workspace ({target})")
            else:
                a.bump(Risk.MEDIUM, f"writes file {target}")
    tokens = _strip_redirects(tokens)
    # subshells, groups and negation: "(rm -rf ~)", "{ rm -rf ~; }", "! cmd" run the inner command
    while tokens and tokens[0] in ("(", "{", "!"):
        tokens = tokens[1:]
    while tokens and tokens[-1] in (")", "}"):
        tokens = tokens[:-1]
    if tokens and tokens[0].startswith("(") and len(tokens[0]) > 1:
        tokens = [tokens[0].lstrip("("), *tokens[1:]]
    if tokens and tokens[-1].endswith(")") and tokens[-1].count(")") > tokens[-1].count("("):
        tokens = [*tokens[:-1], tokens[-1].rstrip(")")]
    tokens = [t for t in tokens if t]
    # strip leading VAR=value assignments and wrappers; some variables make a program load or run other code
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        _check_assignment(tokens[0], a)
        tokens = tokens[1:]
    if not tokens:
        return
    prog = _norm_program(tokens[0])
    args = tokens[1:]
    a.programs.append(prog)

    if prog in PRIVILEGE:
        a.bump(Risk.HIGH, f"privilege escalation via {prog}")
        if prog == "su":
            if "-c" in args:
                inner = " ".join(args[args.index("-c") + 1 :])
                a.merge(classify_command(inner, ctx.workspace, windows=ctx.windows, _depth=depth + 1))
            return
        rest = _skip_options(args, _WRAPPER_VALUE_OPTS.get(prog, set()))
        if rest:
            _classify_segment(rest, shlex.join(rest), ctx, a, depth + 1)
        return
    if prog in WRAPPERS:
        rest = _skip_options(args, _WRAPPER_VALUE_OPTS.get(prog, set()))
        if prog == "timeout":
            rest = rest[1:]  # the duration
        elif prog in ("busybox", "toybox") and not rest:
            return
        elif prog == "env":
            while rest and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", rest[0]):
                _check_assignment(rest[0], a)
                rest = rest[1:]
            if not rest:
                a.bump(Risk.MEDIUM, "prints the environment (may expose secrets)", read_only=True)
                return
        if rest:
            _classify_segment(rest, shlex.join(rest), ctx, a, depth + 1)
        return
    if prog in ("printenv", "set", "export") and not args:
        a.bump(Risk.MEDIUM, "prints the environment (may expose secrets)", read_only=True)
        return
    if prog in CRITICAL_PROGRAMS or prog.startswith("mkfs"):
        if prog == "format" and not (args and re.match(r"^[A-Za-z]:", args[0])):
            a.bump(Risk.MEDIUM, "unknown program 'format'")
            return
        a.bump(Risk.CRITICAL, f"{prog} can destroy disks or filesystems")
        return
    if prog == "dd":
        if any(re.match(r"of=/dev/(sd|hd|nvme|disk|mmcblk|xvd|vd)", x) for x in args):
            a.bump(Risk.CRITICAL, "dd writes to a block device")
        else:
            a.bump(Risk.HIGH, "dd performs raw copies")
        return
    if prog in HIGH_PROGRAMS:
        if prog == "crontab" and args[:1] == ["-l"]:
            return
        a.bump(Risk.HIGH, f"{prog} changes system state, credentials or reaches remote hosts")
        return
    if prog in ("kill", "pkill", "killall", "taskkill", "stop-process"):
        if "-1" in args and ("-9" in args or "-KILL" in args or "-s" in args):
            a.bump(Risk.CRITICAL, "kills every process")
        elif "1" in args or "-u" in args or "--user" in args or (prog in ("killall", "pkill") and "-9" in args and not [x for x in args if not x.startswith("-")][1:]):
            a.bump(Risk.HIGH, f"{prog} targets system or other users' processes")
        else:
            a.bump(Risk.MEDIUM, f"{prog} terminates processes")
        return
    if prog == "defaults" and args[:1] in (["write"], ["delete"], ["import"], ["rename"]):
        a.bump(Risk.HIGH, "changes macOS user or system preferences")
        return
    if prog == "diskutil":
        if args[:1] in (["list"], ["info"], ["activity"]):
            return
        a.bump(Risk.CRITICAL if args and args[0].lower().startswith(("erase", "zero", "partition", "reformat", "secureerase")) else Risk.HIGH, "diskutil changes disks or volumes")
        return
    if prog in ("systemctl", "service", "sc"):
        if args[:1] in (["status"], ["list-units"], ["is-active"], ["query"]):
            return
        a.bump(Risk.HIGH, f"{prog} changes system services")
        return
    if prog in SYSTEM_PACKAGE_MANAGERS:
        sub = next((x for x in args if not x.startswith("-")), "")
        if sub in ("list", "search", "show", "info", "policy", "query", "--version", "-v", "doctor", "config"):
            return
        a.bump(Risk.HIGH, f"{prog} changes system packages")
        return
    if prog in ("rm", "del", "erase", "rd", "rmdir", "remove-item", "ri", "unlink"):
        recursive = any(
            x in ("-r", "-R", "--recursive", "/s", "/S", "-recurse") or (x.startswith("-") and not x.startswith("--") and ("r" in x or "R" in x))
            for x in args
        ) or (prog in ("rd", "rmdir") and any(x.lower() == "/s" for x in args))
        targets = [x for x in args if not x.startswith("-") and not (ctx.windows and x.startswith("/"))]
        if any(ctx.catastrophic_target(t) for t in targets) and (recursive or prog in ("rd", "rmdir")):
            a.bump(Risk.CRITICAL, "recursive deletion of a system, home or root directory")
        elif recursive:
            a.bump(Risk.HIGH, "recursive deletion")
        elif any(_sensitive(t) for t in targets):
            a.bump(Risk.HIGH, "deletes a credential/secret file")
        elif any(ctx.outside_workspace(t) for t in targets):
            a.bump(Risk.HIGH, "deletes files outside the workspace")
        else:
            a.bump(Risk.MEDIUM, "deletes files")
        return
    if prog in ("chmod", "chown", "chgrp", "icacls", "attrib"):
        targets = [x for x in args if not x.startswith("-")]
        recursive = any(x in ("-R", "--recursive") for x in args)
        if recursive and any(ctx.catastrophic_target(t) for t in targets[1:] or targets):
            a.bump(Risk.CRITICAL, "recursive permission change on a system directory")
        elif any(".ssh" in t or ctx.outside_workspace(t) for t in targets[1:]):
            a.bump(Risk.HIGH, "changes permissions outside the workspace")
        elif "777" in args or recursive:
            a.bump(Risk.HIGH, "broad permission change")
        else:
            a.bump(Risk.MEDIUM, "changes file permissions")
        return
    if prog in ("mv", "cp", "copy", "move", "xcopy", "robocopy", "move-item", "copy-item", "ln", "install", "tee", "touch", "mkdir", "md", "new-item", "truncate"):
        targets = [x for x in args if not x.startswith("-")]
        if prog == "robocopy" and any(x.lower() == "/mir" for x in args):
            a.bump(Risk.HIGH, "robocopy /MIR deletes files")
        elif any(_sensitive(t) for t in (targets if prog in ("mv", "move", "move-item", "truncate", "tee") else targets[-1:])):
            a.bump(Risk.HIGH, f"{prog} moves, overwrites or truncates a credential/secret file")
        elif any(ctx.outside_workspace(t) for t in targets[-1:] if prog not in ("tee",)) or (prog == "tee" and any(ctx.outside_workspace(t) for t in targets)):
            a.bump(Risk.HIGH, f"{prog} writes outside the workspace")
        elif prog in ("mv", "move", "move-item") and any(t in ("/dev/null", "nul") for t in targets):
            a.bump(Risk.HIGH, "moves files to the null device (deletion)")
        else:
            a.bump(Risk.MEDIUM, f"{prog} writes files")
        return
    if prog == "git":
        _classify_git(args, a)
        return
    if prog in ("docker", "podman", "docker-compose", "nerdctl"):
        sub = next((x for x in args if not x.startswith("-")), "")
        joined = " ".join(args)
        if sub in ("ps", "images", "logs", "inspect", "version", "info", "stats", "top", "history", "search"):
            return
        if "--privileged" in args or re.search(r"-v\s*/:|--volume[= ]/:|/var/run/docker\.sock", joined):
            a.bump(Risk.HIGH, "container with host-level access")
        elif sub in ("system", "volume", "rm", "rmi", "prune", "kill", "network") and ("prune" in args or sub in ("rm", "rmi", "kill") or "rm" in args):
            a.bump(Risk.HIGH, f"{prog} {sub} deletes containers/images/volumes")
        elif sub in ("push", "login", "logout"):
            a.bump(Risk.HIGH, f"{prog} {sub} is outward-facing")
        else:
            a.bump(Risk.MEDIUM, f"{prog} {sub} runs or builds containers")
        return
    if prog in ("kubectl", "helm", "terraform", "tofu", "pulumi", "aws", "gcloud", "az", "gsutil", "bq", "doctl", "flyctl", "heroku", "vercel", "netlify", "firebase", "eksctl"):
        joined = " ".join(args).lower()
        if re.search(r"\b(get|describe|list|ls|show|plan|version|validate|whoami|logs|status|explain|diff|template|lint)\b", joined) and not re.search(
            r"\b(apply|delete|destroy|create|deploy|rm|remove|scale|update|set|patch|replace|rollout|drain|cordon|install|upgrade|uninstall|import|taint|login|auth|iam|secrets?|kms|put|cp|mv|sync|publish|promote)\b",
            joined,
        ):
            a.bump(Risk.MEDIUM, f"{prog} queries cloud/cluster state (network)", read_only=True)
        else:
            a.bump(Risk.HIGH, f"{prog} changes cloud/cluster resources or credentials")
        return
    if prog in NETWORK_PROGRAMS:
        joined = " ".join(args)
        if re.search(r"(^|\s)(-T|--upload-file|-F|--form|--data(-binary|-raw|-urlencode)?|-d)\s*@", joined) or re.search(
            r"(-X|--request)\s*(POST|PUT|DELETE|PATCH)", joined, re.I
        ):
            a.bump(Risk.HIGH, f"{prog} sends data to a remote host")
        elif re.search(r"(^|\s)(-o|-O|--output|--remote-name)\b", joined) or prog == "wget":
            a.bump(Risk.MEDIUM, f"{prog} downloads files (network)")
        else:
            a.bump(Risk.MEDIUM, f"{prog} accesses the network", read_only=True)
        return
    if prog in ("psql", "mysql", "mariadb", "sqlite3", "mongo", "mongosh", "redis-cli", "sqlcmd", "cqlsh", "clickhouse-client"):
        _classify_db(prog, segment, a)
        return
    if prog in ("alembic", "flyway", "liquibase", "knex", "sequelize", "prisma", "rails", "rake", "diesel", "migrate", "goose", "dbmate", "atlas"):
        joined = " ".join(args).lower()
        if re.search(r"\b(downgrade|rollback|reset|drop|db:drop|db:reset|redo|down|clean|purge|force)\b", joined):
            a.bump(Risk.HIGH, f"{prog} destructive migration operation")
        else:
            a.bump(Risk.MEDIUM, f"{prog} changes the database schema or runs tasks")
        return
    if _classify_package_manager(prog, args, a):
        return
    if prog in SHELLS:
        nested: str | None = None
        for flag in ("-c", "/c", "-command", "-Command", "/C", "-lc", "-ic"):
            if flag in args:
                nested = " ".join(args[args.index(flag) + 1 :])
                break
        if nested and depth < 4:
            nested_assessment = classify_command(nested, ctx.workspace, windows=ctx.windows, _depth=depth + 1)
            a.merge(nested_assessment)
            a.bump(max(Risk.MEDIUM, nested_assessment.risk), "runs a nested shell command")
        elif args:
            a.bump(Risk.MEDIUM, f"runs shell script {args[0]}")
        else:
            a.bump(Risk.MEDIUM, "starts a shell")
        return
    if prog in INTERPRETERS or re.fullmatch(r"python3\.\d+", prog):
        if args[:1] == ["-m"] and len(args) > 1:
            module = args[1]
            if module in ANALYZERS or module in ("unittest", "doctest", "compileall", "py_compile", "pydoc", "json.tool", "timeit", "trace"):
                a.bump(Risk.LOW, "")
                return
            if module in ("pip", "pipx", "uv", "poetry"):
                _classify_package_manager(module, args[2:], a)
                return
            if module in ("venv", "virtualenv", "build", "http.server", "ensurepip"):
                a.bump(Risk.MEDIUM, f"python -m {module}")
                return
            a.bump(Risk.MEDIUM, f"runs python module {module}")
            return
        if any(x in ("-c", "-e", "--eval", "-r", "--print", "-p") for x in args[:2]):
            a.bump(Risk.MEDIUM, f"executes inline {prog} code")
            return
        if args and args[0] in ("--version", "-V", "-v", "--help", "-h"):
            return
        a.bump(Risk.MEDIUM, f"runs {prog} script")
        return
    if prog in ANALYZERS:
        writes = any(x in ("--fix", "--write", "-w", "--fix-only", "--in-place", "-i", "format") for x in args) or (prog in ("black", "isort", "autopep8", "yapf", "gofmt", "goimports", "rustfmt") and not any(
            x in ("--check", "--diff", "-l", "--list", "-d") for x in args
        ))
        a.bump(Risk.LOW if not writes else Risk.MEDIUM, "formats/fixes files in place" if writes else "")
        a.read_only = False
        return
    if prog in BUILD_TOOLS:
        target = next((x for x in args if not x.startswith("-")), "")
        if target in ("clean", "distclean", "uninstall", "install", "deploy", "publish", "release", "push"):
            a.bump(Risk.MEDIUM if target in ("clean", "distclean") else Risk.HIGH, f"{prog} {target}")
        else:
            a.bump(Risk.LOW, "")
            a.read_only = False
        return
    if prog in ("sed", "perl") and any(x == "-i" or x.startswith("-i") or x == "--in-place" for x in args):
        a.bump(Risk.MEDIUM, f"{prog} edits files in place")
        return
    if prog == "find":
        roots = [x for x in args[: next((i for i, x in enumerate(args) if x.startswith("-")), len(args))]]
        deleting = "-delete" in args
        if any(x in ("-exec", "-execdir", "-ok", "-okdir") for x in args):
            idx = next(i for i, x in enumerate(args) if x in ("-exec", "-execdir", "-ok", "-okdir"))
            deleting = deleting or _norm_program(args[idx + 1]) in ("rm", "del", "shred", "unlink", "rmdir") if idx + 1 < len(args) else deleting
        if deleting and any(ctx.catastrophic_target(r) for r in roots):
            a.bump(Risk.CRITICAL, "find deletes files across a system, home or root directory")
            return
        if deleting:
            a.bump(Risk.HIGH, "find deletes the files it matches")
        output_files = [args[i + 1] for i, x in enumerate(args[:-1]) if x in ("-fprint", "-fprint0", "-fprintf", "-fls")]
        if output_files:
            outside = any(ctx.outside_workspace(w) for w in output_files)
            a.bump(Risk.HIGH if outside else Risk.MEDIUM, "find writes its output to a file" + (" outside the workspace" if outside else ""))
        if any(x in ("-delete",) for x in args):
            a.bump(Risk.HIGH, "find -delete removes files")
        elif any(x in ("-exec", "-execdir", "-ok", "-okdir") for x in args):
            idx = next(i for i, x in enumerate(args) if x in ("-exec", "-execdir", "-ok", "-okdir"))
            exec_tokens = [x for x in args[idx + 1 :] if x not in ("{}", ";", "+", "\\;")]
            exec_assessment = CommandAssessment(command=" ".join(exec_tokens), workspace=ctx.workspace)
            if exec_tokens:
                _classify_segment(exec_tokens, " ".join(exec_tokens), ctx, exec_assessment, depth + 1)
            a.merge(exec_assessment)
            a.bump(Risk.MEDIUM, "find -exec runs a command per file")
        return
    if prog in ("awk", "gawk") and re.search(r"system\s*\(|\|\s*\"", segment):
        a.bump(Risk.MEDIUM, "awk runs shell commands")
        return
    if prog in ("cat", "head", "tail", "less", "more", "type", "get-content", "grep", "rg") and any(
        re.search(r"(^|/)\.ssh/|\.aws/credentials|\.netrc|\.pgpass|\.docker/config\.json|\.kube/config|id_rsa|id_ed25519", x) for x in args
    ):
        a.bump(Risk.HIGH, "reads credential files")
        return
    if prog in ("cat", "head", "tail", "less", "more", "type", "get-content") and any(re.search(r"(^|/)\.env(\.|$)", x) for x in args):
        a.bump(Risk.MEDIUM, "reads an environment file (output will be redacted)", read_only=True)
        return
    if prog in READ_ONLY_PROGRAMS:
        escape = _read_only_escape(prog, args, segment)
        if escape:
            a.bump(Risk.MEDIUM, escape)
        return
    if prog.startswith("./") or prog.startswith("../") or "/" in tokens[0]:
        a.bump(Risk.MEDIUM, f"runs local program {tokens[0]}")
        return
    a.bump(Risk.MEDIUM, f"unknown program '{prog}'")


def classify_command(
    command: str, workspace: str | os.PathLike[str] | None = None, windows: bool | None = None, _depth: int = 0
) -> CommandAssessment:
    """Classify a shell command string. See module docstring."""
    win = (sys.platform == "win32") if windows is None else windows
    ws = os.fspath(workspace) if workspace is not None else None
    ctx = _Context(ws, win)
    a = CommandAssessment(command=command, workspace=ws)
    if not command.strip():
        a.bump(Risk.LOW, "empty command", read_only=True)
        return a
    if _FORK_BOMB.search(command):
        a.bump(Risk.CRITICAL, "fork bomb")
        return a
    segments, subs = split_segments(command)
    for sub in subs:
        if _depth < 4:
            inner = classify_command(sub, ws, windows=win, _depth=_depth + 1)
            a.merge(inner)
            a.bump(max(Risk.MEDIUM, inner.risk), "uses command substitution")
    for index, segment in enumerate(segments):
        tokens = _tokenize(segment, win)
        if tokens is None:
            a.bump(Risk.MEDIUM, "could not parse command (unbalanced quotes)")
            continue
        if not tokens:
            continue
        _classify_segment(tokens, segment, ctx, a, _depth)
        prog = _norm_program(tokens[0]) if tokens else ""
        if index > 0 and (prog in SHELLS or prog in INTERPRETERS) and len(tokens) == 1:
            prev = _tokenize(segments[index - 1], win) or [""]
            if _norm_program(prev[0]) in NETWORK_PROGRAMS or _norm_program(prev[0]) in ("cat", "echo", "printf"):
                a.bump(Risk.HIGH, "pipes content into an interpreter")
    if _DANGEROUS_SQL.search(command) and a.risk < Risk.HIGH and any(p in a.programs for p in ("echo", "printf", "cat")) and "|" in command:
        a.bump(Risk.HIGH, "pipes a destructive SQL statement")
    return a
