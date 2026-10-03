#!/usr/bin/env sh
# Install AI Engineer into an isolated virtual environment and expose the `aie` command.
#
# Usage: scripts/install.sh [--prefix DIR] [--extras all|anthropic,web,browser|none] [--source PATH_OR_URL]
#   --prefix   where the virtual environment lives (default: ~/.ai-engineer)
#   --extras   optional feature sets (default: all)
#   --source   local checkout or pip-installable URL (default: the checkout containing this script)
set -eu

PREFIX="${HOME}/.ai-engineer"
EXTRAS="all"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE=$(dirname "$SCRIPT_DIR")
BIN_DIR="${HOME}/.local/bin"

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --extras) EXTRAS="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PYTHON="$candidate"; break
    fi
  fi
done
if [ -z "$PYTHON" ]; then
  echo "Python 3.11 or newer is required (https://www.python.org/downloads/)." >&2
  exit 1
fi
command -v git >/dev/null 2>&1 || echo "warning: git not found; checkpoints will use file backups and git features are disabled" >&2

echo "Using $($PYTHON --version) at $(command -v "$PYTHON")"
"$PYTHON" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/python" -m pip install --quiet --upgrade pip
if [ "$EXTRAS" = "none" ]; then
  SPEC="$SOURCE"
else
  SPEC="$SOURCE[$EXTRAS]"
fi
"$PREFIX/venv/bin/python" -m pip install --quiet "$SPEC"

mkdir -p "$BIN_DIR"
ln -sf "$PREFIX/venv/bin/aie" "$BIN_DIR/aie"
echo "Installed: $("$PREFIX/venv/bin/aie" --version)"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Add $BIN_DIR to your PATH to use 'aie' from anywhere." ;;
esac
echo "Next: cd your/project && aie init && aie doctor"
