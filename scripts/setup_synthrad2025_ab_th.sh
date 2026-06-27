#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
	PYTHON_EXE="$REPO_ROOT/.venv/Scripts/python.exe"
else
	PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
fi

"$PYTHON_EXE" "$REPO_ROOT/scripts/setup_synthrad2025_ab_th.py" --all "$@"