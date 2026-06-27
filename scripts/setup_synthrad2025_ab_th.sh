#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

"$REPO_ROOT/.venv/Scripts/python.exe" "$REPO_ROOT/scripts/setup_synthrad2025_ab_th.py" --all "$@"