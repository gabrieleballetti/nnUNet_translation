#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/train_synthrad2025_ab_fold0.sh" "$@"
"$SCRIPT_DIR/train_synthrad2025_th_fold0.sh" "$@"