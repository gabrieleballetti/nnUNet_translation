#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export nnUNet_raw="$REPO_ROOT/nnunet_data/raw"
export nnUNet_preprocessed="$REPO_ROOT/nnunet_data/preprocessed"
export nnUNet_results="$REPO_ROOT/nnunet_data/results"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$REPO_ROOT/.venv/Scripts/nnUNetv2_train.exe" 105 3d_fullres 0 \
  -tr nnUNetTrainerMRCT_mae \
  -p nnResUNetPlans_ABTH_48x192x224_CTNorm \
  -device cuda \
  "$@"