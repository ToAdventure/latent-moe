#!/usr/bin/env bash
set -euo pipefail

: "${GRPO_CATEGORIES:=CDs_and_Vinyl}"
export GRPO_CATEGORIES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/grpo_train_four_datasets.sh" "$@"
