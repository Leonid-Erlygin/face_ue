#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/modern_ai/kappa_root_sensitivity.yaml"
PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" experiments/prepare_beir.py --config "$CONFIG"
"$PYTHON_BIN" experiments/modern_ai_experiments.py --config "$CONFIG"
