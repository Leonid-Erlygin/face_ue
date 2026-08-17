#!/usr/bin/env bash
set -euo pipefail
python experiments/modern_ai_experiments.py --config configs/modern_ai/posterior_sensitivity.yaml "$@"
