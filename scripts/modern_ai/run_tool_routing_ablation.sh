#!/usr/bin/env bash
set -euo pipefail

# Main condition: native ArcFace embedding + learned per-query SCF concentration.
bash scripts/modern_ai/run_toolbench_osr.sh

SUMMARY="outputs/modern_ai/toolbench_g1_open_set_routing/summary.json"
MEDIAN_KAPPA="$(python - "$SUMMARY" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    obj=json.load(f)
print(obj["query_kappa_stats"]["calibration"]["median"])
PY
)"

echo "Matched constant-kappa ablation: kappa=${MEDIAN_KAPPA}"
python experiments/modern_ai_experiments.py \
  --config configs/modern_ai/toolbench_tool_routing_osr.yaml \
  --override embedder.kind=repo_arcface \
  --override query_uncertainty.source=default \
  --override query_uncertainty.default_kappa="${MEDIAN_KAPPA}" \
  --override output_dir=outputs/modern_ai/toolbench_g1_open_set_routing_constant_kappa

# External transfer to BFCL.  Both conditions freeze gallery kappa learned on
# ToolBench; the constant condition also freezes the ToolBench calibration-median
# query concentration, so BFCL labels cannot tune either spherical parameter.
bash scripts/modern_ai/run_bfcl.sh
python experiments/modern_ai_experiments.py \
  --config configs/modern_ai/bfcl.yaml \
  --override embedder.kind=repo_arcface \
  --override query_uncertainty.source=default \
  --override query_uncertainty.default_kappa="${MEDIAN_KAPPA}" \
  --override output_dir=outputs/modern_ai/bfcl_open_set_routing_constant_kappa
