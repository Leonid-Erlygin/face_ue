#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f datasets/tool_routing/bfcl_manifest.json ]]; then
  python experiments/prepare_tool_routing.py download-bfcl \
    --output-root external/gorilla \
    --manifest datasets/tool_routing/bfcl_manifest.json
fi
if [[ ! -f outputs/modern_ai/toolbench_g1_open_set_routing/summary.json ]]; then
  echo "ToolBench OSR calibration summary is missing." >&2
  echo "Run: bash scripts/modern_ai/run_toolbench_osr.sh" >&2
  echo "BFCL intentionally transfers gallery kappa from ToolBench instead of fitting it on BFCL." >&2
  exit 1
fi
if [[ ! -f model_weights/text_models/trained_scf/toolbench.ckpt ]]; then
  echo "ToolBench-trained SCF model is missing." >&2
  echo "Run: bash scripts/modern_ai/train_tool_routing.sh" >&2
  exit 1
fi
python experiments/modern_ai_experiments.py --config configs/modern_ai/bfcl.yaml
