#!/usr/bin/env bash
set -euo pipefail

for path in \
  datasets/tool_routing/toolbench_g1/manifest.json \
  model_weights/backbone/bert_toolbench_arcface/backbone.pth \
  model_weights/text_models/trained_scf/toolbench.ckpt; do
  if [[ ! -f "$path" ]]; then
    echo "Missing $path" >&2
    echo "Run: bash scripts/modern_ai/prepare_tool_routing.sh" >&2
    echo "then: bash scripts/modern_ai/train_tool_routing.sh" >&2
    exit 1
  fi
done
python experiments/modern_ai_experiments.py --config configs/modern_ai/toolbench_tool_routing_osr.yaml
