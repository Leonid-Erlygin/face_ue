#!/usr/bin/env bash
set -euo pipefail

BACKBONE="model_weights/backbone/bert_toolbench_arcface/backbone.pth"
CENTERS="model_weights/backbone/bert_toolbench_arcface/softmax_weight.pt"
if [[ ! -f "$BACKBONE" || ! -f "$CENTERS" ]]; then
  echo "ArcFace export is missing. Run scripts/modern_ai/train_tool_routing.sh first." >&2
  exit 1
fi
if [[ ! -f datasets/tool_routing/toolbench_g1/train_scf.jsonl ]]; then
  echo "Prepared ToolBench SCF training data is missing." >&2
  exit 1
fi

rm -rf outputs/tool_routing/scf outputs/tool_routing/scf_concentration_audit
rm -f model_weights/text_models/trained_scf/toolbench.ckpt

python training/trainers/train.py --config-name text_model_toolbench_scf.yaml

SCF_CKPT="outputs/tool_routing/scf/last.ckpt"
if [[ ! -f "$SCF_CKPT" ]]; then
  echo "Expected SCF checkpoint not found: $SCF_CKPT" >&2
  exit 1
fi

python experiments/audit_tool_routing_scf.py \
  --prepared-dir datasets/tool_routing/toolbench_g1 \
  --backbone-path "$BACKBONE" \
  --softmax-weights-path "$CENTERS" \
  --scf-checkpoint-path "$SCF_CKPT" \
  --output-dir outputs/tool_routing/scf_concentration_audit

mkdir -p model_weights/text_models/trained_scf
cp "$SCF_CKPT" model_weights/text_models/trained_scf/toolbench.ckpt

echo "Tool-routing SCF retraining/audit complete."
