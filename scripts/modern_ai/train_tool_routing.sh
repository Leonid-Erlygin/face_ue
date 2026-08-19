#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f datasets/tool_routing/toolbench_g1/train_arcface.jsonl || ! -f datasets/tool_routing/toolbench_g1/train_scf.jsonl ]]; then
  echo "Prepared ToolBench protocol missing/incomplete; running dataset preparation first."
  bash scripts/modern_ai/prepare_tool_routing.sh
fi

python training/trainers/train.py --config-name text_model_toolbench_arcface.yaml

ARC_CKPT="outputs/tool_routing/arcface/last.ckpt"
if [[ ! -f "$ARC_CKPT" ]]; then
  echo "Expected ArcFace checkpoint not found: $ARC_CKPT" >&2
  exit 1
fi
python experiments/export_tool_routing_arcface.py \
  --checkpoint "$ARC_CKPT" \
  --output-dir model_weights/backbone/bert_toolbench_arcface

python experiments/audit_tool_routing_arcface.py \
  --prepared-dir datasets/tool_routing/toolbench_g1 \
  --backbone-path model_weights/backbone/bert_toolbench_arcface/backbone.pth \
  --softmax-weights-path model_weights/backbone/bert_toolbench_arcface/softmax_weight.pt \
  --output-dir outputs/tool_routing/arcface_geometry_audit

python training/trainers/train.py --config-name text_model_toolbench_scf.yaml

SCF_CKPT="outputs/tool_routing/scf/last.ckpt"
if [[ ! -f "$SCF_CKPT" ]]; then
  echo "Expected SCF checkpoint not found: $SCF_CKPT" >&2
  exit 1
fi
mkdir -p model_weights/text_models/trained_scf
cp "$SCF_CKPT" model_weights/text_models/trained_scf/toolbench.ckpt

echo "Tool-routing ArcFace/SCF training complete."
echo "  backbone: model_weights/backbone/bert_toolbench_arcface/backbone.pth"
echo "  centers:  model_weights/backbone/bert_toolbench_arcface/softmax_weight.pt"
echo "  SCF:      model_weights/text_models/trained_scf/toolbench.ckpt"
