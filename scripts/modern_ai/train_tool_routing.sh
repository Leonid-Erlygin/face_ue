#!/usr/bin/env bash
set -euo pipefail

EXPECTED_PROTOCOL="toolbench_g1_stage_class_disjoint_v2"
CURRENT_PROTOCOL="$(python - <<'PY'
import json
from pathlib import Path
p=Path("datasets/tool_routing/toolbench_g1/manifest.json")
if not p.exists():
    print("")
else:
    print(json.loads(p.read_text(encoding="utf-8")).get("protocol", ""))
PY
)"
if [[ "$CURRENT_PROTOCOL" != "$EXPECTED_PROTOCOL" ]]; then
  echo "Prepared ToolBench protocol is '$CURRENT_PROTOCOL'; rebuilding as $EXPECTED_PROTOCOL."
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
  --output-dir outputs/tool_routing/arcface_geometry_audit \
  --min-query-gallery-accuracy 0.05 \
  --min-api-center-accuracy 0.05

# Do not train an uncertainty head on top of a demonstrably broken mean
# embedding geometry.  In stage-disjoint v2 the fail-fast routing metric is on
# calibration API identities that ArcFace never saw during training.
python training/trainers/train.py --config-name text_model_toolbench_scf.yaml

SCF_CKPT="outputs/tool_routing/scf/last.ckpt"
if [[ ! -f "$SCF_CKPT" ]]; then
  echo "Expected SCF checkpoint not found: $SCF_CKPT" >&2
  exit 1
fi

# Validation-only concentration audit.  Do this before any final OSR test so
# model changes cannot be informed by the held-out ToolBench test split.
python experiments/audit_tool_routing_scf.py \
  --prepared-dir datasets/tool_routing/toolbench_g1 \
  --backbone-path model_weights/backbone/bert_toolbench_arcface/backbone.pth \
  --softmax-weights-path model_weights/backbone/bert_toolbench_arcface/softmax_weight.pt \
  --scf-checkpoint-path "$SCF_CKPT" \
  --output-dir outputs/tool_routing/scf_concentration_audit

mkdir -p model_weights/text_models/trained_scf
cp "$SCF_CKPT" model_weights/text_models/trained_scf/toolbench.ckpt

echo "Tool-routing ArcFace/SCF training complete."
echo "  backbone: model_weights/backbone/bert_toolbench_arcface/backbone.pth"
echo "  centers:  model_weights/backbone/bert_toolbench_arcface/softmax_weight.pt"
echo "  SCF:      model_weights/text_models/trained_scf/toolbench.ckpt"
