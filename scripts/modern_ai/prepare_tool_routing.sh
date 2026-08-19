#!/usr/bin/env bash
set -euo pipefail

python experiments/prepare_tool_routing.py all \
  --toolbench-root external/ToolBench \
  --prepared-dir datasets/tool_routing/toolbench_g1 \
  --bfcl-root external/gorilla \
  --bfcl-manifest datasets/tool_routing/bfcl_manifest.json \
  --num-known-tools "${TOOL_ROUTING_KNOWN_TOOLS:-1024}" \
  --num-unknown-tools "${TOOL_ROUTING_UNKNOWN_TOOLS:-256}" \
  --min-queries-per-tool "${TOOL_ROUTING_MIN_QUERIES:-3}" \
  --unknown-calibration-fraction "${TOOL_ROUTING_UNKNOWN_CAL_FRACTION:-0.5}" \
  --seed "${TOOL_ROUTING_SEED:-777}"
