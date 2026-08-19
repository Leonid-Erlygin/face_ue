#!/usr/bin/env bash
set -euo pipefail

python experiments/prepare_tool_routing.py stats-toolbench \
  --g1-query-path "${TOOLBENCH_G1_PATH:-external/ToolBench/data/instruction/G1_query.json}" \
  --output "${TOOL_ROUTING_STATS_OUTPUT:-outputs/tool_routing/toolbench_g1_query_density.json}"
