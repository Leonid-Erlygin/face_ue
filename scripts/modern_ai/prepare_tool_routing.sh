#!/usr/bin/env bash

set -euo pipefail

TOOLBENCH_ARCHIVE_DIR="${TOOLBENCH_ARCHIVE_DIR:-/app/datasets/ToolBench_data}"
TOOLBENCH_ROOT="${TOOLBENCH_ROOT:-/app/external/ToolBench}"

export TOOLBENCH_ARCHIVE_DIR

G1_PATH="${TOOLBENCH_ROOT}/data/instruction/G1_query.json"
DATA_ZIP="${TOOLBENCH_ARCHIVE_DIR}/data.zip"
REPRO_ZIP="${TOOLBENCH_ARCHIVE_DIR}/reproduction_data.zip"

echo "[tool-routing] ToolBench archive dir: ${TOOLBENCH_ARCHIVE_DIR}"
echo "[tool-routing] ToolBench working dir: ${TOOLBENCH_ROOT}"

if [[ -f "${G1_PATH}" ]]; then
    echo "[tool-routing] ToolBench G1 already extracted:"
    echo "  ${G1_PATH}"
else
    if [[ ! -f "${DATA_ZIP}" ]]; then
        echo "ERROR: ToolBench dataset not found."
        echo
        echo "Expected either:"
        echo "  ${G1_PATH}"
        echo
        echo "or:"
        echo "  ${DATA_ZIP}"
        exit 1
    fi

    echo "[tool-routing] Found local ToolBench archive:"
    ls -lh "${DATA_ZIP}"
fi

if [[ -f "${REPRO_ZIP}" ]]; then
    echo "[tool-routing] Found reproduction archive:"
    ls -lh "${REPRO_ZIP}"
else
    echo "[tool-routing] WARNING: reproduction_data.zip not found."
    echo "[tool-routing] It is not required for G1 tool-routing preparation."
fi

python experiments/prepare_tool_routing.py all \
  --toolbench-root "${TOOLBENCH_ROOT}" \
  --prepared-dir datasets/tool_routing/toolbench_g1 \
  --bfcl-root external/gorilla \
  --bfcl-manifest datasets/tool_routing/bfcl_manifest.json \
  --num-known-tools "${TOOL_ROUTING_KNOWN_TOOLS:-1024}" \
  --num-unknown-tools "${TOOL_ROUTING_UNKNOWN_TOOLS:-256}" \
  --min-queries-per-tool "${TOOL_ROUTING_MIN_QUERIES:-3}" \
  --unknown-calibration-fraction "${TOOL_ROUTING_UNKNOWN_CAL_FRACTION:-0.5}" \
  --seed "${TOOL_ROUTING_SEED:-777}"