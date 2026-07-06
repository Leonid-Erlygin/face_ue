#!/usr/bin/env bash
set -euo pipefail

cd /app

LOG_DIR="outputs/logs/new_paper_parallel"
mkdir -p "${LOG_DIR}"

echo "Logs will be saved to: ${LOG_DIR}"

PIDS=()
NAMES=()

run_bg () {
  local name="$1"
  shift

  local log_file="${LOG_DIR}/${name}.log"

  echo "[START] ${name}"
  echo "        log: ${log_file}"
  echo "        cmd: $*"

  (
    set -euo pipefail
    cd /app
    HYDRA_FULL_ERROR=1 "$@"
  ) > "${log_file}" 2>&1 &

  PIDS+=("$!")
  NAMES+=("${name}")
}

wait_all () {
  local fail=0

  for i in "${!PIDS[@]}"; do
    local pid="${PIDS[$i]}"
    local name="${NAMES[$i]}"

    if wait "${pid}"; then
      echo "[DONE] ${name}"
    else
      echo "[FAILED] ${name}"
      echo "         see ${LOG_DIR}/${name}.log"
      fail=1
    fi
  done

  PIDS=()
  NAMES=()

  if [[ "${fail}" -ne 0 ]]; then
    echo "At least one process failed."
    exit 1
  fi
}

echo "============================================================"
echo "STAGE 1: Core experiments and cache warm-up"
echo "============================================================"

# These two jobs create the main template caches and core outputs.
# Running all later stages before these finish can cause cache races.
run_bg "core_bio"  python experiments/mprisk_core_experiments.py -cn=mprisk_core_bio_complete
run_bg "core_text" python experiments/mprisk_core_experiments.py -cn=mprisk_core_text_complete

wait_all

echo "============================================================"
echo "STAGE 2: Tuning, diagnostics, and fair comparisons"
echo "============================================================"

run_bg "tuning_bio" python experiments/mprisk_tuning_experiments.py -cn=mprisk_tuning_bio_complete recompute_template_pooling=False
run_bg "tuning_text" python experiments/mprisk_tuning_experiments.py -cn=mprisk_tuning_text_complete recompute_template_pooling=False

run_bg "diagnostics_bio" python experiments/mprisk_diagnostics_experiments.py -cn=mprisk_diagnostics_bio_complete recompute_template_pooling=False
run_bg "diagnostics_text" python experiments/mprisk_diagnostics_experiments.py -cn=mprisk_diagnostics_text_complete recompute_template_pooling=False

run_bg "fair_tuning_bio" python experiments/mprisk_fair_tuning_experiments.py -cn=mprisk_fair_tuning_bio recompute_template_pooling=False
run_bg "fair_tuning_text" python experiments/mprisk_fair_tuning_experiments.py -cn=mprisk_fair_tuning_text recompute_template_pooling=False

wait_all

echo "============================================================"
echo "STAGE 3: Generate LaTeX tables and figures"
echo "============================================================"

run_bg "tables_bio" python paper_utils/prepare_new_paper_tables.py -cn=prepare_new_paper_tables_bio
run_bg "tables_text" python paper_utils/prepare_new_paper_tables.py -cn=prepare_new_paper_tables_text

# Optional compatibility configs if you still keep them.
if [[ -f "configs/latex_tables_new_paper/prepare_new_paper_tables_bio_diagnostics.yaml" ]]; then
  run_bg "tables_bio_diagnostics" python paper_utils/prepare_new_paper_tables.py -cn=prepare_new_paper_tables_bio_diagnostics
fi

if [[ -f "configs/latex_tables_new_paper/prepare_new_paper_tables_text_diagnostics.yaml" ]]; then
  run_bg "tables_text_diagnostics" python paper_utils/prepare_new_paper_tables.py -cn=prepare_new_paper_tables_text_diagnostics
fi

wait_all

echo "============================================================"
echo "ALL NEW PAPER EXPERIMENTS FINISHED SUCCESSFULLY"
echo "============================================================"

echo "Core outputs:"
echo "  outputs/experiments/mprisk_core_bio_complete"
echo "  outputs/experiments/mprisk_core_text_complete"

echo "Tuning outputs:"
echo "  outputs/experiments/mprisk_tuning_bio_complete"
echo "  outputs/experiments/mprisk_tuning_text_complete"

echo "Diagnostics outputs:"
echo "  outputs/experiments/mprisk_diagnostics_bio_complete"
echo "  outputs/experiments/mprisk_diagnostics_text_complete"

echo "Fair tuning outputs:"
echo "  outputs/experiments/mprisk_fair_tuning_bio"
echo "  outputs/experiments/mprisk_fair_tuning_text"

echo "Tables:"
echo "  outputs/latex_tables_new_paper_bio"
echo "  outputs/latex_tables_new_paper_text"