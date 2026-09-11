#!/usr/bin/env bash
set -euo pipefail

cd "${1:-.}"
OUT="holue_fixed_decision_mc_results_$(date +%Y%m%d_%H%M%S).tar.gz"

paths=()
for d in \
  outputs/experiments/holue_fixed_decision_mc_text \
  outputs/experiments/holue_fixed_decision_mc_bio; do
  if [[ -d "$d" ]]; then
    paths+=("$d/tables" "$d/study_manifest.json")
  fi
done

if [[ ${#paths[@]} -eq 0 ]]; then
  echo "No fixed-decision HolUE result directories found." >&2
  exit 1
fi

tar -czf "$OUT" "${paths[@]}"
echo "$OUT"
