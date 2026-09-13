#!/usr/bin/env bash
# This wrapper authorizes the existing full driver; it does not change the model.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${HOST_APP_DIR:-$(cd -- "$here/.." && pwd)}"
root="$(cd -- "$root" && pwd)"
launcher="$root/run_mprisk_evidence_full_text_docker.sh"
[[ -f "$launcher" ]] || { echo "Unzip this folder in the repository root; missing $launcher" >&2; exit 2; }
python3 "$here/verify_review_files.py"
case "$here/" in "$root/"*) ;; *) echo "Review files must be inside HOST_APP_DIR for the existing launcher." >&2; exit 2;; esac
# Full runs must not accidentally inherit sanity replay variables.
unset PREVIOUS_RUN REPLAY_ONLY
export APPROVE_FULL=1
export SANITY_MANIFEST="$here/reviewed_sanity/text/manifest.json"
export REVIEW_RESOLUTION="$here/text_review_resolution.json"
export HOST_APP_DIR="$root"
seed="${SEED:-777}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
echo "Full text benchmark, seed $seed: existing validation retained; calibration not certified."
exec bash "$launcher" --seed "$seed" --max-fit 3000 --fit-iterations 250 --search-budget 1024 --bootstrap 200 "$@"
