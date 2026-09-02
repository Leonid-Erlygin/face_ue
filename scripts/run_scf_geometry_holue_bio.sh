#!/usr/bin/env bash
set -euo pipefail
python experiments/scf_geometry_holue_experiments.py --config-name scf_geometry_holue_bio "$@"
