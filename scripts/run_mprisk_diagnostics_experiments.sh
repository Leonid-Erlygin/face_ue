#!/usr/bin/env bash
set -e

cd /app

HYDRA_FULL_ERROR=1 python experiments/mprisk_diagnostics_experiments.py -cn=mprisk_diagnostics_experiments

python experiments/mprisk_diagnostics_plots.py --exp_dir outputs/experiments/mprisk_diagnostics