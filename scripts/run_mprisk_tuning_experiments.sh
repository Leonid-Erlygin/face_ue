#!/usr/bin/env bash
set -e

cd /app

HYDRA_FULL_ERROR=1 python experiments/mprisk_tuning_experiments.py -cn=mprisk_tuning_experiments

python experiments/mprisk_tuning_plots.py --exp_dir outputs/experiments/mprisk_tuning