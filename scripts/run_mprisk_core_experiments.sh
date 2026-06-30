#!/usr/bin/env bash
set -e

cd /app

HYDRA_FULL_ERROR=1 python experiments/mprisk_core_experiments.py -cn=mprisk_core_experiments