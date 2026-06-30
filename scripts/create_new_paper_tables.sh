#!/usr/bin/env bash
set -e

cd /app

HYDRA_FULL_ERROR=1 python paper_utils/prepare_new_paper_tables.py -cn=prepare_new_paper_tables