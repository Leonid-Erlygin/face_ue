#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || { echo "Usage: bash $0 /path/to/stopped/run --stopped" >&2; exit 2; }
exec python3 "$root/scripts/mprisk_evidence/package_results.py" "$@"
