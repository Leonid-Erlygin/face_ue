#!/usr/bin/env bash
# Shared launcher. Public wrappers at the repository root select stage/domain.
set -euo pipefail
stage="${1:?stage required: sanity or full}"
domain="${2:?domain required: text or bio}"
shift 2
[[ "$stage" == sanity || "$stage" == full ]] || { echo "Unknown stage: $stage" >&2; exit 2; }
[[ "$domain" == text || "$domain" == bio ]] || { echo "Unknown domain: $domain" >&2; exit 2; }
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_APP_DIR="${HOST_APP_DIR:-$script_root}"
HOST_APP_DIR="$(cd -- "$HOST_APP_DIR" && pwd)"
[[ -f "$HOST_APP_DIR/experiments/mprisk_evidence_experiments.py" ]] || { echo "Repository root not found: $HOST_APP_DIR" >&2; exit 2; }
user_name="${USER:-$(id -un)}"
GPU_DEVICE="${GPU_DEVICE:-4}"
CPUS="${CPUS:-40}"
MEMORY="${MEMORY:-160g}"
SHM_SIZE="${SHM_SIZE:-16g}"
THREADS="${THREADS:-${CPUS%%.*}}"
DOCKER_IMAGE="${DOCKER_IMAGE:-${user_name}_$(basename -- "$HOST_APP_DIR")}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-${stage}_${domain}_${stamp}_$$}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "RUN_ID must be a simple filename" >&2; exit 2; }
relative="outputs/mprisk_evidence/$RUN_ID"
CONTAINER_NAME="${CONTAINER_NAME:-${user_name}_$(basename -- "$HOST_APP_DIR")_${RUN_ID}}"
mkdir -p -- "$HOST_APP_DIR/outputs/mprisk_evidence"
args=(python3 experiments/mprisk_evidence_experiments.py --stage "$stage" --domain "$domain" --device auto --run-dir "/app/$relative")
if [[ "$stage" == full ]]; then
  [[ "${APPROVE_FULL:-0}" == 1 ]] || {
    echo "Full runs are gated. After reviewing sanity results set APPROVE_FULL=1 and SANITY_MANIFEST=/path/to/manifest.json" >&2; exit 2;
  }
  [[ -n "${SANITY_MANIFEST:-}" && -f "$SANITY_MANIFEST" ]] || { echo "SANITY_MANIFEST must point to the reviewed run's manifest.json" >&2; exit 2; }
  approval="$(cd -- "$(dirname -- "$SANITY_MANIFEST")" && pwd)/$(basename -- "$SANITY_MANIFEST")"
  case "$approval" in
    "$HOST_APP_DIR"/*) approval="/app/${approval#"$HOST_APP_DIR"/}" ;;
    *) echo "The sanity manifest must be inside HOST_APP_DIR so Docker can read it." >&2; exit 2 ;;
  esac
  args+=(--confirm-full --approved-sanity "$approval")
fi
args+=("$@")
env_args=(--env PYTHONUNBUFFERED=1 --env HYDRA_FULL_ERROR=1 --env TERM=xterm
          --env "OMP_NUM_THREADS=$THREADS" --env "MKL_NUM_THREADS=$THREADS"
          --env "OPENBLAS_NUM_THREADS=$THREADS" --env HOME=/tmp/mprisk-home
          --env MPLCONFIGDIR=/tmp/mprisk-matplotlib)
# Retain the user's MLflow environment when supplied, but never copy credentials
# into experiment outputs. These study scripts do not require a tracking server.
uri_file="${MLFLOW_URI_FILE:-/home/${user_name}/face_ue/configs/mlflow_uri.yaml}"
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  env_args+=(--env "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI")
elif [[ -f "$uri_file" ]]; then
  env_args+=(--env "MLFLOW_TRACKING_URI=$(cat -- "$uri_file")")
fi
resource_args=(--shm-size="$SHM_SIZE" --memory="$MEMORY" --cpus="$CPUS")
if [[ "$GPU_DEVICE" != cpu ]]; then
  [[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] || { echo "GPU_DEVICE must be one GPU index or cpu" >&2; exit 2; }
  resource_args+=(--gpus "device=$GPU_DEVICE")
fi
printf 'Starting %s\nImage: %s\nRepository: %s\nStage: %s / %s\n' "$CONTAINER_NAME" "$DOCKER_IMAGE" "$HOST_APP_DIR" "$stage" "$domain"
printf 'Results: %s/%s\nShare ZIP: %s/%s.zip\n' "$HOST_APP_DIR" "$relative" "$HOST_APP_DIR" "$relative"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'docker run -d --rm --init '; printf '%q ' "${resource_args[@]}" "${env_args[@]}" --user "$(id -u):$(id -g)" --name "$CONTAINER_NAME" -v "$HOST_APP_DIR:/app" -w /app "$DOCKER_IMAGE" "${args[@]}"; printf '\n'; exit 0
fi
container_id="$(docker run -d --rm --init "${resource_args[@]}" "${env_args[@]}" \
  --user "$(id -u):$(id -g)" --name "$CONTAINER_NAME" -v "$HOST_APP_DIR:/app" -w /app \
  "$DOCKER_IMAGE" "${args[@]}")"
printf 'Container: %s\nFollow: docker logs -f %q\n' "$container_id" "$CONTAINER_NAME"
echo 'The ZIP is created by the Python process after completion (also on ordinary Python failures).'
echo 'Check manifest.json: status must be complete. No timing/performance conclusion is automated.'
