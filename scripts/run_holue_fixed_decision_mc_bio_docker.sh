#!/usr/bin/env bash
set -euo pipefail

# Override any of these on the command line, e.g.:
#   GPU_DEVICE=5 CPUS=32 ./run_holue_fixed_decision_mc_bio_docker.sh
GPU_DEVICE="${GPU_DEVICE:-4}"
SHM_SIZE="${SHM_SIZE:-16g}"
MEMORY="${MEMORY:-160g}"
CPUS="${CPUS:-40}"
HOST_APP_DIR="${HOST_APP_DIR:-$(dirname "$PWD")}"
PROJECT_NAME="$(basename "$HOST_APP_DIR")"
DOCKER_IMAGE="${DOCKER_IMAGE:-${USER}_${PROJECT_NAME}}"
CONTAINER_NAME="${CONTAINER_NAME:-${USER}_${PROJECT_NAME}_holue_fixed_mc_bio}"
MLFLOW_URI_FILE="${MLFLOW_URI_FILE:-/home/${USER}/face_ue/configs/mlflow_uri.yaml}"

if [[ ! -f "$MLFLOW_URI_FILE" ]]; then
  echo "MLflow URI file not found: $MLFLOW_URI_FILE" >&2
  echo "Set MLFLOW_URI_FILE=/path/to/mlflow_uri.yaml and rerun." >&2
  exit 1
fi

MLFLOW_TRACKING_URI="$(cat "$MLFLOW_URI_FILE")"

echo "Starting $CONTAINER_NAME"
echo "  image: $DOCKER_IMAGE"
echo "  GPU:   $GPU_DEVICE"
echo "  mount: $HOST_APP_DIR -> /app"

docker run \
  -d \
  --shm-size="$SHM_SIZE" \
  --memory="$MEMORY" \
  --cpus="$CPUS" \
  --user "${UID}:${UID}" \
  --name "$CONTAINER_NAME" \
  --env HYDRA_FULL_ERROR=1 \
  --env MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  --rm \
  --init \
  -v "$HOST_APP_DIR:/app" \
  --gpus "device=${GPU_DEVICE}" \
  -w="/app" \
  "$DOCKER_IMAGE" \
  python3 experiments/holue_fixed_decision_mc_experiments.py \
    --config-name holue_fixed_decision_mc_bio

echo
printf 'Follow logs with:\n  docker logs -f %q\n' "$CONTAINER_NAME"
echo "Results: $HOST_APP_DIR/outputs/experiments/holue_fixed_decision_mc_bio"
