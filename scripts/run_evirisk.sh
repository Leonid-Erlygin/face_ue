docker run \
  -d \
  --shm-size=16g \
  --memory=160g \
  --cpus=40 \
  --user "${UID}:${UID}" \
  --name "${USER}_$(basename "$(dirname "$PWD")")_evi_risk" \
  --env HYDRA_FULL_ERROR=1 \
  --env MLFLOW_TRACKING_URI="$(cat "/home/${USER}/face_ue/configs/mlflow_uri.yaml")" \
  --rm \
  --init \
  -v "$(dirname "$PWD"):/app" \
  --gpus '"device=4"' \
  -w /app \
  "${USER}_$(basename "$(dirname "$PWD")")" \
  python3 -u experiments/evirisk_full_suite.py \
    --out /app/outputs/evirisk_full_prr \
    --resume \
    --resume-audit-fix