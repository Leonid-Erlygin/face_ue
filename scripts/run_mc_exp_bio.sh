docker run \
 -d \
 --shm-size=16g \
 --memory=160g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_mc_bio \
 --env HYDRA_FULL_ERROR=1 \
 --env MLFLOW_TRACKING_URI=$(cat /home/${USER}/face_ue/configs/mlflow_uri.yaml) \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=4"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 experiments/holue_temperature_scaling_experiments.py \
    --config-name holue_temperature_scaling_bio \
    holue_temperature_study.published_M=16 \
    exp_dir=outputs/experiments/holue_temperature_scaling_bio_M16