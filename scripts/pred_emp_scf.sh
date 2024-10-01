docker run \
 --shm-size=8g \
 --memory=160g \
 --gpus '"device=2"' \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_scf_pred \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python trainers/train.py predict \
 --config configs/train/train_scf.yaml \
 --ckpt_path=/app/outputs/scf_uniform_batch/default_scf/epoch=14-step=218340.ckpt \
# --ckpt_path=/app/model_weights/scf_base.ckpt \
