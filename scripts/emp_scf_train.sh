docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1005:1005 \
 --name scf \
 --env WANDB_API_KEY=b2c5aadfb0bf526689d07a4bb4aae1eb58faf5b9 \
 --rm \
 --init \
 -v /home/l.erlygin/face_ue:/app \
 --gpus '"device=0"' \
 -w="/app" \
 face-eval \
 python trainers/train.py fit \
 --config configs/train/train_scf.yaml \
# --ckpt_path=/app/model_weights/scf_base.ckpt