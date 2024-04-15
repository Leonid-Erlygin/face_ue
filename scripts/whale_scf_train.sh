docker run \
 --shm-size=8g \
 --memory=160g \
 --gpus '"device=0,2"' \
 --user 1005:1005 \
 --name whale_scf_train \
 --env WANDB_API_KEY=b2c5aadfb0bf526689d07a4bb4aae1eb58faf5b9 \
 --rm \
 --init \
 -v $HOME/face_ue:/app \
 -w="/app" \
 face-eval \
 python trainers/train.py fit \
 --config configs/train/train_whale_scf.yaml
# --ckpt_path=/app/models/scf/epoch=3-step=90000.ckpt