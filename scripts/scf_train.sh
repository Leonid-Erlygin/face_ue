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
 --gpus '"device=1"' \
 -w="/app" \
 face-eval \
 python trainers/train.py fit \
 --config configs/train/train_scf.yaml \
# --ckpt_path=/app/outputs/scf_new_data/grid_distortion/weights_scf_grid_distort_album_75%/epoch=11-step=174672_pretrained.ckpt 