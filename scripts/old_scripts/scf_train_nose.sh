docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1133:1134 \
 --name scf_train \
 --env WANDB_API_KEY=b2c5aadfb0bf526689d07a4bb4aae1eb58faf5b9 \
 --rm \
 --init \
 -v /home/l.erlygin/face-evaluation:/app \
 --gpus all \
 -w="/app" \
 face-eval \
 python trainers/train.py fit \
 --config configs/train/train_sphere_face_with_noise.yaml
# --ckpt_path=/app/models/scf/epoch=3-step=90000.ckpt