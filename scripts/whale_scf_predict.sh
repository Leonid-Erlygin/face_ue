docker run \
 --shm-size=8g \
 --memory=160g \
 --gpus '"device=0"' \
 --user 1005:1005 \
 --name whale_scf_train \
 --rm \
 --init \
 -v $HOME/face_ue:/app \
 -w="/app" \
 face-eval \
 python trainers/train.py predict \
 --config configs/train/train_whale_scf.yaml \
 --ckpt_path=/app/outputs/scf_train/weights/epoch=5-step=19000.ckpt \
 --trainer.devices=1