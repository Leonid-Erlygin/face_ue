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
 --ckpt_path=/app/model_weights/b6_whale_softmax/scf_whale_train.ckpt \
 --trainer.devices=1