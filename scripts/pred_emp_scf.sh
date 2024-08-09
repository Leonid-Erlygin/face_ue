docker run \
 --shm-size=8g \
 --memory=160g \
 --gpus '"device=1"' \
 --cpus=40 \
 --user 1005:1005 \
 --name scf_pred \
 --rm \
 --init \
 -v $HOME/face_ue:/app \
 -w="/app" \
 face-eval \
 python trainers/train.py predict \
 --config configs/train/train_scf.yaml \
 --ckpt_path=/app/model_weights/scf_base.ckpt \
 --trainer.devices=1