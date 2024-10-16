docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1012:1012 \
 --name scf_train \
 --env WANDB_API_KEY=fc366b0c6dc3150de383b028451e1cfa35009932 \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus '"device=1"' \
 -w="/app" \
 kolesnikov-face \
 python trainers/train.py fit \
 --config configs/train/train_scf_psd_new_emb.yaml
# --ckpt_path=/app/models/scf/epoch=3-step=90000.ckpt