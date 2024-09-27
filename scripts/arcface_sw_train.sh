docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 4200214:4200214 \
 --name arcface_train_first \
 --env WANDB_API_KEY=fc366b0c6dc3150de383b028451e1cfa35009932 \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus '"device=0"' \
 -w="/app" \
 kolesnikov-face \
 python trainers/train.py fit \
 --config configs/train/train_arcface_sw.yaml \
#  --ckpt_path=/app/outputs/scf_new_data/focal_loss/weights_scf_focal_minmax_gamma2/epoch=11-step=174672_pretrained.ckpt