docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 4200214:4200214 \
 --name scf_train_first \
 --env WANDB_API_KEY=fc366b0c6dc3150de383b028451e1cfa35009932 \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus '"device=0"' \
 -w="/app" \
 kolesnikov-face \
 python trainers/train.py fit \
 --config configs/train/train_scf_with_psd.yaml \
#  --ckpt_path=/app/outputs/scf_new_data/vgg_old_backbone_from_zero/epoch=11-step=174672_pretrained.ckpt