docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 4200214:4200214 \
 --name scf_predict \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus '"device=1"' \
 -w="/app" \
 ikolesnikov-face \
 python trainers/train.py predict \
 --config configs/train/train_scf_with_psd.yaml \
 --ckpt_path=/app/outputs/scf_new_data/vgg_old_backbone_from_zero/epoch=6-step=54908.ckpt
#  --trainer.devices=1 