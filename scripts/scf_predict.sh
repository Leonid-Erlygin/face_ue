docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1012:1012 \
 --name scf_predict \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus '"device=2"' \
 -w="/app" \
 kolesnikov-face \
 python trainers/train.py predict \
 --config configs/train/train_scf_with_psd.yaml \
 --ckpt_path=/app/outputs/scf_train/weights_power_dul_rgs/epoch=4-step=56000.ckpt
#  --trainer.devices=1 