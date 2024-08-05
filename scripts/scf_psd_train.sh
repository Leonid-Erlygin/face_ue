docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1012:1012 \
 --name scf_train_second \
 --env WANDB_API_KEY=fc366b0c6dc3150de383b028451e1cfa35009932 \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus '"device=1"' \
 -w="/app" \
 kolesnikov-face \
 python trainers/train.py fit \
 --config configs/train/train_scf_with_psd.yaml \
 --ckpt_path=/app/outputs/scf_new_data/grid_distortion/weights_scf_grid_distort_album_75%/epoch=11-step=174672_pretrained.ckpt 