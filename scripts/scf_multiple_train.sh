docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 4200214:4200214 \
 --name scf_train_multiple \
 --env WANDB_API_KEY=fc366b0c6dc3150de383b028451e1cfa35009932 \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus '"device=2"' \
 -w="/app" \
 ikolesnikov-face \
 python3 trainers/train_multiple_runs_scf.py \
#  --config configs/train/train_scf_with_psd.yaml \
#  --ckpt_path=/app/outputs/scf_new_data/vgg_old_backbone_from_zero/epoch=11-step=174672_pretrained.ckpt
#--multirun model.head.latent_vector_size=1024,2048,4096,8192