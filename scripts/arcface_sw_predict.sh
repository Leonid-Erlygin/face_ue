docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1012:1012 \
 --name arcface_predict \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus '"device=2"' \
 -w="/app" \
 kolesnikov-face \
 python trainers/train.py predict \
 --config configs/train/train_arcface_sw.yaml \
 --ckpt_path=/app/outputs/scf_new_data/arcface_vgg2/epoch=19-step=122560.ckpt
#  --trainer.devices=1 