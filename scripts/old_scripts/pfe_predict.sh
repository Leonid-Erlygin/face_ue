docker run -d \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1133:1134 \
 --name scf_train \
 --rm \
 --init \
 -v /home/l.erlygin/face_ue:/app \
 --gpus all \
 -w="/app" \
 face-eval \
 python trainers/train.py predict \
 --config configs/train/train_pfe.yaml \
 --trainer.devices=1