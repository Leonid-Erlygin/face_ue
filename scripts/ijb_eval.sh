docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1012:1012 \
 --name scf_train \
 --rm \
 --init \
 -v /home/i.kolesnikov/face_ue:/app \
 --gpus all \
 -w="/app" \
 kolesnikov-face \
 python evaluation/ijb_evals.py
