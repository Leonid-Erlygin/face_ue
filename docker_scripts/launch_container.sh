docker run -d --shm-size=8g --memory=80g --gpus '"device=0,1"' --user 1005:1005 --name face_eval_new --rm -it --init -v $HOME/face_ue:/app face-eval bash
docker exec face_eval_new pip install -e .