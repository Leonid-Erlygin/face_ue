docker run -d --shm-size=8g --memory=80g --gpus '"device=1,2"' --user 1012:1012 --name kolesnikov_face_eval_new --rm -it --init -v $HOME/face_ue:/app kolesnikov-face bash
docker exec kolesnikov_face_eval_new pip install -e .
