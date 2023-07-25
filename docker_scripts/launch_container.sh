docker run -d --shm-size=8g --memory=80g --cpus=16 --user 1000:1000 --name face_eval_new --rm -it --init -v $HOME/face_ue:/app face-eval bash
docker exec face_eval_new pip install -e .