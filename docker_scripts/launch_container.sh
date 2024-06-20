docker run -d --shm-size=16g --memory=160g --gpus '"device=2"' --user 1005:1005 --name erlygin_face_eval --rm -it --init -v $HOME/face_ue:/app face-eval bash
docker exec erlygin_face_eval pip install -e .
