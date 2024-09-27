# docker run -d --shm-size=8g --memory=80g --cpus=40 --gpus '"device=0,1,2"' --user ${UID}:${UID} --name ${USER}_$(basename $(dirname "$PWD"))_dev_cont --rm -it --init -v $(dirname "$PWD"):/app ${USER}_$(basename $(dirname "$PWD")) bash
# docker exec ${USER}_$(basename $(dirname "$PWD"))_dev_cont pip install -e .
docker run -d --shm-size=8g --memory=80g --gpus '"device=2"' --user 4200214:4200214 --name kolesnikov_face_eval_new --rm -it --init -v $HOME/face_ue:/app ikolesnikov-face bash
docker exec kolesnikov_face_eval_new pip install -e .
