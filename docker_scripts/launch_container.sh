docker run -d --shm-size=8g --memory=80g --cpus=40 --gpus '"device=1"' --user ${UID}:${UID} --name ${USER}_$(basename $(dirname "$PWD"))_dev_cont --rm -it --init -v $(dirname "$PWD"):/app ${USER}_$(basename $(dirname "$PWD")) bash
docker exec ${USER}_$(basename $(dirname "$PWD"))_dev_cont pip install -e .
