docker run \
 --shm-size=16g \
 --memory=80g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_scf_predict \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=0"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 trainers/train_multiple_runs_scf.py -cn=predict_scf