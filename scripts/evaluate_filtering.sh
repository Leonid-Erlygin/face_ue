docker run \
 --shm-size=4g \
 --memory=16g \
 --cpus=16 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_filter_plots \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=2"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 evaluation/evaluate.py -cn=text_osr_clinc150