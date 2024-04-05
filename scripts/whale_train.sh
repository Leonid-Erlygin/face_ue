docker run -d \
 --shm-size=8g \
 --memory=160g \
 --user 1005:1005 \
 --name whale_train \
 --env WANDB_API_KEY=fc366b0c6dc3150de383b028451e1cfa35009932 \
 --rm \
 --init \
 -v $HOME/face_ue:/app \
 --gpus all \
 -w="/app/sandbox/happy_whale/kaggle-happywhale-1st-place" \
 face-eval \
 python -m src.train --config_path config/efficientnet_b7.yaml --exp_name b7 --save_checkpoint
