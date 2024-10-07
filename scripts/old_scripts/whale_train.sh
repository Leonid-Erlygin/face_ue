docker run \
 --shm-size=8g \
 --memory=160g \
 --gpus '"device=0"' \
 --user 1005:1005 \
 --name whale_train \
 --env WANDB_API_KEY=b2c5aadfb0bf526689d07a4bb4aae1eb58faf5b9 \
 --rm \
 --init \
 -v $HOME/face_ue:/app \
 -w="/app/sandbox/happy_whale/kaggle-happywhale-1st-place" \
 face-eval \
 python -m src.train --config_path config/efficientnet_b6_new.yaml --exp_name b6_bottleneck_feature_fix_nb --save_checkpoint --wandb_logger
