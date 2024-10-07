docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_scf_train \
 --env WANDB_API_KEY=b2c5aadfb0bf526689d07a4bb4aae1eb58faf5b9 \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=1"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 trainers/train_multiple_runs_scf.py -cn=train_scf_base
#  --config configs/train/train_scf_with_psd.yaml \
#  --ckpt_path=/app/outputs/scf_new_data/vgg_old_backbone_from_zero/epoch=11-step=174672_pretrained.ckpt
#--multirun data.train_dataset.torch_augments.0.init_args.sigma=2,3,4,5,7,9,11

#--env WANDB_API_KEY=fc366b0c6dc3150de383b028451e1cfa35009932 \