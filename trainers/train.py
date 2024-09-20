from pytorch_lightning.cli import LightningCLI
import sys

sys.path.append("/app")
sys.path.append("/app/sandbox/AdaFace")
# code based on original repo: https://github.com/MathsShen/SCF
# main config for training: configs/hydra/train_sphere_face.yaml
# simple demo classes for your convenience

#  resume_from_checkpoint: /app/outputs/scf_train/weights/epoch=0-step=45000-v1.ckpt


def cli_main():
    cli = LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
