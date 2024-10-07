from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer, seed_everything
import sys
import torch
import importlib

# sys.path.append('/app/face_lib')
# sys.path.append('/home/i.kolesnikov/face_ue/face_lib')
# code based on original repo: https://github.com/MathsShen/SCF
# main config for training: configs/hydra/train_sphere_face.yaml
# simple demo classes for your convenience
# from face_lib.models.scf import SphereConfidenceFace
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="/app/configs/train/train_hydra/blur_different_sigma", config_name="train_blur.yaml")
def train_model(cfg):
    
    seed_everything(cfg.seed_everything, workers=True)

    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    checkpoint = torch.load(cfg.weights_path, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    dataclass = instantiate(cfg.data)

    trainer.fit(model = model, datamodule=dataclass)

if __name__ == "__main__":
    train_model()