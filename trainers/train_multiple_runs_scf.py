from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer, seed_everything
import sys
import torch
import importlib

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import sys

sys.path.append("/app")


@hydra.main(version_base=None, config_path="/app/configs/train/train_hydra")
def train_model(cfg):

    seed_everything(cfg.seed_everything, workers=True)

    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    if hasattr(cfg, "weights_path"):
        checkpoint = torch.load(cfg.weights_path, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
    dataclass = instantiate(cfg.data)

    trainer.fit(model=model, datamodule=dataclass)


if __name__ == "__main__":
    train_model()
