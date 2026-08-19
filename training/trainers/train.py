from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer, seed_everything
import sys
import torch
import importlib

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="/app/configs/uncertainty_models")
def train_model(cfg):
    print(cfg)
    seed_everything(cfg.seed_everything, workers=True)

    trainer = instantiate(cfg.trainer)
    dataclass = instantiate(cfg.data)

    # Some modern tasks (e.g. ToolBench tool routing) build a class-disjoint
    # protocol at preparation time, so the number of known classes is data driven.
    # Resolve ``num_labels: auto`` only for configs that explicitly opt into it;
    # existing fixed-class experiments are unchanged.
    if str(cfg.model.get("num_labels", "")) == "auto":
        dataclass.setup(stage="fit")
        num_classes = getattr(dataclass, "num_classes", None)
        if num_classes is None:
            raise ValueError(
                "model.num_labels=auto requires the datamodule to expose num_classes after setup('fit')"
            )
        OmegaConf.set_struct(cfg, False)
        cfg.model.num_labels = int(num_classes)
        OmegaConf.set_struct(cfg, True)
        print(f"Resolved model.num_labels={num_classes} from {type(dataclass).__name__}")

    model = instantiate(cfg.model)
    if hasattr(cfg, "weights_path"):
        checkpoint = torch.load(cfg.weights_path, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    if cfg.mode == "train":
        for logger in trainer.loggers:
            logger.log_hyperparams(OmegaConf.to_container(cfg))
        trainer.fit(model=model, datamodule=dataclass)
    elif cfg.mode == "predict":
        trainer.predict(model=model, datamodule=dataclass)
    else:
        raise ValueError


if __name__ == "__main__":
    train_model()
