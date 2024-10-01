import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import importlib
import pickle
import wandb
from decimal import Decimal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Dataset
import math
from typing import List

from typing import Tuple, Dict


class ArcFace_SW(LightningModule):
    def __init__(
        self,
        backbone,
        arcface_loss: torch.nn.Module,
        optimizer_params,
        scheduler_params,
        permute_batch: bool,
        softmax_weights: torch.nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.backbone.eval()

        self.arcface_loss = arcface_loss
        self.softmax_weights = softmax_weights.softmax_weights
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.permute_batch = permute_batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        :param x: batch of images
        :return a tuple of:
            - features: outputs of the backbone model a.k.a. embeddings
            - logits: result of the last linear transformations
        """
        with torch.no_grad():
            backbone_outputs = self.backbone(x)["feature"]
            backbone_outputs = torch.nn.functional.normalize(backbone_outputs, p=2.0, dim=1)

        norm_weights = F.normalize(self.softmax_weights, dim=1)
        logits = F.linear(backbone_outputs, norm_weights)

        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do a training step of the model.

        :param batch: batch of input images and labels
        :param idx: batch number
        :return: value of the loss function
        """
        images, labels = batch
        logits = self(images)

        loss = self.arcface_loss(logits, labels)

        # log loss value
        self.log("train_loss", loss.item(), prog_bar=True)
        self.log(
            "cos distance",
            torch.mean(torch.max(logits, dim=1)[0]).item(),
            prog_bar=True,
        )

        return loss

    # def forward(self, x):
    #     backbone_outputs = self.backbone(x)
    #     backbone_outputs = torch.nn.functional.normalize(backbone_outputs, p=2.0, dim=1)
    #     return backbone_outputs

    # def training_step(self, batch):
    #     images, labels = batch

    #     feature = self(images)

    #     wc = self.softmax_weights
    #     cosine = feature @ wc.T
    #     new_cosine, index = self.arcface_loss(cosine, labels, 64)

    #     total_loss = torch.nn.CrossEntropyLoss()(new_cosine, labels)

    #     self.log("train_loss", total_loss.item(), prog_bar=True)
    #     self.log("cos", cosine.mean().item())

    #     return total_loss

    def configure_optimizers(self):
        optimizer = getattr(
            importlib.import_module(self.optimizer_params["optimizer_path"]),
            self.optimizer_params["optimizer_name"],
        )(
            [self.softmax_weights],
            **self.optimizer_params["params"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.scheduler_params["scheduler"],
                )(optimizer, **self.scheduler_params["params"]),
                # "interval": "step",
            },
        }

    def predict_step(self, batch, batch_idx):
        if len(batch) == 2:
            # ms1m pred
            images_batch, labels = batch
            return self(images_batch)
        else:
            images_batch = batch
        if self.permute_batch:
            images_batch = images_batch.permute(0, 3, 1, 2)
        return self(images_batch)


class MetricLearningModel(LightningModule):
    """Lightning wrapper for a Metric Learning model."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        loss: torch.nn.Module,
        num_labels: int,
        train_set: Dataset,
        val_set: Dataset,
        num_features: int = 2,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-5,
        num_workers: int = 2,
    ) -> None:
        """Initialize MetricLearningModel.

        :param backbone: core deef model to be trained
        :param loss: loss function to be used
        :param num_labels: number of target classes (people)
        :param train_set - dataset with training data
        :param val_set - dataset with test data
        :param num_features - dimensionality of the feature space
        :param batch_size, learning_rate, weight_decay - model training parameters
        :param num_workers - number of CPUs to be used (for dataloaders)
        """
        super().__init__()

        self.backbone = backbone
        self.loss = loss

        # parameters of the last linear layer initialized by the 'kaiming_uniform_'
        self.softmax_weights = torch.nn.Parameter(
            torch.empty((num_labels, num_features))
        )
        torch.nn.init.kaiming_uniform_(self.softmax_weights, a=math.sqrt(5))

        self.train_set = train_set
        self.val_set = val_set
        self.validation_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        :param x: batch of images
        :return a tuple of:
            - features: outputs of the backbone model a.k.a. embeddings
            - logits: result of the last linear transformations
        """
        backbone_outputs = self.backbone(x)
        features = backbone_outputs["feature"]

        norm_weights = F.normalize(self.softmax_weights, dim=1)
        logits = F.linear(features, norm_weights)

        return features, logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do a training step of the model.

        :param batch: batch of input images and labels
        :param idx: batch number
        :return: value of the loss function
        """
        images, labels = batch
        features, logits = self(images)
        loss = self.loss(logits, labels)

        # log loss value
        self.log("train_loss", loss.item(), prog_bar=True)

        return {"loss": loss, "out": features, "label": labels}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do a validation step of the model.

        :param batch: batch of input images and labels
        :param idx: batch number
        :return: value of the loss function
        """
        images, labels = batch
        features, logits = self(images)

        loss = self.loss(logits, labels)
        # log loss value
        self.log("val_loss", loss.item(), prog_bar=True)
        self.validation_step_outputs.append(
            {"loss": loss, "out": features, "label": labels}
        )
        # return

    def on_validation_epoch_end(self) -> None:
        """Compute metrics and log figures at every validation epoch.

        :param outputs - List validation_step() outputs (List of dicts in our case)
        """
        # aggreaget predicted features and labels; need to use CPU for matplotlib
        outputs = self.validation_step_outputs
        features = (
            torch.vstack([batch_out["out"] for batch_out in outputs]).detach().cpu()
        )
        labels = torch.hstack([batch_out["label"] for batch_out in outputs]).cpu()

        # get normalized softmax weights for visualization
        weights = F.normalize(self.softmax_weights, p=2, dim=1).detach().cpu()

        dists = []
        colors = list(mcolors.TABLEAU_COLORS)[: self.hparams.num_labels]
        self.log("val_avg_distance", np.mean(dists), prog_bar=True)

        # if self.hparams.num_features == 2:
        #     # plot feature space in 2D
        #     fig = plt.figure(figsize=(6, 6))
        #     for i, (center, color) in enumerate(zip(weights, colors)):
        #         points = features[labels == i]

        #         # compute average distance in the current class
        #         dists.append(((points - center) ** 2).sum(axis=1).mean().item())

        #         # visualize the results
        #         x, y = [0, center[0]], [0, center[1]]
        #         plt.plot(x, y, marker="", c=color)
        #         plt.scatter(points[:, 0], points[:, 1], color=color, s=3)

        #     plt.gca().set_aspect("equal")
        #     plt.axis("off")
        #     plt.title("Feature space visualization", fontsize=14)

        # elif self.hparams.num_features == 3:
        #     # plot feature space in 3D
        #     fig = plt.figure(figsize=(6, 6))
        #     ax = plt.axes(projection="3d")

        #     u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
        #     x = np.cos(u) * np.sin(v)
        #     y = np.sin(u) * np.sin(v)
        #     z = np.cos(v)

        #     ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, rstride=2, cstride=2)

        #     for i, (center, color) in enumerate(zip(weights, colors)):
        #         points = features[labels == i]

        #         dists.append(((points - center) ** 2).sum(axis=1).mean().item())

        #         x, y, z = [0, center[0]], [0, center[1]], [0, center[2]]
        #         ax.plot3D(x, y, z, marker="", c=color)
        #         ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color=color, s=3)

        #     plt.gca().set_aspect("equal")
        #     plt.axis("off")
        #     plt.title("Feature space visualization", fontsize=14)

        # log matplotlib.figure() and the metrics to Logger
        # figure logging works only with several loggers (e.g. comet)
        # self.logger.experiment.log_figure(f"val_picture_num_{self.current_epoch}", plt)
        # if self.hparams.num_features == 2:
        #     image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #     image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #     images = wandb.Image(image_array)
        #     self.log({"Rays": images})

        # plt.show()
        # plt.clf()

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Create optimizer for model training."""
        params = list(self.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Create velidation dataloader."""
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )
