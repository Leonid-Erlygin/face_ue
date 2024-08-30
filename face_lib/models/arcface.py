import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import importlib
import pickle
from decimal import Decimal
from pathlib import Path
import numpy as np
import onnx2torch

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
        self.backbone = torch.load(backbone)
        self.backbone.eval()

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
            backbone_outputs = self.backbone(x)
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
        rows = torch.arange(logits.shape[0])
        self.log("cos distance", torch.mean(logits[rows,labels]).item(), prog_bar=True)
        loss = self.arcface_loss(logits, labels)

        # log loss value
        self.log("train_loss", loss.item(), prog_bar=True)
        #self.log("cos distance", torch.mean(torch.max(logits, dim = 1)[0]).item(), prog_bar=True)
        #self.log("cos distance", torch.mean(logits[:,labels]).item(), prog_bar=True)
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

