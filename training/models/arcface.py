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
import random
from sklearn.metrics import roc_auc_score
from collections import defaultdict
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
            backbone_outputs = torch.nn.functional.normalize(
                backbone_outputs, p=2.0, dim=1
            )

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
        scheduler_params,
        optimizer_params,
        num_features: int,
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
        self.validation_step_outputs = []
        self.scheduler_params = scheduler_params
        self.optimizer_params = optimizer_params

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

        # Accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train_loss", loss.item(), prog_bar=True)

        with torch.no_grad():
            # Normalize class centers (ensure unit norm)
            class_centers = F.normalize(self.softmax_weights, dim=1)  # [C, D]
            # Gather centers for true labels
            true_centers = class_centers[labels]  # [B, D]
            # Cosine similarity = dot product (both L2-normalized)
            cos_sim = (features * true_centers).sum(dim=1)  # [B]
            avg_cos_sim = cos_sim.mean()

        self.log("train_cos_sim", avg_cos_sim, on_step=True)
        return {"loss": loss, "out": features, "label": labels}

    def validation_step(self, batch, batch_idx):
        # For val set with unseen authors: labels will be -1, but we need author_id metadata
        images, labels = batch

        # Forward pass
        features, logits = self(images)
        # loss = self.loss(logits, labels)
        # self.log("val_loss", loss, prog_bar=True)

        # CRITICAL: Store embeddings + author IDs for verification metrics
        # Assumes your dataloader's collate_fn passes author_id through batch metadata
        # If not, modify collate_fn to include author_id in batch dict
        self.validation_step_outputs.append(
            {
                "embeddings": features.detach().cpu(),
                "author_ids": images["author_ids"],  # See note below
                "labels": labels.detach().cpu(),
            }
        )

        # return loss

    def on_validation_epoch_end(self):
        # Aggregate embeddings and author IDs
        all_embeddings = []
        all_author_ids = []

        for batch_out in self.validation_step_outputs:
            all_embeddings.append(batch_out["embeddings"])
            all_author_ids.extend(batch_out["author_ids"])  # List of strings

        embeddings = torch.cat(all_embeddings, dim=0).numpy()  # (N, D)
        author_ids = np.array(all_author_ids)  # (N,)

        # Build author → indices mapping
        author_to_indices = defaultdict(list)
        for idx, author in enumerate(author_ids):
            author_to_indices[author].append(idx)

        # Sample verification pairs (efficient: ~2K pairs total)
        pos_pairs = []
        neg_pairs = []

        # Positive pairs: sample 2 docs from same author
        authors_with_multiple = [
            a for a, idxs in author_to_indices.items() if len(idxs) >= 2
        ]
        sampled_authors = random.sample(
            authors_with_multiple,
            min(500, len(authors_with_multiple)),  # 500 authors → ~500 pos pairs
        )

        for author in sampled_authors:
            idxs = author_to_indices[author]
            i, j = random.sample(idxs, 2)
            sim = np.dot(
                embeddings[i], embeddings[j]
            )  # Cosine (embeddings are L2-normalized)
            pos_pairs.append(sim)

        # Negative pairs: sample docs from different authors
        all_authors = list(author_to_indices.keys())
        for _ in range(len(pos_pairs)):  # Match pos pair count
            a1, a2 = random.sample(all_authors, 2)
            i = random.choice(author_to_indices[a1])
            j = random.choice(author_to_indices[a2])
            sim = np.dot(embeddings[i], embeddings[j])
            neg_pairs.append(sim)

        # Compute ROCAUC
        scores = np.array(pos_pairs + neg_pairs)
        labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))

        try:
            roc_auc = roc_auc_score(labels, scores)
        except ValueError:
            roc_auc = 0.5  # Degenerate case (all scores identical)

        # Also compute TAR@FAR=0.01 (more interpretable for authorship)
        sorted_idx = np.argsort(-scores)  # Descending
        sorted_labels = labels[sorted_idx]
        far_threshold_idx = int(0.01 * len(neg_pairs))
        if far_threshold_idx < len(sorted_labels):
            tar_far01 = sorted_labels[:far_threshold_idx].mean()
        else:
            tar_far01 = 0.0

        # Log metrics
        self.log("val_verif_rocauc", roc_auc, prog_bar=True)
        self.log("val_verif_tar_far0.01", tar_far01, prog_bar=False)
        self.log("val_verif_pos_mean", np.mean(pos_pairs), prog_bar=False)
        self.log("val_verif_neg_mean", np.mean(neg_pairs), prog_bar=False)

        # Clear outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizer = getattr(
            importlib.import_module(self.optimizer_params["optimizer_path"]),
            self.optimizer_params["optimizer_name"],
        )(
            params,
            **self.optimizer_params["params"],
        )

        scheduler_kwargs = dict(self.scheduler_params["params"])
        # Tool-routing datasets are prepared dynamically, so their number of
        # batches can change with the class-disjoint protocol.  Allow OneCycleLR
        # to use Lightning's resolved stepping budget instead of baking a stale
        # steps_per_epoch into the YAML. Existing configs without ``auto`` keep
        # their previous behavior unchanged.
        if scheduler_kwargs.get("total_steps") == "auto":
            scheduler_kwargs["total_steps"] = int(self.trainer.estimated_stepping_batches)
            if scheduler_kwargs["total_steps"] <= 0:
                raise ValueError("Could not resolve a positive OneCycleLR total_steps")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.scheduler_params["scheduler"],
                )(optimizer, **scheduler_kwargs),
                "interval": self.scheduler_params["interval"],
            },
        }

    # def train_dataloader(self) -> DataLoader:
    #     """Create training dataloader."""
    #     return DataLoader(
    #         self.train_set,
    #         batch_size=self.hparams.batch_size,
    #         shuffle=True,
    #         drop_last=True,
    #         num_workers=self.hparams.num_workers,
    #     )

    # def val_dataloader(self) -> DataLoader:
    #     """Create velidation dataloader."""
    #     return DataLoader(
    #         self.val_set,
    #         batch_size=self.hparams.batch_size,
    #         shuffle=False,
    #         drop_last=False,
    #         num_workers=self.hparams.num_workers,
    #     )
