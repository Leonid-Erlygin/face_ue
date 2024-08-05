import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BasePredictionWriter
import importlib
import pickle
from pathlib import Path
import numpy as np


class Prediction_writer(BasePredictionWriter):
    def __init__(self, output_dir: str, file_name: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.file_name = file_name

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        embs = torch.cat([batch[0] for batch in predictions], axis=0).numpy()
        unc = torch.cat([batch[1] for batch in predictions], axis=0).numpy()
        print(embs.shape, unc.shape)
        np.savez(self.output_dir / f"{self.file_name}.npz", embs=embs, unc=unc)


class SoftmaxWeights(torch.nn.Module):
    def __init__(self, softmax_weights_path: str, radius: int) -> None:
        super().__init__()
        self.softmax_weights = torch.load(softmax_weights_path)
        softmax_weights_norm = torch.norm(
            self.softmax_weights, dim=1, keepdim=True
        )  # [N, 512]
        self.softmax_weights = (
            self.softmax_weights / softmax_weights_norm * radius
        )  # $ w_c \in rS^{d-1} $

        self.softmax_weights = torch.nn.Parameter(
            self.softmax_weights, requires_grad=False
        )


class SphereConfidenceFace(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        scf_loss: torch.nn.Module,
        optimizer_params,
        scheduler_params,
        permute_batch: bool,
        softmax_weights: torch.nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.scf_loss = scf_loss
        if softmax_weights is None:
            # assume that weights are stored in the backbone
            self.softmax_weights = self.backbone.backbone.head_id.weight.data
            delattr(self.backbone.backbone, "head_id")
            softmax_weights_norm = torch.norm(
                self.softmax_weights, dim=1, keepdim=True
            )  # [N, 1]
            self.softmax_weights = (
                self.softmax_weights / softmax_weights_norm * scf_loss.radius
            )  # $ w_c \in rS^{d-1} $
            self.softmax_weights = torch.nn.Parameter(
                self.softmax_weights, requires_grad=False
            )
        else:
            self.softmax_weights = softmax_weights.softmax_weights
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.permute_batch = permute_batch

    def forward(self, x):
        backbone_outputs = self.backbone(x)
        log_kappa = self.head(backbone_outputs["bottleneck_feature"])
        return backbone_outputs["feature"], log_kappa

    def training_step(self, batch):
        images, labels = batch
        # freezing bn layers
        self.backbone.backbone.eval()
        feature, log_kappa = self(images)
        kappa = torch.exp(log_kappa)
        wc = self.softmax_weights[labels, :]
        losses, l1, l2, l3, cos = self.scf_loss(feature, kappa, wc)

        kappa_mean = kappa.mean()
        total_loss = losses.mean()

        self.log("train_loss", total_loss.item(), prog_bar=True)
        self.log("kappa", kappa_mean.item())
        self.log("l1", l1.mean().item())
        self.log("l2", l2.mean().item())
        self.log("l3", l3.mean().item())
        self.log("cos", cos.mean().item())

        return total_loss

    def configure_optimizers(self):
        optimizer = getattr(
            importlib.import_module(self.optimizer_params["optimizer_path"]),
            self.optimizer_params["optimizer_name"],
        )(
            [*self.head.parameters()],
            **self.optimizer_params["params"],
        )  #!!! добавил бэкбон
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
        images_batch = batch
        if self.permute_batch:
            images_batch = images_batch.permute(0, 3, 1, 2)

        return self(images_batch)

    # def validation_step(self, batch, batch_idx):
    #     self._shared_eval(batch, batch_idx, "val")

    # def test_step(self, batch, batch_idx):
    #     self._shared_eval(batch, batch_idx, "test")

    # def _shared_eval(self, batch, batch_idx, prefix):
    #     x, _ = batch
    #     x_hat = self.auto_encoder(x)
    #     loss = self.metric(x, x_hat)
    #     self.log(f"{prefix}_loss", loss)


class SphereConfidenceFaceV2(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        embed_head: torch.nn.Module,
        scf_loss: torch.nn.Module,
        optimizer_params,
        scheduler_params,
        permute_batch: bool,
        softmax_weights: torch.nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.embed_head = embed_head
        self.scf_loss = scf_loss
        if softmax_weights is None:
            # assume that weights are stored in the backbone
            self.softmax_weights = self.backbone.backbone.head_id.weight.data
            delattr(self.backbone.backbone, "head_id")
            softmax_weights_norm = torch.norm(
                self.softmax_weights, dim=1, keepdim=True
            )  # [N, 1]
            self.softmax_weights = (
                self.softmax_weights / softmax_weights_norm * scf_loss.radius
            )  # $ w_c \in rS^{d-1} $
            self.softmax_weights = torch.nn.Parameter(
                self.softmax_weights, requires_grad=False
            )
        else:
            self.softmax_weights = softmax_weights.softmax_weights
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.permute_batch = permute_batch

    def forward(self, x):
        backbone_outputs = self.backbone(x)
        log_kappa = self.head(backbone_outputs["bottleneck_feature"])
        new_embed = self.embed_head(backbone_outputs["bottleneck_feature"])
        return new_embed, log_kappa

    def training_step(self, batch):
        images, labels = batch

        # freezing bn layers
        self.backbone.backbone.eval()

        feature, log_kappa = self(images)

        kappa = torch.exp(log_kappa)
        wc = self.softmax_weights[labels, :]
        losses, l1, l2, l3, cos = self.scf_loss(feature, kappa, wc)

        kappa_mean = kappa.mean()
        total_loss = losses.mean()

        self.log("train_loss", total_loss.item(), prog_bar=True)
        self.log("kappa", kappa_mean.item())
        self.log("l1", l1.mean().item())
        self.log("l2", l2.mean().item())
        self.log("l3", l3.mean().item())
        self.log("cos", cos.mean().item())

        return total_loss

    def configure_optimizers(self):
        optimizer = getattr(
            importlib.import_module(self.optimizer_params["optimizer_path"]),
            self.optimizer_params["optimizer_name"],
        )(
            [*self.head.parameters(), *self.embed_head.parameters()],
            **self.optimizer_params["params"],
        )  #!!! добавил бэкбон
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

    # Trying 2 optimizers - need to rewrite training step
    # def configure_optimizers(self):
    #     optimizer_kappa = getattr(
    #         importlib.import_module(self.optimizer_params["optimizer_path"]),
    #         self.optimizer_params["optimizer_name"],
    #     )(
    #         [*self.head.parameters()],
    #         **self.optimizer_params["params"],
    #     )  #!!! добавил новую голову
    #     optimizer_embed = getattr(
    #         importlib.import_module(self.optimizer_params["optimizer_path"]),
    #         self.optimizer_params["optimizer_name"],
    #     )(
    #         [*self.embed_head.parameters()],
    #         **self.optimizer_params["params"],
    #     )

    #     return (
    #         {
    #         "optimizer": optimizer_kappa,
    #         "lr_scheduler": {
    #             "scheduler": getattr(
    #                 importlib.import_module("torch.optim.lr_scheduler"),
    #                 self.scheduler_params["scheduler"],
    #             )(optimizer_kappa, **self.scheduler_params["params"]),
    #             # "interval": "step",
    #         },
    #         },
    #         {
    #         "optimizer": optimizer_embed,
    #         "lr_scheduler": {
    #             "scheduler": getattr(
    #                 importlib.import_module("torch.optim.lr_scheduler"),
    #                 self.scheduler_params["scheduler"],
    #             )(optimizer_embed, **self.scheduler_params["params"]),
    #             # "interval": "step",
    #         },
    #     }
    #     )

    def predict_step(self, batch, batch_idx):
        images_batch = batch
        if self.permute_batch:
            images_batch = images_batch.permute(0, 3, 1, 2)

        return self(images_batch)
