import torch
import math
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
import importlib
from pathlib import Path
import numpy as np
from evaluation.ijb_evals import instantiate_list


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
    def __init__(
        self, softmax_weights_path: str, radius: int, requires_grad=False
    ) -> None:
        super().__init__()
        self.softmax_weights = torch.load(softmax_weights_path)
        softmax_weights_norm = torch.norm(
            self.softmax_weights, dim=1, keepdim=True
        )  # [N, 512]
        self.softmax_weights = (
            self.softmax_weights / softmax_weights_norm
        )  # $ w_c \in rS^{d-1} $

        self.softmax_weights = torch.nn.Parameter(
            self.softmax_weights, requires_grad=requires_grad
        )
        # self.softmax_weights = torch.nn.Parameter(
        #     torch.empty((8631, 512))
        # )
        # torch.nn.init.kaiming_uniform_(self.softmax_weights, a=math.sqrt(5))


class SphereConfidenceFace(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        scf_loss: torch.nn.Module,
        optimizer_params,
        scheduler_params,
        softmax_weights: torch.nn.Module,
        permute_batch: bool,
        validation_dataset=None,
        template_pooling_strategy=None,
        recognition_method=None,
        verification_metrics=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.head = head
        self.scf_loss = scf_loss
        self.softmax_weights = softmax_weights.softmax_weights

        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.permute_batch = permute_batch
        self.validation_step_outputs = []
        self.validation_dataset = validation_dataset
        self.template_pooling_strategy = template_pooling_strategy
        self.recognition_method = recognition_method
        if verification_metrics is not None:
            self.verification_metrics = instantiate_list(verification_metrics)
            print(self.verification_metrics)

    def forward(self, x):
        self.backbone.eval()
        backbone_outputs = self.backbone(x)
        log_kappa = self.head(backbone_outputs["bottleneck_feature"])
        return backbone_outputs["feature"], log_kappa

    def training_step(self, batch):
        images, labels = batch
        # freezing bn layers
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
        print(len(batch))
        if len(batch) == 2:
            # ms1m pred
            images_batch, labels = batch
            return self(images_batch)
        else:
            images_batch = batch
        if self.permute_batch:
            images_batch = images_batch.permute(0, 3, 1, 2)
        return self(images_batch)

    def validation_step(self, batch, batch_idx):
        images_batch = batch
        if self.permute_batch:
            images_batch = images_batch.permute(0, 3, 1, 2)
        pred = self(images_batch)
        self.validation_step_outputs.append(pred)
        return pred

    def on_validation_epoch_end(self):
        image_input_feats = (
            torch.cat([batch[0] for batch in self.validation_step_outputs], axis=0)
            .cpu()
            .numpy()
        )
        unc = (
            torch.cat([batch[1] for batch in self.validation_step_outputs], axis=0)
            .cpu()
            .numpy()
        )
        unc = np.exp(unc)
        self.validation_step_outputs.clear()

        test_dataset = self.validation_dataset
        pooled_data = self.template_pooling_strategy(
            image_input_feats,
            unc,
            test_dataset.templates,
            test_dataset.medias,
        )
        template_pooled_emb = pooled_data[0]
        template_pooled_unc = pooled_data[1]
        template_ids = np.unique(test_dataset.templates)
        scores, unc = self.recognition_method(
            template_pooled_emb,
            template_pooled_unc,
            template_ids,
            test_dataset.p1,
            test_dataset.p2,
        )
        print(scores.shape)
        metrics = {}
        for metric in self.verification_metrics:
            print(metric)
            metrics.update(
                metric(
                    scores=scores,
                    labels=test_dataset.label,
                )
            )
        print(metrics)
        for metric_name, value in metrics.items():
            if "TAR" in metric_name:
                self.log(metric_name, value)


###################


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
