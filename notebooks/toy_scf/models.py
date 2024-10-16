import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from face_lib.models.losses import ArcFaceLoss
from face_lib.models.arcface import MetricLearningModel
from torch.utils.data import DataLoader
from utils_notebooks import predict_features, compute_distance_and_visualize
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import datetime
from face_lib.models.scf import SphereConfidenceFace, SoftmaxWeights
from face_lib.models.heads import SCFHead
from face_lib.models.losses import KLDiracVMF
import pytorch_lightning
from pytorch_lightning.callbacks import LearningRateMonitor
from evaluation.visualize import (
    plot_rejection_scores,
)


# 1. Define simple model
class Backbone(nn.Module):
    def __init__(self, num_features=128):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(
            32,
            eps=1e-05,
        )
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(
            64,
            eps=1e-05,
        )
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        self.fc1 = nn.Linear(9216, num_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        sig_x = x
        x = self.fc1(x)
        x = self.features(x)
        x = F.normalize(x, p=2.0, dim=1)
        output = {
            "feature": x,
            "bottleneck_feature": sig_x,
        }
        return output


def train_arcface(
    arcface_model, dirpath: str, project: str, run_name: str = "", max_epoch=200
):
    dirpath = Path(dirpath)
    wab_logger = WandbLogger(
        project=project,
        name=run_name,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=dirpath / "ckpt",
        filename="{epoch:02d}-{val_loss:.2f}",
    )
    lr_logger = LearningRateMonitor(logging_interval="step")
    # initialize trainer, use one GPU for training
    trainer = Trainer(
        max_epochs=max_epoch,
        default_root_dir=dirpath,  # "/app/outputs/arcface_mnist",
        accelerator="gpu",
        devices=[0],
        benchmark=True,
        check_val_every_n_epoch=5,
        logger=wab_logger,
        callbacks=[checkpoint_callback, lr_logger],
    )

    # fit the model
    trainer.fit(arcface_model)
    del wab_logger


def compute_cosine_sim(arcface_model, train_dl, test_dl, file_name: str):
    predicted_train_features, train_labels = predict_features(arcface_model, train_dl)
    softmax_weights = F.normalize(arcface_model.softmax_weights, dim=1)
    wc = softmax_weights[train_labels, :].detach().cpu()
    predicted_train_features = torch.tensor(predicted_train_features)
    cosine_sim_train = torch.sum(predicted_train_features * wc, dim=1, keepdim=True)
    cosine_sim_train = cosine_sim_train[:, 0].numpy()
    np.save(f"outputs/train_{file_name}_cosine_sim.npy", cosine_sim_train)

    predicted_test_features, test_labels = predict_features(arcface_model, test_dl)
    predicted_test_features = torch.tensor(predicted_test_features)
    wc_test = softmax_weights[test_labels, :].detach().cpu()
    cosine_sim_test = torch.sum(predicted_test_features * wc_test, dim=1, keepdim=True)
    cosine_sim_test = cosine_sim_test[:, 0].numpy()
    np.save(f"outputs/test_{file_name}_cosine_sim.npy", cosine_sim_test)


def get_rejection_accuracy(model_names, features_list, fractions, softmax_weights):
    metric_scores = []
    fractions_linspace = np.linspace(fractions[0], fractions[1], fractions[2])

    for features, labels, kappa in features_list:
        accuracies = []
        unc_indexes = np.argsort(-kappa)
        for fraction in fractions_linspace:
            good_idx = unc_indexes[: int((1 - fraction) * kappa.shape[0])]
            accuracies.append(
                predict_accuracy(
                    features=features[good_idx],
                    labels=labels[good_idx],
                    softmax_weights=softmax_weights,
                )
            )

        metric_scores.append((fractions_linspace, accuracies))

    metric_pretty_name = "Accuracy"
    fig, rejection_metric_values = plot_rejection_scores(
        scores=metric_scores,
        names=model_names,
        y_label=f"{metric_pretty_name}",
    )
    return fig


def predict_accuracy(features, labels, softmax_weights):
    predictions = np.argmax(features @ softmax_weights.T, axis=-1)
    return np.mean(predictions == labels)


def compute_kappa(scf_model, dataset):
    mnist_dl = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=32,
    )
    scf_model.eval()
    features = []
    labels = []
    kappa = []
    for batch in mnist_dl:
        labels.append(batch[1].numpy())
        predictions = scf_model(batch[0])
        features.append(predictions[0].detach().cpu().numpy())
        kappa.append(torch.exp(predictions[1]).detach().cpu().numpy())
    labels = np.concatenate(labels)
    features = np.concatenate(features, axis=0)
    kappa = np.concatenate(kappa, axis=0)[:, 0]
    return features, labels, kappa


def load_arcface_model(
    mnist_ds_train, mnist_ds_test, checkpoint_path: str, visualize: bool = False
):
    scheduler_params = {
        "scheduler": "OneCycleLR",
        "params": {
            "max_lr": 3e-2,
            "steps_per_epoch": len(mnist_ds_train) // 42,
            "epochs": 42,
            "div_factor": 1e2,
            "final_div_factor": 1e2,
        },
        "interval": "step",
        "frequency": 1,
    }
    NUM_FEATURES = 2
    backbone_model = Backbone(num_features=NUM_FEATURES)
    arcface_loss = ArcFaceLoss()
    arcface_model = MetricLearningModel(
        backbone_model,
        arcface_loss,
        num_labels=10,
        train_set=mnist_ds_train,
        val_set=mnist_ds_test,
        scheduler_params=scheduler_params,
        num_features=NUM_FEATURES,
    )
    arcface_model = arcface_model.load_from_checkpoint(checkpoint_path)

    predicted_features, image_labels = predict_features(
        arcface_model, arcface_model.val_dataloader()
    )

    softmax_weights = arcface_model.softmax_weights.detach().cpu()
    softmax_weights = F.normalize(softmax_weights, dim=1).numpy()
    if visualize:
        dists = compute_distance_and_visualize(
            predicted_features, image_labels, softmax_weights, num_features=NUM_FEATURES
        )
    print(
        f"Test accuracy: {predict_accuracy(predicted_features, image_labels, softmax_weights)}"
    )
    # print(f"Average distance: {np.mean(dists)}")
    return arcface_model


def load_scf_model(mnist_ds_train, mnist_ds_test, scf_path, arcface_path, radius=1):
    optimizer_params = {
        "optimizer_path": "torch.optim",
        "optimizer_name": "AdamW",
        "params": {"lr": 3e-2, "weight_decay": 0.01},
    }
    scheduler_params = {
        "scheduler": "StepLR",
        "params": {"step_size": 10, "gamma": 0.5},
        "interval": "step",
        "frequency": 1,
    }
    arcface_model = load_arcface_model(mnist_ds_train, mnist_ds_test, arcface_path)
    arcface_model = arcface_model.load_from_checkpoint(arcface_path)
    backbone = arcface_model.backbone
    head = SCFHead(convf_dim=28 * 28, latent_vector_size=10, activation="sigmoid")
    softmax_weights = SoftmaxWeights(
        softmax_weights_path=f"outputs/softmax_weights.pt", radius=radius
    )
    scf_loss = KLDiracVMF(z_dim=2, radius=radius)
    scf_model = SphereConfidenceFace(
        backbone=backbone,
        head=head,
        scf_loss=scf_loss,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        softmax_weights=softmax_weights,
        permute_batch=False,
        predict_kappa_by_input=True,
    )

    scf_model.load_state_dict(torch.load(scf_path)["state_dict"])
    return scf_model


def train_scf(
    project,
    run_name,
    mnist_ds_train,
    mnist_ds_test,
    arcface_path: str,
    optimizer_params,
    scheduler_params,
    batch_size=400,
    radius=1,
    max_epoch=20,
    num_features=2,
):

    backbone_model = Backbone(num_features=num_features)
    arcface_loss = ArcFaceLoss()

    arcface_model = MetricLearningModel(
        backbone_model,
        arcface_loss,
        num_labels=10,
        train_set=mnist_ds_train,
        val_set=mnist_ds_test,
        num_features=num_features,
        scheduler_params=scheduler_params,
    )
    arcface_model = arcface_model.load_from_checkpoint(arcface_path)
    arcface_model.eval()
    backbone = arcface_model.backbone
    head = SCFHead(convf_dim=28 * 28, latent_vector_size=10, activation="sigmoid")
    softmax_weights = SoftmaxWeights(
        softmax_weights_path=f"outputs/softmax_weights.pt", radius=radius
    )
    scf_loss = KLDiracVMF(z_dim=2, radius=radius)
    scf_model = SphereConfidenceFace(
        backbone=backbone,
        head=head,
        scf_loss=scf_loss,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        softmax_weights=softmax_weights,
        permute_batch=False,
        predict_kappa_by_input=True,
    )

    train_dl = DataLoader(
        mnist_ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=32,
    )
    now = datetime.datetime.now()
    wab_logger = WandbLogger(project=project, name=f"{run_name}-{str(now)}")
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        dirpath=f"/app/outputs/{project}/{run_name}/ckpt/{str(now)}",
        save_last=True,
    )
    lr_callback = pytorch_lightning.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    # initialize trainer, use one GPU for training
    trainer = Trainer(
        max_epochs=max_epoch,
        default_root_dir=f"/app/outputs/{project}/{run_name}/ckpt/{str(now)}",
        accelerator="gpu",
        devices=[0],
        logger=wab_logger,
        callbacks=[checkpoint_callback, lr_callback],
    )

    # fit the model
    trainer.fit(scf_model, train_dl)
    del wab_logger
    return scf_model
