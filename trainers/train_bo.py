from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer, seed_everything
import sys
import torch
import importlib
import wandb
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll

from botorch.optim import optimize_acqf
from botorch.acquisition import LogExpectedImprovement
import gpytorch
import numpy as np


from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import sys

sys.path.append("/app")
import pandas as pd


def is_recorded(point, considered_points):
    is_in = False
    for considered_point in considered_points[:, :2]:
        if np.allclose(point, considered_point):
            is_in = True
    return is_in


def visualize(train_X, Y, run, cfg, target_metric: float):
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
    )
    # log metric
    run.log({cfg.target_metric: target_metric})

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    gp.likelihood.eval()

    probabilities = torch.linspace(*cfg.bounds.torch_probability, 30)
    sigmas = torch.linspace(*cfg.bounds.sigma, 30)
    product = torch.cartesian_prod(probabilities, sigmas)
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = gp.posterior(product)
    lower, upper = observed_pred.confidence_region()

    logEI = LogExpectedImprovement(model=gp, best_f=Y.max())
    acq = logEI(product[:, None, :])

    fig = go.Figure()
    trace1 = go.Mesh3d(
        x=(product[:, 0]),
        y=(product[:, 1]),
        z=(observed_pred.mean.numpy()[:, 0]),
        opacity=0.5,
        color="rgba(244,22,100,0.6)",
    )
    fig.add_trace(trace1)

    trace_lower = go.Mesh3d(
        x=(product[:, 0]),
        y=(product[:, 1]),
        z=(lower.numpy()),
        opacity=0.5,
        color="blue",
    )
    fig.add_trace(trace_lower)

    trace_upper = go.Mesh3d(
        x=(product[:, 0]),
        y=(product[:, 1]),
        z=(upper.numpy()),
        opacity=0.5,
        color="blue",
    )
    fig.add_trace(trace_upper)

    trace2 = go.Scatter3d(x=train_X[:, 0], y=train_X[:, 1], z=Y[:, 0], mode="markers")

    fig.add_trace(trace2)

    fig.update_layout(
        scene=dict(xaxis_title="p", yaxis_title="sigma", zaxis_title="Accuracy"),
        # zaxis=dict(title="accuracy"),
    )
    run.log({"GP predictions": fig})

    fig = go.Figure()
    trace_acq = go.Mesh3d(
        x=(product[:, 0]),
        y=(product[:, 1]),
        z=(acq.detach().numpy()),
        opacity=0.5,
        color="blue",
    )
    fig.add_trace(trace_acq)

    fig.update_layout(
        scene=dict(xaxis_title="p", yaxis_title="sigma", zaxis_title="Acq"),
        # zaxis=dict(title="accuracy"),
    )
    run.log({"Acq function": fig})


def compute_metric(cfg, p, sigma):
    cfg.trainer.logger.name = f"p={p};s={sigma}"
    cfg.data.train_dataset.torch_probability = float(p)
    cfg.data.train_dataset.torch_augments[0].init_args.sigma = float(sigma)
    print(cfg.trainer.logger.name)
    print(cfg.data.train_dataset.torch_probability)
    print(cfg.data.train_dataset.torch_augments[0].init_args.sigma)
    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    dataclass = instantiate(cfg.data)
    trainer.fit(model=model, datamodule=dataclass)

    log_file_path = (
        f"/app/outputs/scf_multiple/scf_bo/p={p};s={sigma}/version_0/metrics.csv"
    )
    df = pd.read_csv(log_file_path)
    return df[cfg.target_metric].iloc[-1]
    # print([point[0], point[1], np.round(, 6)])


@hydra.main(version_base=None, config_path="/app/configs/train/train_hydra")
def train_model(cfg):
    # plot logger

    run = wandb.init(project="SCF-BO", name="plots")

    # seed_everything(cfg.seed_everything, workers=True)
    initial_points = np.array(cfg.initial_points)
    considered_points = np.array(cfg.considered_points)

    # initial points train
    # for point in initial_points:
    #     if len(cfg.considered_points)==0 or is_recorded(point, considered_points) is False:
    #         # train model
    #         cfg.trainer.logger.name = f'p={point[0]};s={point[1]}'
    #         cfg.data.train_dataset.torch_probability=float(point[0])
    #         cfg.data.train_dataset.torch_augments[0].init_args.sigma = float(point[1])
    #         print(cfg.trainer.logger.name)
    #         print(cfg.data.train_dataset.torch_probability)
    #         print(cfg.data.train_dataset.torch_augments[0].init_args.sigma)
    #         trainer = instantiate(cfg.trainer)
    #         model = instantiate(cfg.model)
    #         dataclass = instantiate(cfg.data)
    #         trainer.fit(model=model, datamodule=dataclass)

    # look at metrics

    # for point in initial_points:
    #     log_file_path = f'/app/outputs/scf_multiple/scf_bo/p={point[0]};s={point[1]}/version_0/metrics.csv'
    #     df = pd.read_csv(log_file_path)
    #     print([point[0], point[1], np.round(df[cfg.target_metric].iloc[-1], 6)])

    train_X = torch.tensor(considered_points[:, [0, 1]], dtype=torch.double)
    Y = torch.tensor(considered_points[:, [2]], dtype=torch.double)

    for i in range(cfg.num_points):
        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=Y,
            input_transform=Normalize(d=2),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        logEI = LogExpectedImprovement(model=gp, best_f=Y.max())
        bounds = torch.tensor(
            [
                [cfg.bounds.torch_probability[0], cfg.bounds.sigma[0]],
                [cfg.bounds.torch_probability[1], cfg.bounds.sigma[1]],
            ]
        ).to(torch.double)
        candidate, acq_value = optimize_acqf(
            logEI,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        target_metric = compute_metric(
            cfg=cfg,
            p=candidate[0, 0].item(),
            sigma=candidate[0, 1].item(),
        )
        train_X = torch.concatenate([train_X, candidate])
        Y = torch.concatenate([Y, torch.tensor([[target_metric]], dtype=torch.double)])

        visualize(train_X, Y, run, cfg, target_metric)
        print(train_X)
        print(Y)
        print(f"iteration {i}: {cfg.target_metric}={target_metric}")

    # visualize
    # fit gp

    train_X = torch.tensor(considered_points[:, [0, 1]], dtype=torch.double)
    Y = torch.tensor(considered_points[:, [2]], dtype=torch.double)


if __name__ == "__main__":
    train_model()
