#!/usr/bin/env python3
"""
Toy MNIST HolUE open-set recognition example.

Creates a complete dissertation-friendly toy experiment in `toy_outputs/`:

1. Trains a simple ArcFace-like 2D embedding model on known MNIST classes.
2. Trains a simple SCF-like concentration head using the SCF KL-vMF objective.
3. Constructs MNIST open-set recognition validation/test protocols.
4. Computes OSR predictions at a validation-selected FPIR threshold.
5. Compares:
   - HolUE without validation calibration
   - HolUE with validation calibration
   - HolUE raw KL, GalUE, SCF, AccScr, MaxSim, Random, Oracle
6. Saves:
   - model checkpoints
   - protocol indices
   - metric tables
   - rejection curves
   - teaser figure on the 2D unit circle

All outputs are stored under `toy_outputs/`.

The no-calibration HolUE score is the predictive entropy of:
    p(c | x) = ∫ p(c | z) p(z | x) dz
computed deterministically by quadrature over the unit circle.

The calibrated HolUE score trains a small logistic regression on validation-set
HolUE/GalUE/SCF features to predict OSR errors. This is included only for
comparison; the teaser highlights the no-calibration HolUE behavior.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required for YAML config support. Install with: pip install pyyaml"
    ) from exc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from scipy.special import iv, ive, gamma, logsumexp
import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def seed_everything(seed: int = 777, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

# ---------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


def cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
def to_jsonable(obj):
    """
    Recursively convert objects to JSON-serializable Python types.

    Handles:
      - SimpleNamespace
      - dict/list/tuple
      - pathlib.Path
      - NumPy scalars/arrays
      - torch.Tensor
      - plain Python scalars
    """
    if isinstance(obj, SimpleNamespace):
        return {k: to_jsonable(v) for k, v in vars(obj).items()}

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Last-resort fallback for unusual config objects.
    return str(obj)

def resolve_under_out_dir(path_value: str, out_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return out_dir / p


def load_yaml_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Create toy_mnist_holue_config.yaml or pass a config path as first argument."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    return data


def flatten_toy_config(raw: dict) -> SimpleNamespace:
    """
    Flatten nested YAML into attributes expected by the existing script.

    We keep `args.full_config` as the complete nested config, and expose
    top-level convenience attributes for minimal changes in old code.
    """
    output = raw["output"]
    runtime = raw["runtime"]
    data = raw["data"]
    protocol = raw["protocol"]
    model = raw["model"]
    training = raw["training"]
    osr = raw["osr"]
    integration = raw["integration"]
    calibration = raw["calibration"]
    evaluation = raw["evaluation"]
    three_d = raw["three_d"]

    flat = {}

    # Runtime
    flat["seed"] = int(runtime["seed"])
    flat["deterministic"] = bool(runtime.get("deterministic", True))
    flat["device"] = str(runtime.get("device", "auto"))

    # Output paths
    flat["out_dir"] = str(output["out_dir"])
    flat["data_dir"] = str(output.get("data_dir", "mnist_data"))
    flat["model_dir"] = str(output.get("model_dir", "models"))
    flat["protocol_dir"] = str(output.get("protocol_dir", "protocols"))
    flat["result_dir_2d"] = str(output.get("result_dir_2d", "results"))
    flat["figure_dir_2d"] = str(output.get("figure_dir_2d", "figures"))
    flat["result_dir_3d"] = str(output.get("result_dir_3d", "results_3d"))
    flat["figure_dir_3d"] = str(output.get("figure_dir_3d", "figures_3d"))

    checkpoints = output.get("checkpoints", {})
    flat["arcface_checkpoint_2d"] = checkpoints.get("arcface_2d", "tiny_arcface_mnist_2d.pt")
    flat["scf_checkpoint_2d"] = checkpoints.get("scf_2d", "tiny_scf_mnist_2d.pt")
    flat["arcface_checkpoint_3d"] = checkpoints.get("arcface_3d", "tiny_arcface_mnist_3d.pt")
    flat["scf_checkpoint_3d"] = checkpoints.get("scf_3d", "tiny_scf_mnist_3d.pt")

    # Data
    flat["mnist_download"] = bool(data.get("mnist_download", True))
    flat["known_classes"] = [int(x) for x in data["known_classes"]]
    flat["unknown_classes"] = [int(x) for x in data["unknown_classes"]]

    # Protocol
    flat["train_per_known_class"] = int(protocol["train_per_known_class"])
    flat["val_gallery_per_class"] = int(protocol["val_gallery_per_class"])
    flat["val_probe_known_per_class"] = int(protocol["val_probe_known_per_class"])
    flat["val_probe_unknown_per_class"] = int(protocol["val_probe_unknown_per_class"])
    flat["test_gallery_per_class"] = int(protocol["test_gallery_per_class"])
    flat["test_probe_known_per_class"] = int(protocol["test_probe_known_per_class"])
    flat["test_probe_unknown_per_class"] = int(protocol["test_probe_unknown_per_class"])
    flat["corrupt_known_frac"] = float(protocol["corrupt_known_frac"])
    flat["corrupt_unknown_frac"] = float(protocol["corrupt_unknown_frac"])

    # Model
    flat["embedding_dim_2d"] = int(model.get("embedding_dim_2d", 2))
    flat["embedding_dim_3d"] = int(model.get("embedding_dim_3d", 3))
    flat["scf_kappa_min"] = float(model["scf"].get("kappa_min", 1.0))
    flat["scf_kappa_max"] = float(model["scf"].get("kappa_max", 80.0))

    # Training
    flat["force_train"] = bool(training.get("force_train", False))
    flat["batch_size"] = int(training["batch_size"])

    flat["arcface_epochs"] = int(training["arcface"]["epochs_2d"])
    flat["arcface_epochs_3d"] = int(training["arcface"]["epochs_3d"])
    flat["lr_arcface"] = float(training["arcface"]["lr"])
    flat["arcface_s"] = float(training["arcface"].get("scale", 16.0))
    flat["arcface_m"] = float(training["arcface"].get("margin", 0.30))
    flat["arcface_weight_decay"] = float(training["arcface"].get("weight_decay", 1e-4))

    flat["scf_epochs"] = int(training["scf"]["epochs_2d"])
    flat["scf_epochs_3d"] = int(training["scf"]["epochs_3d"])
    flat["lr_scf"] = float(training["scf"]["lr"])
    flat["scf_weight_decay"] = float(training["scf"].get("weight_decay", 1e-4))
    flat["scf_kappa_regularizer"] = float(training["scf"].get("kappa_regularizer", 1e-4))
    
    train_corr = training.get("train_corruption", {})

    arc_corr = train_corr.get("arcface", {})
    scf_corr = train_corr.get("scf", {})

    flat["train_corrupt_arcface_enabled"] = bool(arc_corr.get("enabled", False))
    flat["train_corrupt_arcface_probability"] = float(arc_corr.get("probability", 0.0))
    flat["train_corrupt_arcface_seed_shift"] = int(arc_corr.get("seed_shift", 30000))

    flat["train_corrupt_scf_enabled"] = bool(scf_corr.get("enabled", False))
    flat["train_corrupt_scf_probability"] = float(scf_corr.get("probability", 0.0))
    flat["train_corrupt_scf_seed_shift"] = int(scf_corr.get("seed_shift", 40000))
    
    # OSR and integration
    flat["target_fpir"] = float(osr["target_fpir"])
    flat["gallery_kappa"] = float(osr["gallery_kappa"])
    beta_raw = osr.get("beta", "auto")
    if beta_raw is None or str(beta_raw).lower() == "auto":
        flat["osr_beta"] = None
    else:
        flat["osr_beta"] = float(beta_raw)

    # If true, use the paper-style continuous OOG prior for HolUE KL.
    # If false, fall back to the old collapsed K+1 discrete posterior.
    flat["continuous_oog"] = bool(osr.get("continuous_oog", True))
    
    flat["circle_grid"] = int(integration["circle_grid"])
    flat["sphere_theta_grid"] = int(integration["sphere_theta_grid"])
    flat["sphere_phi_grid"] = int(integration["sphere_phi_grid"])

    # Calibration
    flat["calibration_enabled"] = bool(calibration.get("enabled", True))
    flat["calibration_solver"] = str(calibration.get("solver", "lbfgs"))
    flat["calibration_max_iter"] = int(calibration.get("max_iter", 2000))
    flat["calibration_class_weight"] = calibration.get("class_weight", "balanced")
    flat["calibration_random_state"] = int(calibration.get("random_state", flat["seed"]))

    # Evaluation fractions
    frac = evaluation["rejection_fractions"]
    flat["rejection_fraction_start"] = float(frac["start"])
    flat["rejection_fraction_stop"] = float(frac["stop"])
    flat["rejection_fraction_num"] = int(frac["num"])

    # 3D
    flat["skip_3d"] = not bool(three_d.get("enabled", True))
    flat["teaser_3d_points"] = int(three_d["plotly"].get("max_probe_points", 900))

    ns = SimpleNamespace(**flat)
    ns.full_config = dict_to_namespace(raw)
    ns.plots = ns.full_config.plots
    ns.three_d = ns.full_config.three_d
    ns.corruption = ns.full_config.corruption
    ns.methods = ns.full_config.methods
    return ns


def load_toy_config() -> SimpleNamespace:
    """
    No argparse. Config path resolution:

      1. first positional argument, if provided:
           python toy_mnist_holue.py my_config.yaml

      2. environment variable:
           TOY_MNIST_HOLUE_CONFIG=my_config.yaml python toy_mnist_holue.py

      3. default:
           toy_mnist_holue_config.yaml
    """
    if len(sys.argv) > 2:
        raise ValueError(
            "This script does not use argparse. "
            "Pass at most one positional YAML config path."
        )

    if len(sys.argv) == 2:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(
            os.environ.get("TOY_MNIST_HOLUE_CONFIG", "toy_mnist_holue_config.yaml")
        )

    raw = load_yaml_file(config_path)
    args = flatten_toy_config(raw)
    args.config_path = str(config_path)

    print(f"Loaded config: {config_path.resolve()}")
    return args

# ---------------------------------------------------------------------
# Simple ArcFace / SCF implementation for 2D MNIST embeddings
# ---------------------------------------------------------------------


class ArcFaceLoss(nn.Module):
    """
    Minimal ArcFace loss matching the standard implementation:
      logits_y = cos(theta_y + m) * s
      logits_j = cos(theta_j) * s for j != y
    """

    def __init__(self, s: float = 16.0, m: float = 0.30):
        super().__init__()
        self.s = float(s)
        self.m = float(m)

    def forward(
        self, cosine_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        eps = 1e-6
        cosine_logits = torch.clamp(cosine_logits, -1.0 + eps, 1.0 - eps)

        one_hot = torch.zeros_like(cosine_logits)
        one_hot.scatter_(1, labels[:, None], 1.0)

        target_cos = torch.sum(cosine_logits * one_hot, dim=1)
        target_theta = torch.acos(target_cos)
        target_cos_m = torch.cos(target_theta + self.m)

        logits = cosine_logits.clone()
        logits = logits + one_hot * (target_cos_m[:, None] - target_cos[:, None])
        logits = logits * self.s

        return F.cross_entropy(logits, labels)


class TinyArcFaceMNIST(nn.Module):
    """
    CNN -> 128D hidden -> 2D L2-normalized embedding.
    Classifier weights are also L2-normalized, so logits are cosine similarities.
    """

    def __init__(self, num_classes: int = 5, embedding_dim: int = 2):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),  # 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),  # 7 x 7
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.PReLU(),
        )
        self.embed = nn.Linear(128, embedding_dim)

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def extract(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = self.fc(h)
        emb = self.embed(h)
        emb = F.normalize(emb, p=2.0, dim=1)
        return h, emb

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, emb = self.extract(x)
        w = F.normalize(self.weight, p=2.0, dim=1)
        logits = F.linear(emb, w)
        return emb, logits

    def class_centers(self) -> torch.Tensor:
        return F.normalize(self.weight.detach(), p=2.0, dim=1)


class SCFVMFLoss(nn.Module):
    """
    SCF-style negative log-likelihood / KL-to-Dirac objective for vMF.

    Supports:
      - embedding_dim = 2: S^1, C_2(kappa)=1/(2*pi*I_0(kappa))
      - embedding_dim = 3: S^2, C_3(kappa)=kappa/(4*pi*sinh(kappa))

    Loss:
        -log p_vMF(mu_target | mu, kappa)
      = -kappa * cos(mu, target) - log C_d(kappa)
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim not in [2, 3]:
            raise ValueError("This toy SCFVMFLoss supports only embedding_dim=2 or 3.")
        self.embedding_dim = int(embedding_dim)

    @staticmethod
    def log_i0(kappa: torch.Tensor) -> torch.Tensor:
        if hasattr(torch.special, "i0e"):
            return torch.log(torch.special.i0e(kappa).clamp_min(1e-12)) + kappa
        return torch.log(torch.i0(kappa).clamp_min(1e-12))

    @staticmethod
    def log_sinh_stable(kappa: torch.Tensor) -> torch.Tensor:
        """
        Stable log(sinh(kappa)).
        """
        small = kappa < 1e-4
        out = torch.empty_like(kappa)

        out[small] = torch.log(torch.sinh(kappa[small]).clamp_min(1e-12))

        ks = kappa[~small]
        out[~small] = (
            ks - math.log(2.0) + torch.log1p(-torch.exp(-2.0 * ks)).clamp_min(-1e12)
        )

        return out

    def forward(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        class_center: torch.Tensor,
    ) -> torch.Tensor:
        cos = torch.sum(mu * class_center, dim=1, keepdim=True).clamp(-1.0, 1.0)
        kappa = kappa.clamp_min(1e-8)

        if self.embedding_dim == 2:
            # -log C_2(kappa) = log(2*pi) + log I_0(kappa)
            nll = -kappa * cos + math.log(2.0 * math.pi) + self.log_i0(kappa)

        elif self.embedding_dim == 3:
            # C_3(kappa) = kappa / (4*pi*sinh(kappa))
            # -log C_3(kappa) = log(4*pi) + log sinh(kappa) - log kappa
            nll = (
                -kappa * cos
                + math.log(4.0 * math.pi)
                + self.log_sinh_stable(kappa)
                - torch.log(kappa)
            )

        else:
            raise RuntimeError

        return nll.mean()


# Backward-compatible alias if other code still references SCF2DLoss.
class SCF2DLoss(SCFVMFLoss):
    def __init__(self):
        super().__init__(embedding_dim=2)


class TinySCFMNIST(nn.Module):
    """
    Frozen ArcFace backbone + small head predicting concentration kappa(x).

    kappa is capped for numerical stability:
      kappa in [kappa_min, kappa_min + kappa_max]
    """

    def __init__(
        self,
        arc_model: TinyArcFaceMNIST,
        hidden_dim: int = 128,
        kappa_min: float = 1.0,
        kappa_max: float = 80.0,
    ):
        super().__init__()
        self.arc_model = arc_model
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

        for p in self.arc_model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.arc_model.eval()
        with torch.no_grad():
            h, emb = self.arc_model.extract(x)

        raw = self.head(h)
        kappa = self.kappa_min + self.kappa_max * torch.sigmoid(raw)
        log_kappa = torch.log(kappa.clamp_min(1e-8))
        return emb, log_kappa


# ---------------------------------------------------------------------
# MNIST protocol and corruption utilities
# ---------------------------------------------------------------------


def get_targets(ds: datasets.MNIST) -> np.ndarray:
    targets = ds.targets
    if torch.is_tensor(targets):
        return targets.cpu().numpy().astype(int)
    return np.asarray(targets, dtype=int)


def corrupt_tensor_mnist(img: torch.Tensor, seed: int, corruption_cfg=None) -> torch.Tensor:
    """
    Deterministic corruption for MNIST tensors.

    Input/output tensor is normalized by mean=0.5, std=0.5.
    Corruption parameters are YAML-controlled.
    """
    rng = np.random.default_rng(seed)

    x = img.clone()
    x = x * 0.5 + 0.5
    x = x.clamp(0.0, 1.0)

    enabled_modes = cfg_get(
        corruption_cfg,
        "enabled_modes",
        ["occlusion", "gaussian_noise", "translation", "erase_noise"],
    )
    enabled_modes = list(enabled_modes)

    if len(enabled_modes) == 0:
        return img

    mode = str(rng.choice(enabled_modes))

    if mode == "occlusion":
        ocfg = cfg_get(corruption_cfg, "occlusion", None)

        h_min = int(cfg_get(ocfg, "h_min", 8))
        h_max = int(cfg_get(ocfg, "h_max", 15))
        w_min = int(cfg_get(ocfg, "w_min", 8))
        w_max = int(cfg_get(ocfg, "w_max", 15))
        fill = float(cfg_get(ocfg, "fill_value", 0.0))

        h = int(rng.integers(h_min, h_max + 1))
        w = int(rng.integers(w_min, w_max + 1))
        y = int(rng.integers(0, 28 - h + 1))
        z = int(rng.integers(0, 28 - w + 1))

        x[:, y : y + h, z : z + w] = fill

    elif mode == "gaussian_noise":
        ncfg = cfg_get(corruption_cfg, "gaussian_noise", None)
        std = float(cfg_get(ncfg, "std", 0.35))

        noise = torch.tensor(
            rng.normal(0.0, std, size=tuple(x.shape)),
            dtype=x.dtype,
            device=x.device,
        )
        x = (x + noise).clamp(0.0, 1.0)

    elif mode == "translation":
        tcfg = cfg_get(corruption_cfg, "translation", None)

        max_shift = int(cfg_get(tcfg, "max_shift", 5))
        fill = float(cfg_get(tcfg, "fill_value", 0.0))

        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))

        x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))

        if dy > 0:
            x[:, :dy, :] = fill
        elif dy < 0:
            x[:, dy:, :] = fill

        if dx > 0:
            x[:, :, :dx] = fill
        elif dx < 0:
            x[:, :, dx:] = fill

    elif mode == "erase_noise":
        ecfg = cfg_get(corruption_cfg, "erase_noise", None)

        h_min = int(cfg_get(ecfg, "h_min", 6))
        h_max = int(cfg_get(ecfg, "h_max", 12))
        w_min = int(cfg_get(ecfg, "w_min", 6))
        w_max = int(cfg_get(ecfg, "w_max", 12))
        noise_std = float(cfg_get(ecfg, "noise_std", 0.20))

        h = int(rng.integers(h_min, h_max + 1))
        w = int(rng.integers(w_min, w_max + 1))
        y = int(rng.integers(0, 28 - h + 1))
        z = int(rng.integers(0, 28 - w + 1))

        x[:, y : y + h, z : z + w] = float(rng.uniform(0.0, 1.0))

        noise = torch.tensor(
            rng.normal(0.0, noise_std, size=tuple(x.shape)),
            dtype=x.dtype,
            device=x.device,
        )
        x = (x + noise).clamp(0.0, 1.0)

    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    return (x - 0.5) / 0.5
# ---------------------------------------------------------------------
# 3D HolUE support on S^2
# ---------------------------------------------------------------------


def sphere_quadrature_grid(
    n_theta: int = 96,
    n_phi: int = 48,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Midpoint quadrature grid on S^2.

    Returns:
        points: (G, 3), unit vectors
        log_weights: (G,), proportional quadrature weights log(sin(phi))

    Constants cancel after softmax-normalization, so we do not need dtheta*dphi.
    """
    theta = (np.arange(n_theta) + 0.5) * (2.0 * math.pi / n_theta)
    phi = (np.arange(n_phi) + 0.5) * (math.pi / n_phi)

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    weights = np.sin(phi_grid).reshape(-1)

    log_weights = np.log(np.maximum(weights, 1e-300))
    return points.astype(np.float64), log_weights.astype(np.float64)


def holue_posterior_sphere_quadrature(
    mu: np.ndarray,
    kappa: np.ndarray,
    gallery_mu: np.ndarray,
    tau: float,
    gallery_kappa: float,
    n_theta: int = 96,
    n_phi: int = 48,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Deterministic 3D HolUE integration:

        p(c|x) = ∫_{S^2} p(c|z) p(z|x) dz

    where p(z|x) is vMF(mu(x), kappa(x)).

    Since constants cancel during weight normalization:
        weights(z_i) ∝ exp(kappa * mu^T z_i) * sin(phi_i)
    """
    mu = normalize_np(np.asarray(mu, dtype=np.float64))
    kappa = np.asarray(kappa, dtype=np.float64).reshape(-1)
    gallery_mu = normalize_np(np.asarray(gallery_mu, dtype=np.float64))

    grid_z, log_grid_w = sphere_quadrature_grid(n_theta=n_theta, n_phi=n_phi)

    post_grid = posterior_from_z(
        grid_z,
        gallery_mu=gallery_mu,
        tau=tau,
        gallery_kappa=gallery_kappa,
    )  # (G, K+1)

    out = []

    for start in range(0, len(mu), batch_size):
        end = min(start + batch_size, len(mu))
        mu_b = mu[start:end]
        kappa_b = kappa[start:end]

        cos_to_mu = mu_b @ grid_z.T  # (B, G)
        log_w = kappa_b[:, None] * cos_to_mu + log_grid_w[None, :]
        w = stable_softmax_np(log_w, axis=1)

        post = w @ post_grid
        out.append(post)

    return np.concatenate(out, axis=0)


def compute_osr_stats_nd(
    probe_mu: np.ndarray,
    probe_kappa: np.ndarray,
    probe_labels: np.ndarray,
    gallery_mu: np.ndarray,
    tau: float,
    gallery_kappa: float,
    circle_grid: int = 720,
    sphere_theta_grid: int = 96,
    sphere_phi_grid: int = 48,
    beta: Optional[float] = None,
    continuous_oog: bool = True,
) -> OSRStats:
    """
    Same as compute_osr_stats, but supports embedding_dim=2 and embedding_dim=3.

    If continuous_oog=True, this uses the paper-style mixed
    discrete-continuous prior over identities.
    """
    probe_mu = normalize_np(probe_mu.astype(np.float64))
    gallery_mu = normalize_np(gallery_mu.astype(np.float64))
    probe_kappa = np.asarray(probe_kappa, dtype=np.float64).reshape(-1)

    dim = probe_mu.shape[1]
    K = gallery_mu.shape[0]

    if beta is None:
        beta = beta_from_tau_mixed_prior(
            dim=dim,
            tau=tau,
            gallery_kappa=gallery_kappa,
            K=K,
        )

    sim = probe_mu @ gallery_mu.T
    max_sim = np.max(sim, axis=1)
    pred_idx = np.argmax(sim, axis=1)
    rejected = max_sim < tau

    if continuous_oog:
        gal_post = posterior_from_z_mixed_prior(
            probe_mu,
            gallery_mu=gallery_mu,
            gallery_kappa=gallery_kappa,
            beta=beta,
        )

        if dim == 2:
            hol_post, kl_known, kl_oog, kl_total = mixed_holue_posterior_circle_quadrature(
                mu=probe_mu,
                kappa=probe_kappa,
                gallery_mu=gallery_mu,
                gallery_kappa=gallery_kappa,
                beta=beta,
                n_grid=circle_grid,
            )

        elif dim == 3:
            hol_post, kl_known, kl_oog, kl_total = mixed_holue_posterior_sphere_quadrature(
                mu=probe_mu,
                kappa=probe_kappa,
                gallery_mu=gallery_mu,
                gallery_kappa=gallery_kappa,
                beta=beta,
                n_theta=sphere_theta_grid,
                n_phi=sphere_phi_grid,
            )

        else:
            raise ValueError(f"Only 2D and 3D toy embeddings are supported, got dim={dim}")

    else:
        gal_post = posterior_from_z(
            probe_mu,
            gallery_mu=gallery_mu,
            tau=tau,
            gallery_kappa=gallery_kappa,
        )

        if dim == 2:
            hol_post = holue_posterior_quadrature(
                mu=probe_mu,
                kappa=probe_kappa,
                gallery_mu=gallery_mu,
                tau=tau,
                gallery_kappa=gallery_kappa,
                n_grid=circle_grid,
            )

        elif dim == 3:
            hol_post = holue_posterior_sphere_quadrature(
                mu=probe_mu,
                kappa=probe_kappa,
                gallery_mu=gallery_mu,
                tau=tau,
                gallery_kappa=gallery_kappa,
                n_theta=sphere_theta_grid,
                n_phi=sphere_phi_grid,
            )

        else:
            raise ValueError(f"Only 2D and 3D toy embeddings are supported, got dim={dim}")

        kl_known, kl_oog, kl_total = discrete_kl_components(hol_post)

    gal_entropy = entropy_normalized(gal_post)
    hol_entropy = entropy_normalized(hol_post)

    return OSRStats(
        labels=np.asarray(probe_labels, dtype=int),
        sim=sim,
        max_sim=max_sim,
        pred_idx=pred_idx,
        rejected=rejected,
        scf_kappa=probe_kappa,
        gal_posterior=gal_post,
        holue_posterior=hol_post,
        gal_entropy=gal_entropy,
        holue_entropy=hol_entropy,
        kl_known=kl_known,
        kl_oog=kl_oog,
        kl_total=kl_total,
    )

class KnownMNISTDataset(Dataset):
    """
    Training dataset containing only known classes with labels remapped to 0..K-1.

    It optionally applies deterministic corruption with probability p. This is used
    as train-time augmentation for both ArcFace and SCF.

    Important:
      - ArcFace can use smaller corruption probability.
      - SCF can use larger corruption probability so it learns lower kappa on
        degraded samples.
    """

    def __init__(
        self,
        base: datasets.MNIST,
        indices: np.ndarray,
        class_to_idx: Dict[int, int],
        corruption_cfg=None,
        corruption_prob: float = 0.0,
        corrupt_seed: int = 777,
    ):
        self.base = base
        self.indices = np.asarray(indices, dtype=int)
        self.class_to_idx = class_to_idx
        self.corruption_cfg = corruption_cfg
        self.corruption_prob = float(corruption_prob)
        self.corrupt_seed = int(corrupt_seed)

        if not (0.0 <= self.corruption_prob <= 1.0):
            raise ValueError(
                f"corruption_prob must be in [0, 1], got {self.corruption_prob}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _should_corrupt(self, orig_idx: int) -> bool:
        if self.corruption_prob <= 0.0:
            return False
        if self.corruption_prob >= 1.0:
            return True

        # Deterministic per sample.
        rng = np.random.default_rng(self.corrupt_seed + int(orig_idx) * 1000003)
        return bool(rng.random() < self.corruption_prob)

    def __getitem__(self, i: int):
        orig_idx = int(self.indices[i])
        img, label = self.base[orig_idx]

        if self._should_corrupt(orig_idx):
            img = corrupt_tensor_mnist(
                img,
                seed=self.corrupt_seed + orig_idx,
                corruption_cfg=self.corruption_cfg,
            )

        label = self.class_to_idx[int(label)]
        return img, torch.tensor(label, dtype=torch.long)
    
    
def make_known_training_dataset(
    args,
    base: datasets.MNIST,
    train_indices: np.ndarray,
    class_to_idx: Dict[int, int],
    stage: str,
) -> KnownMNISTDataset:
    """
    Create known-class training dataset with stage-specific corruption.

    stage:
      - "arcface"
      - "scf"
    """
    if stage == "arcface":
        enabled = args.train_corrupt_arcface_enabled
        prob = args.train_corrupt_arcface_probability if enabled else 0.0
        seed = args.seed + args.train_corrupt_arcface_seed_shift

    elif stage == "scf":
        enabled = args.train_corrupt_scf_enabled
        prob = args.train_corrupt_scf_probability if enabled else 0.0
        seed = args.seed + args.train_corrupt_scf_seed_shift

    else:
        raise ValueError(f"Unknown training stage: {stage}")

    print(
        f"[Train dataset] stage={stage}, "
        f"corruption_enabled={enabled}, corruption_prob={prob}, seed={seed}"
    )

    return KnownMNISTDataset(
        base=base,
        indices=train_indices,
        class_to_idx=class_to_idx,
        corruption_cfg=args.corruption,
        corruption_prob=prob,
        corrupt_seed=seed,
    )
    
class IndexedMNISTView(Dataset):
    """
    View of MNIST by original indices. Optionally corrupts selected original indices.
    """

    def __init__(
        self,
        base: datasets.MNIST,
        indices: np.ndarray,
        corrupt_indices: Optional[Iterable[int]] = None,
        corrupt_seed: int = 777,
        corruption_cfg=None,
    ):
        self.base = base
        self.indices = np.asarray(indices, dtype=int)

        if corrupt_indices is None:
            self.corrupt_set = set()
        else:
            self.corrupt_set = set(map(int, np.asarray(list(corrupt_indices)).reshape(-1)))

        self.corrupt_seed = int(corrupt_seed)
        self.corruption_cfg = corruption_cfg

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        orig_idx = int(self.indices[i])
        img, label = self.base[orig_idx]

        if orig_idx in self.corrupt_set:
            img = corrupt_tensor_mnist(
                img,
                seed=self.corrupt_seed + orig_idx,
                corruption_cfg=self.corruption_cfg,
            )

        return img, int(label)

@dataclass
class ProtocolIndices:
    gallery: np.ndarray
    probe_known: np.ndarray
    probe_unknown: np.ndarray
    corrupt_probe: np.ndarray

    @property
    def probe(self) -> np.ndarray:
        return np.concatenate([self.probe_known, self.probe_unknown], axis=0)


def build_train_val_indices(
    train_ds: datasets.MNIST,
    known_classes: List[int],
    unknown_classes: List[int],
    train_per_known_class: int,
    val_gallery_per_class: int,
    val_probe_known_per_class: int,
    val_probe_unknown_per_class: int,
    corrupt_known_frac: float,
    corrupt_unknown_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, ProtocolIndices]:
    targets = get_targets(train_ds)
    train_indices: List[int] = []
    val_gallery: List[int] = []
    val_probe_known: List[int] = []
    val_probe_unknown: List[int] = []

    for c in known_classes:
        pool = np.where(targets == c)[0]
        pool = rng.permutation(pool)

        need = train_per_known_class + val_gallery_per_class + val_probe_known_per_class
        if len(pool) < need:
            raise ValueError(
                f"Not enough MNIST train samples for class {c}: need {need}"
            )

        train_indices.extend(pool[:train_per_known_class])
        s = train_per_known_class
        val_gallery.extend(pool[s : s + val_gallery_per_class])
        s += val_gallery_per_class
        val_probe_known.extend(pool[s : s + val_probe_known_per_class])

    for c in unknown_classes:
        pool = np.where(targets == c)[0]
        pool = rng.permutation(pool)
        if len(pool) < val_probe_unknown_per_class:
            raise ValueError(f"Not enough MNIST train unknown samples for class {c}")
        val_probe_unknown.extend(pool[:val_probe_unknown_per_class])

    val_probe_known = np.asarray(val_probe_known, dtype=int)
    val_probe_unknown = np.asarray(val_probe_unknown, dtype=int)

    corrupt_known_n = int(round(corrupt_known_frac * len(val_probe_known)))
    corrupt_unknown_n = int(round(corrupt_unknown_frac * len(val_probe_unknown)))

    corrupt = []
    if corrupt_known_n > 0:
        corrupt.extend(rng.choice(val_probe_known, size=corrupt_known_n, replace=False))
    if corrupt_unknown_n > 0:
        corrupt.extend(
            rng.choice(val_probe_unknown, size=corrupt_unknown_n, replace=False)
        )

    protocol = ProtocolIndices(
        gallery=np.asarray(val_gallery, dtype=int),
        probe_known=val_probe_known,
        probe_unknown=val_probe_unknown,
        corrupt_probe=np.asarray(corrupt, dtype=int),
    )
    return np.asarray(train_indices, dtype=int), protocol


def build_test_indices(
    test_ds: datasets.MNIST,
    known_classes: List[int],
    unknown_classes: List[int],
    gallery_per_class: int,
    probe_known_per_class: int,
    probe_unknown_per_class: int,
    corrupt_known_frac: float,
    corrupt_unknown_frac: float,
    rng: np.random.Generator,
) -> ProtocolIndices:
    targets = get_targets(test_ds)

    gallery: List[int] = []
    probe_known: List[int] = []
    probe_unknown: List[int] = []

    for c in known_classes:
        pool = np.where(targets == c)[0]
        pool = rng.permutation(pool)

        need = gallery_per_class + probe_known_per_class
        if len(pool) < need:
            raise ValueError(
                f"Not enough MNIST test samples for class {c}: need {need}"
            )

        gallery.extend(pool[:gallery_per_class])
        probe_known.extend(
            pool[gallery_per_class : gallery_per_class + probe_known_per_class]
        )

    for c in unknown_classes:
        pool = np.where(targets == c)[0]
        pool = rng.permutation(pool)
        if len(pool) < probe_unknown_per_class:
            raise ValueError(f"Not enough MNIST test unknown samples for class {c}")
        probe_unknown.extend(pool[:probe_unknown_per_class])

    probe_known = np.asarray(probe_known, dtype=int)
    probe_unknown = np.asarray(probe_unknown, dtype=int)

    corrupt_known_n = int(round(corrupt_known_frac * len(probe_known)))
    corrupt_unknown_n = int(round(corrupt_unknown_frac * len(probe_unknown)))

    corrupt = []
    if corrupt_known_n > 0:
        corrupt.extend(rng.choice(probe_known, size=corrupt_known_n, replace=False))
    if corrupt_unknown_n > 0:
        corrupt.extend(rng.choice(probe_unknown, size=corrupt_unknown_n, replace=False))

    return ProtocolIndices(
        gallery=np.asarray(gallery, dtype=int),
        probe_known=probe_known,
        probe_unknown=probe_unknown,
        corrupt_probe=np.asarray(corrupt, dtype=int),
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_arcface(
    model: TinyArcFaceMNIST,
    train_ds: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    arcface_s: float = 16.0,
    arcface_m: float = 0.30,
    weight_decay: float = 1e-4,
) -> None:
    model.to(device)
    model.train()

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )

    criterion = ArcFaceLoss(s=arcface_s, m=arcface_m)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        losses = []
        accs = []

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item()

            losses.append(loss.item())
            accs.append(acc)

        print(
            f"[ArcFace] epoch {epoch:03d}/{epochs:03d} "
            f"loss={np.mean(losses):.4f} acc={np.mean(accs):.4f}"
        )


def train_scf(
    scf_model: TinySCFMNIST,
    arc_model: TinyArcFaceMNIST,
    train_ds: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    weight_decay: float = 1e-4,
    kappa_regularizer: float = 1e-4,
) -> None:
    scf_model.to(device)
    arc_model.to(device)
    arc_model.eval()

    for p in arc_model.parameters():
        p.requires_grad_(False)

    generator = torch.Generator()
    generator.manual_seed(seed + 100)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )

    criterion = SCFVMFLoss(embedding_dim=arc_model.embedding_dim)
    optimizer = torch.optim.AdamW(
    scf_model.head.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)

    class_centers = arc_model.class_centers().to(device)

    for epoch in range(1, epochs + 1):
        losses = []
        kappas = []
        cosines = []

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            emb, log_kappa = scf_model(imgs)
            kappa = torch.exp(log_kappa)
            wc = class_centers[labels]

            loss = criterion(emb, kappa, wc)
            # small regularizer against always saturating at max kappa
            loss = loss + kappa_regularizer * kappa.mean()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                cos = torch.sum(emb * wc, dim=1).mean().item()
                losses.append(loss.item())
                kappas.append(kappa.mean().item())
                cosines.append(cos)

        print(
            f"[SCF]     epoch {epoch:03d}/{epochs:03d} "
            f"loss={np.mean(losses):.4f} kappa={np.mean(kappas):.2f} "
            f"cos={np.mean(cosines):.3f}"
        )


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------


@torch.no_grad()
def extract_arc_embeddings(
    model: TinyArcFaceMNIST,
    base_ds: datasets.MNIST,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    corrupt_indices: Optional[Iterable[int]] = None,
    corrupt_seed: int = 777,
    corruption_cfg=None,
) -> Tuple[np.ndarray, np.ndarray]:
    view = IndexedMNISTView(
        base_ds,
        indices=indices,
        corrupt_indices=corrupt_indices,
        corrupt_seed=corrupt_seed,
        corruption_cfg=corruption_cfg,
    )
    loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=0)

    model.to(device)
    model.eval()

    embs = []
    labels = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        emb, _ = model(imgs)
        embs.append(emb.detach().cpu().numpy())
        labels.append(np.asarray(y, dtype=int))

    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def extract_scf_embeddings(
    model: TinySCFMNIST,
    base_ds: datasets.MNIST,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    corrupt_indices: Optional[Iterable[int]] = None,
    corrupt_seed: int = 777,
    corruption_cfg=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = IndexedMNISTView(
        base_ds,
        indices=indices,
        corrupt_indices=corrupt_indices,
        corrupt_seed=corrupt_seed,
        corruption_cfg=corruption_cfg,
    )
    loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=0)

    model.to(device)
    model.eval()

    embs = []
    kappas = []
    labels = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        emb, log_kappa = model(imgs)
        kappa = torch.exp(log_kappa)
        embs.append(emb.detach().cpu().numpy())
        kappas.append(kappa.detach().cpu().numpy().reshape(-1))
        labels.append(np.asarray(y, dtype=int))

    return (
        np.concatenate(embs, axis=0),
        np.concatenate(kappas, axis=0),
        np.concatenate(labels, axis=0),
    )


def normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def build_gallery_prototypes(
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    known_classes: List[int],
) -> np.ndarray:
    protos = []
    for c in known_classes:
        mask = gallery_labels == c
        if not np.any(mask):
            raise ValueError(f"No gallery samples for known class {c}")
        proto = gallery_embs[mask].mean(axis=0, keepdims=True)
        proto = normalize_np(proto)[0]
        protos.append(proto)
    return np.asarray(protos, dtype=np.float64)


# ---------------------------------------------------------------------
# Open-set prediction and HolUE/GalUE scores
# ---------------------------------------------------------------------


def stable_softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# ---------------------------------------------------------------------
# Paper-style mixed discrete-continuous OOG prior for HolUE
# ---------------------------------------------------------------------


def sphere_area_np(dim: int) -> float:
    """
    Surface area of S^{dim-1} embedded in R^dim.
    For dim=2: circumference of unit circle = 2*pi.
    For dim=3: area of unit sphere = 4*pi.
    """
    return float(2.0 * math.pi ** (dim / 2.0) / gamma(dim / 2.0))


def log_vmf_normalizer_np(kappa, dim: int):
    """
    log C_d(kappa) for vMF on S^{dim-1}.

    Supports dim=2 and dim=3, matching this toy script.
    """
    scalar_input = np.isscalar(kappa)
    k = np.atleast_1d(np.asarray(kappa, dtype=np.float64))

    if dim == 2:
        # C_2(kappa) = 1 / (2*pi*I_0(kappa)).
        # Use exponentially scaled Bessel for stability:
        # I_0(k) = ive(0,k) * exp(k), k >= 0.
        log_i0 = np.log(np.maximum(ive(0, k), 1e-300)) + k
        out = -math.log(2.0 * math.pi) - log_i0

    elif dim == 3:
        # C_3(kappa) = kappa / (4*pi*sinh(kappa)).
        out = np.empty_like(k)
        small = k < 1e-8

        out[small] = -math.log(4.0 * math.pi)

        ks = k[~small]
        if len(ks) > 0:
            log_sinh = (
                ks
                - math.log(2.0)
                + np.log1p(-np.exp(-2.0 * ks))
            )
            out[~small] = np.log(ks) - math.log(4.0 * math.pi) - log_sinh

    else:
        raise ValueError(f"Only dim=2 and dim=3 are supported, got dim={dim}")

    if scalar_input:
        return float(out[0])
    return out.reshape(np.shape(kappa))


def beta_from_tau_mixed_prior(
    dim: int,
    tau: float,
    gallery_kappa: float,
    K: int,
) -> float:
    """
    Derive beta so that the continuous-OOG Bayesian accept/reject boundary
    matches the cosine threshold tau.

    Boundary condition:

        ((1-beta)/K) * C_d(kappa) * exp(kappa * tau)
        =
        beta / S_{d-1}

    Therefore:

        beta / (1-beta)
        =
        S_{d-1} * C_d(kappa) * exp(kappa*tau) / K.
    """
    S = sphere_area_np(dim)
    log_C = log_vmf_normalizer_np(gallery_kappa, dim=dim)

    log_A = (
        math.log(S)
        + log_C
        + float(gallery_kappa) * float(tau)
        - math.log(K)
    )

    # beta = sigmoid(log_A)
    if log_A >= 0:
        beta = 1.0 / (1.0 + math.exp(-log_A))
    else:
        A = math.exp(log_A)
        beta = A / (1.0 + A)

    return float(np.clip(beta, 1e-8, 1.0 - 1e-8))


def tau_from_beta_mixed_prior(
    dim: int,
    beta: float,
    gallery_kappa: float,
    K: int,
) -> float:
    """
    Inverse of beta_from_tau_mixed_prior.

    Useful for the oracle circle experiment, where beta is fixed first.
    """
    beta = float(np.clip(beta, 1e-8, 1.0 - 1e-8))
    S = sphere_area_np(dim)
    log_C = log_vmf_normalizer_np(gallery_kappa, dim=dim)

    tau = (
        math.log(beta)
        - math.log1p(-beta)
        + math.log(K)
        - math.log(S)
        - log_C
    ) / float(gallery_kappa)

    return float(tau)


def posterior_from_z_mixed_prior(
    z: np.ndarray,
    gallery_mu: np.ndarray,
    gallery_kappa: float,
    beta: float,
) -> np.ndarray:
    """
    Paper-style GalUE posterior for deterministic embeddings z.

    Known classes:
        c = 1,...,K

    Unknown identities:
        psi in S^{d-1}, with uniform continuous prior density beta / S.

    This returns the collapsed action posterior:

        [P(c=1|z), ..., P(c=K|z), P(OOG|z)]

    but the OOG term is induced by a continuous identity prior, not by
    one discrete unknown class.
    """
    z = normalize_np(np.asarray(z, dtype=np.float64))
    gallery_mu = normalize_np(np.asarray(gallery_mu, dtype=np.float64))

    dim = z.shape[1]
    K = gallery_mu.shape[0]
    S = sphere_area_np(dim)

    beta = float(np.clip(beta, 1e-8, 1.0 - 1e-8))

    log_prior_known = math.log1p(-beta) - math.log(K)
    log_prior_oog_density = math.log(beta) - math.log(S)
    log_Cg = log_vmf_normalizer_np(gallery_kappa, dim=dim)

    known_log = (
        log_prior_known
        + log_Cg
        + float(gallery_kappa) * (z @ gallery_mu.T)
    )  # (N,K)

    oog_log = np.full((z.shape[0], 1), log_prior_oog_density, dtype=np.float64)

    logits = np.concatenate([known_log, oog_log], axis=1)
    log_norm = logsumexp(logits, axis=1, keepdims=True)

    return np.exp(logits - log_norm)


def circle_quadrature_grid_with_log_dS(n_grid: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform quadrature grid on S^1.

    Returns:
        grid_z: (G,2)
        log_dS: (G,), log arc-length weight
    """
    S = 2.0 * math.pi
    angles = np.linspace(0.0, S, int(n_grid), endpoint=False)
    grid_z = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)

    dS = S / int(n_grid)
    log_dS = np.full(int(n_grid), math.log(dS), dtype=np.float64)

    return grid_z, log_dS


def sphere_quadrature_grid_with_log_dS(
    n_theta: int,
    n_phi: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Midpoint quadrature grid on S^2 with true surface weights.

    dS = sin(phi) dphi dtheta.
    """
    n_theta = int(n_theta)
    n_phi = int(n_phi)

    dtheta = 2.0 * math.pi / n_theta
    dphi = math.pi / n_phi

    theta = (np.arange(n_theta) + 0.5) * dtheta
    phi = (np.arange(n_phi) + 0.5) * dphi

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)

    grid_z = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)

    dS = np.sin(phi_grid).reshape(-1) * dtheta * dphi
    log_dS = np.log(np.maximum(dS, 1e-300)).astype(np.float64)

    return grid_z, log_dS


def mixed_holue_posterior_grid_quadrature(
    mu: np.ndarray,
    kappa: np.ndarray,
    gallery_mu: np.ndarray,
    gallery_kappa: float,
    beta: float,
    grid_z: np.ndarray,
    log_dS: np.ndarray,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Correct paper-style HolUE computation with continuous OOG identities.

    It computes:

        p(C|x) = ∫ p(C|z) p(z|x) dz

    where C is mixed:

        C in {1,...,K} union S^{d-1}.

    Returned action_post collapses the continuous OOG posterior density only
    for the OSR decision:

        action_post[:, :K] = known identity posterior masses
        action_post[:, K]  = total OOG posterior mass

    But KL is computed against the full mixed prior:

        KL = sum_known P_i log(P_i / prior_i)
             + ∫ rho(psi|x) log(rho(psi|x) / prior_oog_density) dpsi

    This is the key difference from the old K+1 discrete entropy/KL.
    """
    mu = normalize_np(np.asarray(mu, dtype=np.float64))
    kappa = np.asarray(kappa, dtype=np.float64).reshape(-1)
    gallery_mu = normalize_np(np.asarray(gallery_mu, dtype=np.float64))
    grid_z = normalize_np(np.asarray(grid_z, dtype=np.float64))
    log_dS = np.asarray(log_dS, dtype=np.float64).reshape(-1)

    dim = mu.shape[1]
    K = gallery_mu.shape[0]
    S = sphere_area_np(dim)

    beta = float(np.clip(beta, 1e-8, 1.0 - 1e-8))

    log_prior_known = math.log1p(-beta) - math.log(K)
    log_prior_oog_density = math.log(beta) - math.log(S)
    log_Cg = log_vmf_normalizer_np(gallery_kappa, dim=dim)

    # log p(c=i) p(z|c=i), for all grid points and known classes.
    cos_grid_gallery = grid_z @ gallery_mu.T  # (G,K)
    log_known_joint = (
        log_prior_known
        + log_Cg
        + float(gallery_kappa) * cos_grid_gallery
    )  # (G,K)

    # OOG contribution to marginal p(z) is beta / S.
    log_oog_joint = np.full((grid_z.shape[0], 1), log_prior_oog_density)

    # log p(z)
    log_p_z = logsumexp(
        np.concatenate([log_known_joint, log_oog_joint], axis=1),
        axis=1,
    )  # (G,)

    action_posts = []
    kl_known_all = []
    kl_oog_all = []
    kl_total_all = []

    for start in range(0, len(mu), batch_size):
        end = min(start + batch_size, len(mu))

        mu_b = mu[start:end]
        kappa_b = kappa[start:end]
        B = len(mu_b)

        # log p(z|x) on the quadrature grid.
        cos_to_mu = mu_b @ grid_z.T  # (B,G)
        log_Cx = log_vmf_normalizer_np(kappa_b, dim=dim).reshape(B, 1)
        log_f_x = log_Cx + kappa_b[:, None] * cos_to_mu  # (B,G)

        # Known posterior masses:
        #
        # P_i(x) = ∫ [p(i)p(z|i)/p(z)] p(z|x) dz.
        log_integrand_known = (
            log_f_x[:, :, None]
            + log_known_joint[None, :, :]
            - log_p_z[None, :, None]
            + log_dS[None, :, None]
        )  # (B,G,K)

        P_known = np.exp(logsumexp(log_integrand_known, axis=1))  # (B,K)

        # Continuous OOG posterior density:
        #
        # rho(psi|x) = (beta/S) * p(psi|x) / p(psi).
        log_rho = (
            log_prior_oog_density
            + log_f_x
            - log_p_z[None, :]
        )  # (B,G), density wrt surface measure

        rho_dS = np.exp(log_rho + log_dS[None, :])  # posterior mass per grid cell
        P_oog = np.sum(rho_dS, axis=1)  # (B,)

        total_mass = np.maximum(P_known.sum(axis=1) + P_oog, 1e-300)
        log_total_mass = np.log(total_mass)

        # Normalize tiny quadrature error away.
        P_known_n = P_known / total_mass[:, None]
        P_oog_n = P_oog / total_mass
        rho_dS_n = rho_dS / total_mass[:, None]

        action_post = np.concatenate([P_known_n, P_oog_n[:, None]], axis=1)

        # Known KL term.
        P_known_safe = np.clip(P_known_n, 1e-300, 1.0)
        kl_known = np.sum(
            P_known_safe * (np.log(P_known_safe) - log_prior_known),
            axis=1,
        )

        # OOG continuous KL term:
        #
        # ∫ rho'(psi|x) log(rho'(psi|x)/(beta/S)) dpsi.
        #
        # Unnormalized rho/q0 = p(psi|x)/p(psi).
        # After normalization by total_mass:
        # log ratio = log_f_x - log_p_z - log_total_mass.
        log_ratio_oog = (
            log_f_x
            - log_p_z[None, :]
            - log_total_mass[:, None]
        )

        kl_oog = np.sum(rho_dS_n * log_ratio_oog, axis=1)

        kl_total = kl_known + kl_oog
        kl_total = np.maximum(kl_total, 0.0)

        action_posts.append(action_post)
        kl_known_all.append(kl_known)
        kl_oog_all.append(kl_oog)
        kl_total_all.append(kl_total)

    return (
        np.concatenate(action_posts, axis=0),
        np.concatenate(kl_known_all, axis=0),
        np.concatenate(kl_oog_all, axis=0),
        np.concatenate(kl_total_all, axis=0),
    )


def mixed_holue_posterior_circle_quadrature(
    mu: np.ndarray,
    kappa: np.ndarray,
    gallery_mu: np.ndarray,
    gallery_kappa: float,
    beta: float,
    n_grid: int = 720,
    batch_size: int = 512,
):
    grid_z, log_dS = circle_quadrature_grid_with_log_dS(n_grid)
    return mixed_holue_posterior_grid_quadrature(
        mu=mu,
        kappa=kappa,
        gallery_mu=gallery_mu,
        gallery_kappa=gallery_kappa,
        beta=beta,
        grid_z=grid_z,
        log_dS=log_dS,
        batch_size=batch_size,
    )


def mixed_holue_posterior_sphere_quadrature(
    mu: np.ndarray,
    kappa: np.ndarray,
    gallery_mu: np.ndarray,
    gallery_kappa: float,
    beta: float,
    n_theta: int = 96,
    n_phi: int = 48,
    batch_size: int = 256,
):
    grid_z, log_dS = sphere_quadrature_grid_with_log_dS(
        n_theta=n_theta,
        n_phi=n_phi,
    )
    return mixed_holue_posterior_grid_quadrature(
        mu=mu,
        kappa=kappa,
        gallery_mu=gallery_mu,
        gallery_kappa=gallery_kappa,
        beta=beta,
        grid_z=grid_z,
        log_dS=log_dS,
        batch_size=batch_size,
    )


def action_risk_from_posterior(stats: "OSRStats") -> np.ndarray:
    """
    Exact collapsed-action posterior risk for the current OSR decision.

    If decision is reject:
        risk = 1 - P(OOG|x)

    If decision is accept as class j:
        risk = 1 - P(class j|x)

    This is useful as an oracle diagnostic. It is not the same as the
    mixed identity-information KL used by HolUE.
    """
    idx = np.arange(len(stats.labels))
    chosen_prob = np.where(
        stats.rejected,
        stats.holue_posterior[:, -1],
        stats.holue_posterior[idx, stats.pred_idx],
    )
    return 1.0 - chosen_prob

def entropy_normalized(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    return h / math.log(p.shape[1])


def threshold_at_fpir(scores_unknown: np.ndarray, target_fpir: float) -> float:
    """
    Rule: accept if max_similarity >= tau.
    """
    scores = np.asarray(scores_unknown, dtype=np.float64).reshape(-1)
    if len(scores) == 0:
        raise ValueError("Need unknown validation scores to set FPIR threshold.")

    target_fpir = float(target_fpir)
    if target_fpir <= 0:
        return float(np.nextafter(scores.max(), np.inf))
    if target_fpir >= 1:
        return float(np.nextafter(scores.min(), -np.inf))

    n_accept = int(np.floor(target_fpir * len(scores)))
    if n_accept <= 0:
        return float(np.nextafter(scores.max(), np.inf))

    idx = len(scores) - n_accept
    idx = int(np.clip(idx, 0, len(scores) - 1))
    return float(np.partition(scores, idx)[idx])


def posterior_from_z(
    z: np.ndarray,
    gallery_mu: np.ndarray,
    tau: float,
    gallery_kappa: float,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Gallery-aware posterior p(c|z) on K known classes + one out-of-gallery class.

    We use a threshold-aligned Bayesian softmax:
      known logits = k_g * <z, m_c>
      OOG logit   = k_g * tau

    Therefore:
      argmax known > OOG  iff max_c <z,m_c> > tau

    This keeps the OSR decision rule exactly aligned with the standard
    cosine-threshold OSR pipeline, while still providing a smooth posterior.
    """
    z = np.asarray(z, dtype=np.float64)
    gallery_mu = np.asarray(gallery_mu, dtype=np.float64)

    known_logits = gallery_kappa * (z @ gallery_mu.T)
    oog_logits = np.full((z.shape[0], 1), gallery_kappa * tau, dtype=np.float64)
    logits = np.concatenate([known_logits, oog_logits], axis=1)
    logits = logits / float(temperature)
    return stable_softmax_np(logits, axis=1)


def holue_posterior_quadrature(
    mu: np.ndarray,
    kappa: np.ndarray,
    gallery_mu: np.ndarray,
    tau: float,
    gallery_kappa: float,
    n_grid: int = 720,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Deterministic computation of:
      p(c|x) = ∫ p(c|z) p(z|x) dz

    Because embeddings are 2D on the unit circle, we integrate by a dense angle grid.
    p(z|x) is vMF on S^1 with mean direction mu and concentration kappa.
    """
    mu = normalize_np(np.asarray(mu, dtype=np.float64))
    kappa = np.asarray(kappa, dtype=np.float64).reshape(-1)
    gallery_mu = normalize_np(np.asarray(gallery_mu, dtype=np.float64))

    angles = np.linspace(0.0, 2.0 * math.pi, n_grid, endpoint=False)
    grid_z = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    post_grid = posterior_from_z(
        grid_z,
        gallery_mu=gallery_mu,
        tau=tau,
        gallery_kappa=gallery_kappa,
    )  # (G, K+1)

    out = []
    for start in range(0, len(mu), batch_size):
        end = min(start + batch_size, len(mu))
        mu_b = mu[start:end]
        kappa_b = kappa[start:end]

        cos_to_mu = mu_b @ grid_z.T  # (B, G)
        log_w = kappa_b[:, None] * cos_to_mu
        w = stable_softmax_np(log_w, axis=1)
        post = w @ post_grid
        out.append(post)

    return np.concatenate(out, axis=0)


def discrete_kl_components(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KL to uniform prior, split into known-class and OOG terms.
    This is used as a simple calibration feature.
    """
    p = np.asarray(p, dtype=np.float64)
    n_classes = p.shape[1]
    prior = 1.0 / n_classes
    p_safe = np.clip(p, 1e-12, 1.0)

    kl_terms = p_safe * (np.log(p_safe) - math.log(prior))
    kl_known = np.sum(kl_terms[:, :-1], axis=1)
    kl_oog = kl_terms[:, -1]
    kl_total = kl_known + kl_oog
    return kl_known, kl_oog, kl_total


@dataclass
class OSRStats:
    labels: np.ndarray
    sim: np.ndarray
    max_sim: np.ndarray
    pred_idx: np.ndarray
    rejected: np.ndarray
    scf_kappa: np.ndarray
    gal_posterior: np.ndarray
    holue_posterior: np.ndarray
    gal_entropy: np.ndarray
    holue_entropy: np.ndarray
    kl_known: np.ndarray
    kl_oog: np.ndarray
    kl_total: np.ndarray


def compute_osr_stats(
    probe_mu: np.ndarray,
    probe_kappa: np.ndarray,
    probe_labels: np.ndarray,
    gallery_mu: np.ndarray,
    tau: float,
    gallery_kappa: float,
    n_grid: int,
    beta: Optional[float] = None,
    continuous_oog: bool = True,
) -> OSRStats:
    probe_mu = normalize_np(probe_mu.astype(np.float64))
    gallery_mu = normalize_np(gallery_mu.astype(np.float64))
    probe_kappa = np.asarray(probe_kappa, dtype=np.float64).reshape(-1)

    dim = probe_mu.shape[1]
    K = gallery_mu.shape[0]

    if dim != 2:
        raise ValueError(f"compute_osr_stats is the 2D version, got dim={dim}")

    if beta is None:
        beta = beta_from_tau_mixed_prior(
            dim=dim,
            tau=tau,
            gallery_kappa=gallery_kappa,
            K=K,
        )

    sim = probe_mu @ gallery_mu.T
    max_sim = np.max(sim, axis=1)
    pred_idx = np.argmax(sim, axis=1)
    rejected = max_sim < tau

    if continuous_oog:
        # Paper-style deterministic GalUE posterior with continuous OOG prior.
        gal_post = posterior_from_z_mixed_prior(
            probe_mu,
            gallery_mu=gallery_mu,
            gallery_kappa=gallery_kappa,
            beta=beta,
        )

        # Paper-style HolUE mixed posterior and mixed KL.
        hol_post, kl_known, kl_oog, kl_total = mixed_holue_posterior_circle_quadrature(
            mu=probe_mu,
            kappa=probe_kappa,
            gallery_mu=gallery_mu,
            gallery_kappa=gallery_kappa,
            beta=beta,
            n_grid=n_grid,
        )

    else:
        # Old collapsed K+1-discrete OOG version.
        gal_post = posterior_from_z(
            probe_mu,
            gallery_mu=gallery_mu,
            tau=tau,
            gallery_kappa=gallery_kappa,
        )

        hol_post = holue_posterior_quadrature(
            mu=probe_mu,
            kappa=probe_kappa,
            gallery_mu=gallery_mu,
            tau=tau,
            gallery_kappa=gallery_kappa,
            n_grid=n_grid,
        )

        kl_known, kl_oog, kl_total = discrete_kl_components(hol_post)

    gal_entropy = entropy_normalized(gal_post)
    hol_entropy = entropy_normalized(hol_post)

    return OSRStats(
        labels=np.asarray(probe_labels, dtype=int),
        sim=sim,
        max_sim=max_sim,
        pred_idx=pred_idx,
        rejected=rejected,
        scf_kappa=probe_kappa,
        gal_posterior=gal_post,
        holue_posterior=hol_post,
        gal_entropy=gal_entropy,
        holue_entropy=hol_entropy,
        kl_known=kl_known,
        kl_oog=kl_oog,
        kl_total=kl_total,
    )

# ---------------------------------------------------------------------
# Metrics and rejection curves
# ---------------------------------------------------------------------


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or not np.isfinite(b):
        return default
    return float(a) / float(b)


def osr_error_mask(
    pred_idx: np.ndarray,
    rejected: np.ndarray,
    labels: np.ndarray,
    known_classes: List[int],
) -> np.ndarray:
    known_classes_arr = np.asarray(known_classes, dtype=int)
    pred_digit = known_classes_arr[pred_idx]

    labels = np.asarray(labels, dtype=int)
    seen = np.isin(labels, known_classes_arr)

    correct = np.zeros(len(labels), dtype=bool)
    correct[seen] = (~rejected[seen]) & (pred_digit[seen] == labels[seen])
    correct[~seen] = rejected[~seen]

    return ~correct


def compute_osr_metrics(
    pred_idx: np.ndarray,
    rejected: np.ndarray,
    labels: np.ndarray,
    known_classes: List[int],
) -> Dict[str, float]:
    known_classes_arr = np.asarray(known_classes, dtype=int)
    pred_digit = known_classes_arr[pred_idx]

    labels = np.asarray(labels, dtype=int)
    seen = np.isin(labels, known_classes_arr)

    correct_seen_accept = seen & (~rejected) & (pred_digit == labels)
    false_accept = (~seen) & (~rejected)

    tp = int(np.sum(correct_seen_accept))
    fp = int(np.sum(false_accept))
    fn = int(np.sum(seen)) - tp

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    fpir = safe_div(fp, np.sum(~seen))
    fnir = 1.0 - safe_div(tp, np.sum(seen))

    misid = seen & (~rejected) & (pred_digit != labels)
    false_reject = seen & rejected

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        "fpir": fpir,
        "fnir": fnir,
        "error_rate": float(
            np.mean(osr_error_mask(pred_idx, rejected, labels, known_classes))
        ),
        "false_accept_count": int(np.sum(false_accept)),
        "false_reject_count": int(np.sum(false_reject)),
        "misidentification_count": int(np.sum(misid)),
    }


def rejection_curve(
    uncertainty: np.ndarray,
    pred_idx: np.ndarray,
    rejected: np.ndarray,
    labels: np.ndarray,
    known_classes: List[int],
    fractions: np.ndarray,
) -> pd.DataFrame:
    """
    Higher uncertainty means rejected earlier.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    order = np.argsort(-uncertainty)
    n = len(uncertainty)

    rows = []
    for frac in fractions:
        n_drop = int(round(float(frac) * n))
        keep = np.ones(n, dtype=bool)
        if n_drop > 0:
            keep[order[:n_drop]] = False

        m = compute_osr_metrics(
            pred_idx=pred_idx[keep],
            rejected=rejected[keep],
            labels=labels[keep],
            known_classes=known_classes,
        )
        rows.append({"fraction": float(frac), **m})

    return pd.DataFrame(rows)


def compute_prr(
    curve: pd.DataFrame,
    random_curve: pd.DataFrame,
    oracle_curve: pd.DataFrame,
    metric: str = "f1",
) -> float:
    x = curve["fraction"].values
    a = np.trapezoid(curve[metric].values, x)
    r = np.trapezoid(random_curve[metric].values, x)
    o = np.trapezoid(oracle_curve[metric].values, x)

    denom = o - r
    if abs(denom) < 1e-12:
        return float("nan")
    return float((a - r) / denom)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_rejection_curves(
    curves: Dict[str, pd.DataFrame],
    out_dir: Path,
    metric: str = "f1",
    plot_cfg=None,
    method_order: Optional[List[str]] = None,
) -> None:
    """
    Config-driven rejection curves with PRR in legend.

    PRR:
        (AUC_method - AUC_random) / (AUC_oracle - AUC_random)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Random" not in curves or "Oracle" not in curves:
        raise ValueError("Curves dictionary must contain 'Random' and 'Oracle' for PRR.")

    figsize = tuple(cfg_get(plot_cfg, "figsize", [8.5, 5.3]))
    dpi = int(cfg_get(plot_cfg, "dpi", 300))
    title = cfg_get(plot_cfg, "title", "Uncertainty-based filtering")
    xlabel = cfg_get(plot_cfg, "xlabel", "Filtered-out probe fraction")
    ylabel = cfg_get(plot_cfg, "ylabel", metric.upper())
    grid = bool(cfg_get(plot_cfg, "grid", True))
    grid_alpha = float(cfg_get(plot_cfg, "grid_alpha", 0.5))
    legend_fontsize = float(cfg_get(plot_cfg, "legend_fontsize", 8.5))
    show_prr = bool(cfg_get(plot_cfg, "show_prr_in_legend", True))

    lw_holue = float(cfg_get(plot_cfg, "line_width_holue", 2.8))
    lw_default = float(cfg_get(plot_cfg, "line_width_default", 1.7))
    alpha_default = float(cfg_get(plot_cfg, "line_alpha_default", 1.0))
    alpha_random_oracle = float(cfg_get(plot_cfg, "line_alpha_random_oracle", 0.75))

    save_png = cfg_get(plot_cfg, "save_png", "rejection_curves.png")
    save_pdf = cfg_get(plot_cfg, "save_pdf", "rejection_curves.pdf")
    save_prr_csv = cfg_get(plot_cfg, "save_prr_csv", f"rejection_curve_prr_{metric}.csv")

    random_curve = curves["Random"]
    oracle_curve = curves["Oracle"]

    prr_values = {}
    for name, df in curves.items():
        prr_values[name] = compute_prr(
            curve=df,
            random_curve=random_curve,
            oracle_curve=oracle_curve,
            metric=metric,
        )

    if method_order is None:
        method_order = [
            "HolUE no calibration",
            "HolUE calibrated",
            "HolUE raw KL",
            "GalUE",
            "SCF",
            "AccScr",
            "MaxSim",
            "Random",
            "Oracle",
        ]

    names_to_plot = [n for n in method_order if n in curves]
    names_to_plot += [n for n in curves.keys() if n not in names_to_plot]

    plt.figure(figsize=figsize)

    for name in names_to_plot:
        df = curves[name]

        lw = lw_holue if "HolUE" in name else lw_default
        alpha = alpha_random_oracle if name in {"Random", "Oracle"} else alpha_default

        if show_prr:
            prr = prr_values[name]
            label = f"{name} (PRR={prr:.2f})" if np.isfinite(prr) else f"{name} (PRR=nan)"
        else:
            label = name

        plt.plot(
            df["fraction"],
            df[metric],
            label=label,
            linewidth=lw,
            alpha=alpha,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if grid:
        plt.grid(True, linestyle="--", alpha=grid_alpha)

    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    if save_png:
        plt.savefig(out_dir / save_png, dpi=dpi)
    if save_pdf:
        plt.savefig(out_dir / save_pdf, dpi=dpi, bbox_inches="tight")

    plt.close()

    prr_df = pd.DataFrame(
        [{"method": name, f"PRR_{metric}": prr_values[name]} for name in names_to_plot]
    )
    if save_prr_csv:
        prr_df.to_csv(out_dir / save_prr_csv, index=False)
        
        
def plot_teaser_circle(
    stats: OSRStats,
    gallery_mu: np.ndarray,
    known_classes: List[int],
    tau: float,
    gallery_kappa: float,
    uncertainty: np.ndarray,
    out_dir: Path,
    seed: int,
    max_points: int = 900,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    labels = stats.labels
    err = osr_error_mask(stats.pred_idx, stats.rejected, labels, known_classes)

    # For positions we use probe mean directions.
    # Recover mean direction approximately from similarities is impossible, so caller stores
    # them by putting sim only. We infer positions for plotting from nearest class? Instead,
    # use a projection from sim to a display circle by taking actual direction stored in
    # stats via extra attribute if present.
    if not hasattr(stats, "probe_mu_for_plot"):
        raise RuntimeError(
            "stats.probe_mu_for_plot must be attached before teaser plotting."
        )
    probe_mu = getattr(stats, "probe_mu_for_plot")

    probe_mu = normalize_np(probe_mu)
    gallery_mu = normalize_np(gallery_mu)

    n = len(labels)
    if n > max_points:
        # Always keep all errors if possible, fill the rest with random correct points.
        error_idx = np.where(err)[0]
        correct_idx = np.where(~err)[0]
        remaining = max(0, max_points - len(error_idx))
        if remaining > 0 and len(correct_idx) > remaining:
            correct_idx = rng.choice(correct_idx, size=remaining, replace=False)
        selected = np.concatenate([error_idx, correct_idx])
    else:
        selected = np.arange(n)

    unc = np.asarray(uncertainty, dtype=np.float64)
    unc_norm = (unc - np.nanmin(unc)) / (np.nanmax(unc) - np.nanmin(unc) + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Background GalUE uncertainty ring.
    angles = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    z_grid = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    post_grid = posterior_from_z(
        z_grid,
        gallery_mu=gallery_mu,
        tau=tau,
        gallery_kappa=gallery_kappa,
    )
    ring_unc = entropy_normalized(post_grid)

    sc_bg = ax.scatter(
        1.12 * z_grid[:, 0],
        1.12 * z_grid[:, 1],
        c=ring_unc,
        cmap="Blues",
        s=18,
        alpha=0.75,
        linewidths=0,
        label="gallery ambiguity ring",
    )

    # Acceptance arcs around gallery centers.
    width = math.acos(float(np.clip(tau, -1.0, 1.0)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(known_classes)))
    for i, (c, proto) in enumerate(zip(known_classes, gallery_mu)):
        a = math.atan2(proto[1], proto[0])
        arc = np.linspace(a - width, a + width, 120)
        ax.plot(
            1.24 * np.cos(arc),
            1.24 * np.sin(arc),
            color=colors[i],
            linewidth=4,
            alpha=0.65,
        )

    # Probe points: radial jitter by uncertainty for readability.
    selected = np.asarray(selected, dtype=int)
    correct_sel = selected[~err[selected]]
    error_sel = selected[err[selected]]

    def point_xy(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = 0.88 + 0.16 * unc_norm[idx]
        return r * probe_mu[idx, 0], r * probe_mu[idx, 1]

    if len(correct_sel) > 0:
        x, y = point_xy(correct_sel)
        ax.scatter(
            x,
            y,
            c=unc[correct_sel],
            cmap="magma",
            s=22,
            alpha=0.45,
            linewidths=0,
            marker="o",
            label="correct probes",
        )

    if len(error_sel) > 0:
        x, y = point_xy(error_sel)
        sc = ax.scatter(
            x,
            y,
            c=unc[error_sel],
            cmap="magma",
            s=60,
            alpha=0.95,
            linewidths=1.6,
            marker="x",
            label="OSR errors",
        )
    else:
        sc = ax.scatter([], [], c=[], cmap="magma")

    # Gallery prototypes.
    for i, (c, proto) in enumerate(zip(known_classes, gallery_mu)):
        ax.scatter(
            proto[0],
            proto[1],
            s=260,
            marker="*",
            color=colors[i],
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.text(
            1.36 * proto[0],
            1.36 * proto[1],
            f"digit {c}",
            color=colors[i],
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Unit circle.
    circle = plt.Circle(
        (0, 0), 1.0, color="black", fill=False, linewidth=1.0, alpha=0.6
    )
    ax.add_artist(circle)

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "HolUE on MNIST OSR: high uncertainty concentrates near real errors",
        fontsize=14,
    )

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("HolUE no-calibration uncertainty", rotation=90)

    ax.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()

    plt.savefig(out_dir / "teaser_holue_mnist_circle.png", dpi=300)
    plt.savefig(out_dir / "teaser_holue_mnist_circle.pdf", dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------
# Corruption sanity-check visualizations
# ---------------------------------------------------------------------


def mnist_tensor_to_numpy_img(x: torch.Tensor) -> np.ndarray:
    """
    Convert normalized MNIST tensor [1,28,28] from [-1,1] to display image [0,1].
    """
    x = x.detach().cpu().clone()
    x = x * 0.5 + 0.5
    x = x.clamp(0.0, 1.0)
    return x.squeeze(0).numpy()


def save_original_vs_corrupted_grid(
    base: datasets.MNIST,
    indices: np.ndarray,
    out_path: Path,
    corruption_cfg,
    seed: int,
    num_examples: int = 16,
    ncols: int = 8,
    title: str = "Original vs corrupted",
    force_corruption: bool = True,
    corruption_prob: float = 1.0,
) -> None:
    """
    Save a compact grid:

      row 0: original images
      row 1: corrupted images
      row 2: original images
      row 3: corrupted images
      ...

    If force_corruption=True, every displayed sample is corrupted so that the
    corruption modes can be inspected visually.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.asarray(indices, dtype=int).reshape(-1)
    if len(indices) == 0:
        print(f"[Corruption vis] No indices for {out_path}; skipping.")
        return

    rng = np.random.default_rng(seed)

    n = min(int(num_examples), len(indices))
    chosen = rng.choice(indices, size=n, replace=False)

    ncols = min(int(ncols), n)
    nblocks = int(math.ceil(n / ncols))
    nrows = 2 * nblocks

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(1.55 * ncols, 1.75 * nrows),
        squeeze=False,
    )

    for ax in axes.reshape(-1):
        ax.axis("off")

    for j, idx in enumerate(chosen):
        block = j // ncols
        col = j % ncols

        img, label = base[int(idx)]

        if force_corruption:
            do_corrupt = True
        else:
            do_corrupt = bool(rng.random() < corruption_prob)

        if do_corrupt:
            img_corr = corrupt_tensor_mnist(
                img,
                seed=seed + int(idx),
                corruption_cfg=corruption_cfg,
            )
        else:
            img_corr = img.clone()

        ax_orig = axes[2 * block, col]
        ax_corr = axes[2 * block + 1, col]

        ax_orig.imshow(mnist_tensor_to_numpy_img(img), cmap="gray", vmin=0, vmax=1)
        ax_corr.imshow(mnist_tensor_to_numpy_img(img_corr), cmap="gray", vmin=0, vmax=1)

        ax_orig.set_title(f"orig {int(label)}", fontsize=8)
        ax_corr.set_title("corrupt", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Corruption vis] Saved: {out_path}")


def save_training_batch_grid(
    dataset: Dataset,
    out_path: Path,
    seed: int,
    batch_size: int = 32,
    ncols: int = 8,
    title: str = "Corrupted training batch",
    dpi: int = 300,
) -> None:
    """
    Save one actual batch from a training dataset. This shows what the model
    really sees during ArcFace/SCF training.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )

    imgs, labels = next(iter(loader))
    n = imgs.shape[0]

    ncols = min(int(ncols), n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(1.55 * ncols, 1.75 * nrows),
        squeeze=False,
    )

    for ax in axes.reshape(-1):
        ax.axis("off")

    for i in range(n):
        row = i // ncols
        col = i % ncols

        axes[row, col].imshow(
            mnist_tensor_to_numpy_img(imgs[i]),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, col].set_title(f"y={int(labels[i])}", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Corruption vis] Saved: {out_path}")


def save_corruption_visualizations(
    args,
    mnist_train: datasets.MNIST,
    mnist_test: datasets.MNIST,
    train_indices: np.ndarray,
    val_protocol: ProtocolIndices,
    test_protocol: ProtocolIndices,
    train_ds_arcface: Dataset,
    train_ds_scf: Dataset,
    figure_dir: Path,
) -> None:
    """
    Create visual sanity checks for corrupted images and actual train batches.
    """
    vis_cfg = args.plots.corruption_visualization

    if not bool(cfg_get(vis_cfg, "enabled", True)):
        return

    figure_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(cfg_get(vis_cfg, "dpi", 300))
    num_pair_examples = int(cfg_get(vis_cfg, "num_pair_examples", 16))
    pair_grid_cols = int(cfg_get(vis_cfg, "pair_grid_cols", 8))
    batch_size = int(cfg_get(vis_cfg, "batch_size", 32))
    batch_grid_cols = int(cfg_get(vis_cfg, "batch_grid_cols", 8))
    force_pair = bool(cfg_get(vis_cfg, "force_pair_corruption", True))
    title_prefix = str(cfg_get(vis_cfg, "title_prefix", "MNIST corruption sanity check"))

    # 1. Forced corruption pairs from training indices.
    save_original_vs_corrupted_grid(
        base=mnist_train,
        indices=train_indices,
        out_path=figure_dir / cfg_get(vis_cfg, "save_train_pairs", "corruption_pairs_train_forced.png"),
        corruption_cfg=args.corruption,
        seed=args.seed + 111,
        num_examples=num_pair_examples,
        ncols=pair_grid_cols,
        title=f"{title_prefix}: train original vs corrupted",
        force_corruption=force_pair,
        corruption_prob=1.0,
    )

    # 2. Forced corruption pairs from validation corrupted probe indices.
    save_original_vs_corrupted_grid(
        base=mnist_train,
        indices=val_protocol.corrupt_probe,
        out_path=figure_dir / cfg_get(vis_cfg, "save_val_probe_pairs", "corruption_pairs_val_probe_forced.png"),
        corruption_cfg=args.corruption,
        seed=args.seed + 222,
        num_examples=num_pair_examples,
        ncols=pair_grid_cols,
        title=f"{title_prefix}: validation probe original vs corrupted",
        force_corruption=force_pair,
        corruption_prob=1.0,
    )

    # 3. Forced corruption pairs from test corrupted probe indices.
    save_original_vs_corrupted_grid(
        base=mnist_test,
        indices=test_protocol.corrupt_probe,
        out_path=figure_dir / cfg_get(vis_cfg, "save_test_probe_pairs", "corruption_pairs_test_probe_forced.png"),
        corruption_cfg=args.corruption,
        seed=args.seed + 333,
        num_examples=num_pair_examples,
        ncols=pair_grid_cols,
        title=f"{title_prefix}: test probe original vs corrupted",
        force_corruption=force_pair,
        corruption_prob=1.0,
    )

    # 4. Actual ArcFace training batch.
    save_training_batch_grid(
        dataset=train_ds_arcface,
        out_path=figure_dir / cfg_get(vis_cfg, "save_arcface_train_batch", "corrupted_batch_arcface_train.png"),
        seed=args.seed + 444,
        batch_size=batch_size,
        ncols=batch_grid_cols,
        title=(
            f"{title_prefix}: actual ArcFace train batch "
            f"(p={args.train_corrupt_arcface_probability if args.train_corrupt_arcface_enabled else 0.0})"
        ),
        dpi=dpi,
    )

    # 5. Actual SCF training batch.
    save_training_batch_grid(
        dataset=train_ds_scf,
        out_path=figure_dir / cfg_get(vis_cfg, "save_scf_train_batch", "corrupted_batch_scf_train.png"),
        seed=args.seed + 555,
        batch_size=batch_size,
        ncols=batch_grid_cols,
        title=(
            f"{title_prefix}: actual SCF train batch "
            f"(p={args.train_corrupt_scf_probability if args.train_corrupt_scf_enabled else 0.0})"
        ),
        dpi=dpi,
    )

# ---------------------------------------------------------------------
# Plotly 3D teaser from trained 3D MNIST embeddings
# ---------------------------------------------------------------------


def sphere_surface_grid_for_plotly(
    n_theta: int = 120,
    n_phi: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    phi = np.linspace(0.0, np.pi, n_phi)

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return x, y, z, points


def sample_vmf_s2(
    mu: np.ndarray,
    kappa: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    vMF sampler on S^2.

    For S^2:
        t = cos(theta)
    has density proportional to exp(kappa * t), t in [-1,1].
    """
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / max(np.linalg.norm(mu), 1e-12)

    if kappa < 1e-8:
        x = rng.normal(size=(n, 3))
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    u = rng.random(n)
    t = -1.0 + np.log1p(u * np.expm1(2.0 * kappa)) / kappa
    t = np.clip(t, -1.0, 1.0)

    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = np.sqrt(np.maximum(1.0 - t**2, 0.0))

    local = np.stack(
        [
            r * np.cos(phi),
            r * np.sin(phi),
            t,
        ],
        axis=1,
    )

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, mu)) > 0.95:
        helper = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(helper, mu)
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(mu, e1)
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)

    samples = (
        local[:, 0:1] * e1[None, :]
        + local[:, 1:2] * e2[None, :]
        + local[:, 2:3] * mu[None, :]
    )

    return samples / np.maximum(np.linalg.norm(samples, axis=1, keepdims=True), 1e-12)


def plot_trained_3d_holue_teaser_plotly(
    stats: OSRStats,
    probe_mu: np.ndarray,
    gallery_mu: np.ndarray,
    known_classes: List[int],
    tau: float,
    gallery_kappa: float,
    uncertainty: np.ndarray,
    out_dir: Path,
    seed: int = 777,
    n_theta: int = 120,
    n_phi: int = 60,
    max_probe_points: int = 900,
) -> None:
    """
    Interactive Plotly teaser for trained 3D MNIST embeddings.

    Shows:
      - unit sphere colored by GalUE entropy H[p(c|z)];
      - learned gallery prototypes;
      - actual test probes, colored by HolUE uncertainty;
      - OSR errors as X markers;
      - a vMF cloud around the lowest-SCF-kappa probe to visualize p(z|x).
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print("[3D teaser] Plotly is not installed. Run: pip install plotly kaleido")
        print(f"Import error: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    probe_mu = normalize_np(np.asarray(probe_mu, dtype=np.float64))
    gallery_mu = normalize_np(np.asarray(gallery_mu, dtype=np.float64))
    uncertainty = np.asarray(uncertainty, dtype=np.float64)

    error_mask = osr_error_mask(
        pred_idx=stats.pred_idx,
        rejected=stats.rejected,
        labels=stats.labels,
        known_classes=known_classes,
    )

    # Surface entropy from deterministic GalUE p(c|z).
    x_grid, y_grid, z_grid, sphere_points = sphere_surface_grid_for_plotly(
        n_theta=n_theta,
        n_phi=n_phi,
    )

    sphere_post = posterior_from_z(
        sphere_points,
        gallery_mu=gallery_mu,
        tau=tau,
        gallery_kappa=gallery_kappa,
    )
    sphere_unc = entropy_normalized(sphere_post).reshape(x_grid.shape)

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            surfacecolor=sphere_unc,
            colorscale="Turbo",
            cmin=0.0,
            cmax=1.0,
            opacity=0.72,
            showscale=True,
            colorbar=dict(title="GalUE<br>entropy"),
            name="GalUE entropy surface",
            hovertemplate=(
                "z₁=%{x:.2f}<br>z₂=%{y:.2f}<br>z₃=%{z:.2f}"
                "<br>GalUE entropy=%{surfacecolor:.3f}<extra></extra>"
            ),
        )
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Gallery prototypes.
    for i, (digit, mu_i) in enumerate(zip(known_classes, gallery_mu)):
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter3d(
                x=[0.0, mu_i[0]],
                y=[0.0, mu_i[1]],
                z=[0.0, mu_i[2]],
                mode="lines",
                line=dict(color=color, width=5),
                opacity=0.6,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[mu_i[0]],
                y=[mu_i[1]],
                z=[mu_i[2]],
                mode="markers+text",
                marker=dict(
                    size=8,
                    color=color,
                    symbol="diamond",
                    line=dict(color="black", width=2),
                ),
                text=[f"digit {digit}"],
                textposition="top center",
                name=f"gallery digit {digit}",
                hovertemplate=f"Gallery digit {digit}<extra></extra>",
            )
        )

    # Choose probe points for display.
    n = len(probe_mu)
    if n > max_probe_points:
        err_idx = np.where(error_mask)[0]
        ok_idx = np.where(~error_mask)[0]

        # Keep as many errors as possible; fill remaining with correct probes.
        if len(err_idx) >= max_probe_points // 2:
            err_keep = rng.choice(err_idx, size=max_probe_points // 2, replace=False)
        else:
            err_keep = err_idx

        remaining = max_probe_points - len(err_keep)
        if len(ok_idx) > remaining:
            ok_keep = rng.choice(ok_idx, size=remaining, replace=False)
        else:
            ok_keep = ok_idx

        keep_idx = np.concatenate([err_keep, ok_keep])
    else:
        keep_idx = np.arange(n)

    keep_err = keep_idx[error_mask[keep_idx]]
    keep_ok = keep_idx[~error_mask[keep_idx]]

    unc_min = float(np.nanmin(uncertainty))
    unc_max = float(np.nanmax(uncertainty))

    # Correct probes.
    if len(keep_ok) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=probe_mu[keep_ok, 0],
                y=probe_mu[keep_ok, 1],
                z=probe_mu[keep_ok, 2],
                mode="markers",
                marker=dict(
                    size=3.2,
                    color=uncertainty[keep_ok],
                    colorscale="Viridis",
                    cmin=unc_min,
                    cmax=unc_max,
                    opacity=0.55,
                    showscale=False,
                ),
                name="correct probes",
                hovertemplate=(
                    "correct probe<br>"
                    "digit=%{customdata[0]}<br>"
                    "HolUE=%{customdata[1]:.3f}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [stats.labels[keep_ok], uncertainty[keep_ok]]
                ),
            )
        )

    # Error probes.
    if len(keep_err) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=probe_mu[keep_err, 0],
                y=probe_mu[keep_err, 1],
                z=probe_mu[keep_err, 2],
                mode="markers",
                marker=dict(
                    size=5.5,
                    color=uncertainty[keep_err],
                    colorscale="Plasma",
                    cmin=unc_min,
                    cmax=unc_max,
                    opacity=0.95,
                    symbol="x",
                    showscale=True,
                    colorbar=dict(title="HolUE<br>uncertainty", x=1.10),
                ),
                name="OSR errors",
                hovertemplate=(
                    "OSR error<br>"
                    "digit=%{customdata[0]}<br>"
                    "HolUE=%{customdata[1]:.3f}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [stats.labels[keep_err], uncertainty[keep_err]]
                ),
            )
        )

    # Highlight one high-uncertainty error and one low-uncertainty correct probe.
    high_err_idx = None
    err_all = np.where(error_mask)[0]
    if len(err_all) > 0:
        high_err_idx = int(err_all[np.argmax(uncertainty[err_all])])

    low_ok_idx = None
    ok_all = np.where(~error_mask)[0]
    if len(ok_all) > 0:
        low_ok_idx = int(ok_all[np.argmin(uncertainty[ok_all])])

    if high_err_idx is not None:
        p = probe_mu[high_err_idx]
        fig.add_trace(
            go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color="red",
                    symbol="cross",
                    line=dict(color="black", width=2),
                ),
                text=[f"high HolUE<br>{uncertainty[high_err_idx]:.2f}"],
                textposition="bottom center",
                name="high-uncertainty error",
            )
        )

    if low_ok_idx is not None:
        p = probe_mu[low_ok_idx]
        fig.add_trace(
            go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers+text",
                marker=dict(
                    size=9,
                    color="limegreen",
                    symbol="circle",
                    line=dict(color="black", width=2),
                ),
                text=[f"low HolUE<br>{uncertainty[low_ok_idx]:.2f}"],
                textposition="bottom center",
                name="low-uncertainty correct",
            )
        )

    # Visualize p(z|x) for lowest-quality probe via vMF samples.
    low_quality_idx = int(np.argmin(stats.scf_kappa))
    cloud = sample_vmf_s2(
        mu=probe_mu[low_quality_idx],
        kappa=float(stats.scf_kappa[low_quality_idx]),
        n=500,
        rng=rng,
    )

    fig.add_trace(
        go.Scatter3d(
            x=cloud[:, 0],
            y=cloud[:, 1],
            z=cloud[:, 2],
            mode="markers",
            marker=dict(
                size=2.2,
                color="cyan",
                opacity=0.23,
            ),
            name="low-quality p(z|x) cloud",
            hoverinfo="skip",
        )
    )

    p = probe_mu[low_quality_idx]
    fig.add_trace(
        go.Scatter3d(
            x=[p[0]],
            y=[p[1]],
            z=[p[2]],
            mode="markers+text",
            marker=dict(
                size=8,
                color="cyan",
                symbol="circle",
                line=dict(color="black", width=2),
            ),
            text=[f"low SCF κ<br>{stats.scf_kappa[low_quality_idx]:.1f}"],
            textposition="top center",
            name="low-quality mean",
        )
    )

    # Origin.
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker=dict(size=3, color="black"),
            name="origin",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=(
            "Trained 3D MNIST HolUE teaser"
            "<br><sup>Normalized 3D embeddings on S². "
            "Surface: GalUE entropy; probes: HolUE uncertainty.</sup>"
        ),
        width=1000,
        height=840,
        scene=dict(
            xaxis=dict(title="z₁", range=[-1.22, 1.22], showbackground=False),
            yaxis=dict(title="z₂", range=[-1.22, 1.22], showbackground=False),
            zaxis=dict(title="z₃", range=[-1.22, 1.22], showbackground=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.10)),
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.78)",
        ),
        margin=dict(l=0, r=0, t=90, b=0),
    )

    html_path = out_dir / "stylized_3d_holue_trained_teaser.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"[3D teaser] Saved HTML: {html_path}")

    try:
        png_path = out_dir / "stylized_3d_holue_trained_teaser.png"
        pdf_path = out_dir / "stylized_3d_holue_trained_teaser.pdf"
        fig.write_image(str(png_path), scale=2)
        fig.write_image(str(pdf_path))
        print(f"[3D teaser] Saved PNG:  {png_path}")
        print(f"[3D teaser] Saved PDF:  {pdf_path}")
    except Exception as exc:
        print("[3D teaser] Static PNG/PDF export skipped.")
        print("Install with: pip install kaleido")
        print(f"Export error: {exc}")


# ---------------------------------------------------------------------
# Stylized Bayesian circle teaser, similar to circle_plot_utils.py
# ---------------------------------------------------------------------


def toy_get_vectors_by_angle(angles: np.ndarray) -> np.ndarray:
    return np.array(
        [[np.cos(a), np.sin(a)] for a in angles],
        dtype=np.float64,
    )


def toy_z_vonmises_density(
    z: np.ndarray,
    mu_c: np.ndarray,
    kappa: float,
    d: int = 2,
) -> np.ndarray:
    """
    vMF density on S^1. This is the same style as the notebook function.
    For d=2 this is the circular von Mises density.
    """
    z = np.asarray(z, dtype=np.float64)
    mu_c = np.asarray(mu_c, dtype=np.float64)

    C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))
    return C_d * np.exp(kappa * (z @ mu_c))


def toy_z_power_density(
    z: np.ndarray,
    mu_c: np.ndarray,
    kappa: float,
    d: int = 2,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    mu_c = np.asarray(mu_c, dtype=np.float64)

    alpha = (d - 1) / 2 + kappa
    beta = (d - 1) / 2
    M_d = gamma(alpha + beta) / (2 ** (alpha + beta) * np.pi**beta * gamma(alpha))
    return M_d * np.maximum(1.0 + z @ mu_c, 1e-12) ** kappa


def toy_z_prob(
    z: np.ndarray,
    mus: np.ndarray,
    kappa: float,
    beta: float = 0.5,
    d: int = 2,
) -> np.ndarray:
    """
    Marginal p(z) under:
      known classes: vMF(z; mu_c, kappa)
      unknown/OOG: uniform on the unit circle.
    """
    z = np.asarray(z, dtype=np.float64)
    mus = np.asarray(mus, dtype=np.float64)  # shape: 2 x K

    K = mus.shape[1]
    p_c = (1.0 - beta) / K

    class_probs = []
    for i in range(K):
        class_probs.append(toy_z_vonmises_density(z, mus[:, i], kappa, d=d))
    class_probs = np.stack(class_probs, axis=0)  # K x N

    return np.sum(class_probs * p_c, axis=0) + beta / (2.0 * np.pi)


def toy_z_class_prob(
    class_id: int,
    z: np.ndarray,
    mus: np.ndarray,
    kappa: float,
    beta: float = 0.5,
    d: int = 2,
) -> np.ndarray:
    """
    Posterior p(c|z) for known classes and the OOG class.

    class_id = 0..K-1: known class
    class_id = K: OOG class
    """
    z = np.asarray(z, dtype=np.float64)
    mus = np.asarray(mus, dtype=np.float64)

    K = mus.shape[1]
    p_z = np.maximum(toy_z_prob(z, mus, kappa, beta=beta, d=d), 1e-300)

    if class_id == K:
        return (beta / (2.0 * np.pi)) / p_z

    p_c = (1.0 - beta) / K
    return toy_z_vonmises_density(z, mus[:, class_id], kappa, d=d) * p_c / p_z


def toy_compute_all_class_probs(
    zs: np.ndarray,
    mus: np.ndarray,
    kappa: float,
    beta: float,
) -> np.ndarray:
    K = mus.shape[1]
    probs = []
    for i in range(K + 1):
        probs.append(toy_z_class_prob(i, zs, mus, kappa, beta=beta))
    probs = np.stack(probs, axis=0)  # (K+1) x N
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=0, keepdims=True)
    return probs


def toy_draw_circle(ax, linewidth: float = 3.0) -> None:
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.add_patch(plt.Circle((0, 0), 1, color="tab:blue", alpha=0.08, zorder=0))
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color="tab:gray",
        linewidth=linewidth,
        zorder=4,
    )
    ax.scatter([0], [0], color="black", s=20, zorder=6)
    ax.axis("off")


def circular_distance(a: float, b: float) -> float:
    return abs(np.angle(np.exp(1j * (a - b))))


def circular_midpoint(a: float, b: float) -> float:
    """
    Midpoint on the unit circle between angles a and b.
    """
    v = np.array([np.cos(a), np.sin(a)]) + np.array([np.cos(b), np.sin(b)])
    if np.linalg.norm(v) < 1e-12:
        return a + np.pi / 2
    return float(np.arctan2(v[1], v[0]))


def plot_stylized_bayesian_circle_teaser(
    gallery_mu: np.ndarray,
    known_classes: List[int],
    out_dir: Path,
    kappa: float = 8.0,
    beta: float = 0.5,
    unc_type: str = "entropy",
    draw_oog: bool = True,
) -> None:
    """
    Additional stylized toy teaser, following the visual language of
    circle_plot_utils.py.

    It draws:
      - unit circle;
      - gallery class directions;
      - radial posterior curves p(c|z);
      - optional OOG posterior curve;
      - red uncertainty curve;
      - two cyan test points:
          one near a class-boundary, one near a gallery class.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    gallery_mu = normalize_np(np.asarray(gallery_mu, dtype=np.float64))
    angles = np.arctan2(gallery_mu[:, 1], gallery_mu[:, 0])
    angles = np.mod(angles, 2.0 * np.pi)

    order = np.argsort(angles)
    angles = angles[order]
    gallery_mu = gallery_mu[order]
    known_classes_sorted = [known_classes[i] for i in order]

    K = len(known_classes_sorted)
    mus = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # 2 x K

    # Find the closest pair of gallery classes to highlight identification ambiguity.
    best_i, best_j = 0, 1
    best_dist = float("inf")
    for i in range(K):
        for j in range(i + 1, K):
            dist = circular_distance(angles[i], angles[j])
            if dist < best_dist:
                best_dist = dist
                best_i, best_j = i, j

    boundary_angle = circular_midpoint(angles[best_i], angles[best_j])

    # A confident point near one of the classes.
    confident_angle = angles[best_i]

    test_points_angles = np.array([boundary_angle, confident_angle], dtype=np.float64)
    test_point_vectors = toy_get_vectors_by_angle(test_points_angles)

    theta = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    zs = toy_get_vectors_by_angle(theta)

    colors = list(mcolors.TABLEAU_COLORS)[:K]

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    toy_draw_circle(ax, linewidth=3.0)

    # Draw known class posterior radial curves.
    local_range = np.pi / 3
    local_offsets = np.linspace(-local_range, local_range, 180)

    for i, (angle, color) in enumerate(zip(angles, colors)):
        plot_angles = angle + local_offsets
        local_zs = toy_get_vectors_by_angle(plot_angles)
        class_probs = toy_z_class_prob(i, local_zs, mus, kappa, beta=beta)

        v = local_zs.T * (1.0 + class_probs[np.newaxis, :])
        ax.plot(v[0], v[1], color=color, linewidth=3.0)

        ax.scatter(
            [np.cos(angle)],
            [np.sin(angle)],
            c=color,
            s=95,
            zorder=6,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.text(
            1.23 * np.cos(angle),
            1.23 * np.sin(angle),
            str(known_classes_sorted[i]),
            color=color,
            fontsize=15,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Optional OOG posterior curve.
    if draw_oog:
        oog_probs = toy_z_class_prob(K, zs, mus, kappa, beta=beta)
        v_oog = zs.T * (1.0 + oog_probs[np.newaxis, :])
        ax.plot(
            v_oog[0],
            v_oog[1],
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
            label="OOG posterior",
        )

    # Draw uncertainty curve.
    all_probs = toy_compute_all_class_probs(zs, mus, kappa, beta=beta)

    if unc_type == "entropy":
        unc = -np.sum(all_probs * np.log(all_probs), axis=0)
        unc = unc / np.log(K + 1)  # normalized to [0,1]
        unc_label = "normalized entropy"
    elif unc_type == "max_prob":
        unc = 1.0 - np.max(all_probs, axis=0)
        unc_label = "1 - max prob"
    else:
        raise ValueError(f"Unknown unc_type={unc_type}")

    v_unc = zs.T * (1.0 + unc[np.newaxis, :])
    ax.plot(
        v_unc[0],
        v_unc[1],
        color="tab:red",
        linewidth=3.5,
        label=f"uncertainty ({unc_label})",
    )

    # Test points.
    ax.scatter(
        test_point_vectors[:, 0],
        test_point_vectors[:, 1],
        c="tab:cyan",
        s=110,
        zorder=8,
        edgecolor="black",
        linewidth=0.8,
        label="test probes",
    )

    probs_at_test = toy_compute_all_class_probs(
        test_point_vectors,
        mus,
        kappa,
        beta=beta,
    )

    if unc_type == "entropy":
        unc_test = -np.sum(probs_at_test * np.log(probs_at_test), axis=0)
        unc_test = unc_test / np.log(K + 1)
    else:
        unc_test = 1.0 - np.max(probs_at_test, axis=0)

    unc_test = np.round(unc_test, 2)

    # Annotations.
    ax.annotate(
        f"high\n{unc_test[0]:.2f}",
        xy=test_point_vectors[0],
        xytext=(test_point_vectors[0, 0] + 0.12, test_point_vectors[0, 1] + 0.18),
        fontsize=14,
        color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5),
    )

    ax.annotate(
        f"low\n{unc_test[1]:.2f}",
        xy=test_point_vectors[1],
        xytext=(test_point_vectors[1, 0] + 0.12, test_point_vectors[1, 1] - 0.30),
        fontsize=14,
        color="tab:green",
        arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.5),
    )

    ax.set_aspect("equal")
    ax.set_title(
        "Gallery-aware uncertainty on the unit circle",
        fontsize=14,
    )

    ax.legend(loc="lower left", fontsize=8.5, frameon=True)

    plt.savefig(
        out_dir / "stylized_bayesian_circle_teaser.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )
    plt.savefig(
        out_dir / "stylized_bayesian_circle_teaser.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------




def run_3d_extension(
    args,
    known_classes: List[int],
    unknown_classes: List[int],
    class_to_idx: Dict[int, int],
    mnist_train: datasets.MNIST,
    mnist_test: datasets.MNIST,
    train_indices: np.ndarray,
    val_protocol: ProtocolIndices,
    test_protocol: ProtocolIndices,
    device: torch.device,
    out_dir: Path,
) -> None:
    """
    Train and evaluate a separate 3D ArcFace+SCF MNIST HolUE model.

    This is intentionally independent from the 2D branch:
      - new ArcFace checkpoint with embedding_dim=3;
      - new SCF checkpoint with embedding_dim=3;
      - 3D HolUE integration on S^2 by deterministic quadrature;
      - Plotly 3D teaser from trained embeddings.
    """
    print("\n" + "=" * 80)
    print("Running 3D MNIST HolUE extension")
    print("=" * 80)

    model_dir = resolve_under_out_dir(args.model_dir, out_dir)
    result_dir = resolve_under_out_dir(args.result_dir_3d, out_dir)
    figure_dir = resolve_under_out_dir(args.figure_dir_3d, out_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    arc_epochs = (
        args.arcface_epochs
        if args.arcface_epochs_3d is None
        else args.arcface_epochs_3d
    )
    scf_epochs = args.scf_epochs if args.scf_epochs_3d is None else args.scf_epochs_3d

    train_ds_arcface = make_known_training_dataset(
        args=args,
        base=mnist_train,
        train_indices=train_indices,
        class_to_idx=class_to_idx,
        stage="arcface",
    )

    train_ds_scf = make_known_training_dataset(
        args=args,
        base=mnist_train,
        train_indices=train_indices,
        class_to_idx=class_to_idx,
        stage="scf",
    )

    # ---------------------------
    # Train/load 3D ArcFace
    # ---------------------------

    arc3_ckpt = model_dir / args.arcface_checkpoint_3d
    arc3 = TinyArcFaceMNIST(
    num_classes=len(known_classes),
    embedding_dim=args.embedding_dim_3d,
)

    if arc3_ckpt.is_file() and not args.force_train:
        ckpt = torch.load(arc3_ckpt, map_location=device, weights_only=False)
        arc3.load_state_dict(ckpt["state_dict"])
        print(f"Loaded 3D ArcFace checkpoint: {arc3_ckpt}")
    else:
        train_arcface(
    model=arc3,
    train_ds=train_ds_arcface,
    device=device,
    epochs=args.arcface_epochs_3d,
    batch_size=args.batch_size,
    lr=args.lr_arcface,
    seed=args.seed + 3000,
    arcface_s=args.arcface_s,
    arcface_m=args.arcface_m,
    weight_decay=args.arcface_weight_decay,
)
        torch.save(
            {
                "state_dict": arc3.state_dict(),
                "known_classes": known_classes,
                "embedding_dim": 3,
                "args": to_jsonable(args),
            },
            arc3_ckpt,
        )
        print(f"Saved 3D ArcFace checkpoint: {arc3_ckpt}")

    # ---------------------------
    # Train/load 3D SCF
    # ---------------------------

    scf3_ckpt = model_dir / args.scf_checkpoint_3d
    scf3 = TinySCFMNIST(
    arc_model=arc3,
    kappa_min=args.scf_kappa_min,
    kappa_max=args.scf_kappa_max,
)

    if scf3_ckpt.is_file() and not args.force_train:
        ckpt = torch.load(scf3_ckpt, map_location=device, weights_only=False)
        scf3.load_state_dict(ckpt["state_dict"])
        print(f"Loaded 3D SCF checkpoint: {scf3_ckpt}")
    else:
        train_scf(
    scf_model=scf3,
    arc_model=arc3,
    train_ds=train_ds_scf,
    device=device,
    epochs=args.scf_epochs_3d,
    batch_size=args.batch_size,
    lr=args.lr_scf,
    seed=args.seed + 4000,
    weight_decay=args.scf_weight_decay,
    kappa_regularizer=args.scf_kappa_regularizer,
)
        torch.save(
            {
                "state_dict": scf3.state_dict(),
                "known_classes": known_classes,
                "embedding_dim": 3,
                "args": to_jsonable(args),
            },
            scf3_ckpt,
        )
        print(f"Saved 3D SCF checkpoint: {scf3_ckpt}")

    # ---------------------------
    # Extract embeddings
    # ---------------------------

    print("[3D] Extracting validation gallery embeddings...")
    val_gallery_emb, val_gallery_labels = extract_arc_embeddings(
        arc3,
        mnist_train,
        val_protocol.gallery,
        device,
        args.batch_size,
        corrupt_indices=[],
        corrupt_seed=args.seed,
        corruption_cfg=args.corruption,
    )
    val_gallery_mu = build_gallery_prototypes(
        val_gallery_emb,
        val_gallery_labels,
        known_classes,
    )

    print("[3D] Extracting validation probe embeddings + SCF kappa...")
    val_probe_mu, val_probe_kappa, val_probe_labels = extract_scf_embeddings(
        scf3,
        mnist_train,
        val_protocol.probe,
        device,
        args.batch_size,
        corrupt_indices=val_protocol.corrupt_probe,
        corrupt_seed=args.seed + 10000,
        corruption_cfg=args.corruption,
    )

    print("[3D] Extracting test gallery embeddings...")
    test_gallery_emb, test_gallery_labels = extract_arc_embeddings(
        arc3,
        mnist_test,
        test_protocol.gallery,
        device,
        args.batch_size,
        corrupt_indices=[],
        corrupt_seed=args.seed,
        corruption_cfg=args.corruption,
    )
    test_gallery_mu = build_gallery_prototypes(
        test_gallery_emb,
        test_gallery_labels,
        known_classes,
    )

    print("[3D] Extracting test probe embeddings + SCF kappa...")
    test_probe_mu, test_probe_kappa, test_probe_labels = extract_scf_embeddings(
        scf3,
        mnist_test,
        test_protocol.probe,
        device,
        args.batch_size,
        corrupt_indices=test_protocol.corrupt_probe,
        corrupt_seed=args.seed + 20000,
        corruption_cfg=args.corruption,
    )

    np.savez(
        result_dir / "mnist_3d_embeddings_for_toy_osr.npz",
        val_gallery_mu=val_gallery_mu,
        val_probe_mu=val_probe_mu,
        val_probe_kappa=val_probe_kappa,
        val_probe_labels=val_probe_labels,
        test_gallery_mu=test_gallery_mu,
        test_probe_mu=test_probe_mu,
        test_probe_kappa=test_probe_kappa,
        test_probe_labels=test_probe_labels,
        known_classes=np.asarray(known_classes, dtype=int),
        unknown_classes=np.asarray(unknown_classes, dtype=int),
    )

    # ---------------------------
    # Select threshold on validation unknown probes
    # ---------------------------

    val_sim = val_probe_mu @ val_gallery_mu.T
    val_max_sim = val_sim.max(axis=1)
    val_seen = np.isin(val_probe_labels, np.asarray(known_classes))
    tau = threshold_at_fpir(val_max_sim[~val_seen], args.target_fpir)

    print(f"[3D] Validation-selected tau={tau:.6f} for target FPIR={args.target_fpir}")
    effective_beta_3d = args.osr_beta
    if effective_beta_3d is None:
        effective_beta_3d = beta_from_tau_mixed_prior(
            dim=args.embedding_dim_3d,
            tau=tau,
            gallery_kappa=args.gallery_kappa,
            K=len(known_classes),
        )

    print(
        f"[3D] Continuous-OOG HolUE: enabled={args.continuous_oog}, "
        f"effective beta={effective_beta_3d:.6g}"
    )
    # ---------------------------
    # Compute 3D OSR stats
    # ---------------------------

    print("[3D] Computing validation HolUE/GalUE posteriors on S^2...")
    val_stats = compute_osr_stats_nd(
        probe_mu=val_probe_mu,
        probe_kappa=val_probe_kappa,
        probe_labels=val_probe_labels,
        gallery_mu=val_gallery_mu,
        tau=tau,
        gallery_kappa=args.gallery_kappa,
        circle_grid=args.circle_grid,
        sphere_theta_grid=args.sphere_theta_grid,
        sphere_phi_grid=args.sphere_phi_grid,
                beta=effective_beta_3d,
        continuous_oog=args.continuous_oog,
    )
    
    setattr(val_stats, "probe_mu_for_plot", val_probe_mu)

    print("[3D] Computing test HolUE/GalUE posteriors on S^2...")
    test_stats = compute_osr_stats_nd(
        probe_mu=test_probe_mu,
        probe_kappa=test_probe_kappa,
        probe_labels=test_probe_labels,
        gallery_mu=test_gallery_mu,
        tau=tau,
        gallery_kappa=args.gallery_kappa,
        circle_grid=args.circle_grid,
        sphere_theta_grid=args.sphere_theta_grid,
        sphere_phi_grid=args.sphere_phi_grid,
                beta=effective_beta_3d,
        continuous_oog=args.continuous_oog,
    )
    setattr(test_stats, "probe_mu_for_plot", test_probe_mu)

    np.savez(
        result_dir / "posteriors_and_osr_stats_3d.npz",
        tau=tau,
        gallery_kappa=args.gallery_kappa,
        val_labels=val_stats.labels,
        val_max_sim=val_stats.max_sim,
        val_pred_idx=val_stats.pred_idx,
        val_rejected=val_stats.rejected,
        val_scf_kappa=val_stats.scf_kappa,
        val_gal_posterior=val_stats.gal_posterior,
        val_holue_posterior=val_stats.holue_posterior,
        val_gal_entropy=val_stats.gal_entropy,
        val_holue_entropy=val_stats.holue_entropy,
        test_labels=test_stats.labels,
        test_max_sim=test_stats.max_sim,
        test_pred_idx=test_stats.pred_idx,
        test_rejected=test_stats.rejected,
        test_scf_kappa=test_stats.scf_kappa,
        test_gal_posterior=test_stats.gal_posterior,
        test_holue_posterior=test_stats.holue_posterior,
        test_gal_entropy=test_stats.gal_entropy,
        test_holue_entropy=test_stats.holue_entropy,
        test_kl_known=test_stats.kl_known,
        test_kl_oog=test_stats.kl_oog,
        test_kl_total=test_stats.kl_total,
    )

    # ---------------------------
    # Base OSR metrics
    # ---------------------------

    val_error = osr_error_mask(
        pred_idx=val_stats.pred_idx,
        rejected=val_stats.rejected,
        labels=val_stats.labels,
        known_classes=known_classes,
    )
    test_error = osr_error_mask(
        pred_idx=test_stats.pred_idx,
        rejected=test_stats.rejected,
        labels=test_stats.labels,
        known_classes=known_classes,
    )

    test_base_metrics = compute_osr_metrics(
        pred_idx=test_stats.pred_idx,
        rejected=test_stats.rejected,
        labels=test_stats.labels,
        known_classes=known_classes,
    )

    print("\n[3D] Test OSR metrics:")
    for k, v in test_base_metrics.items():
        print(f"  {k}: {v}")

    # ---------------------------
    # Calibrated HolUE comparison
    # ---------------------------

    def make_calibration_features_3d(stats: OSRStats) -> np.ndarray:
        return np.column_stack(
            [
                stats.holue_entropy,
                stats.gal_entropy,
                np.log(stats.scf_kappa + 1e-8),
                np.abs(stats.max_sim - tau),
                stats.kl_known,
                stats.kl_oog,
                stats.kl_total,
                stats.max_sim,
            ]
        ).astype(np.float64)

    X_val = make_calibration_features_3d(val_stats)
    y_val = val_error.astype(int)
    X_test = make_calibration_features_3d(test_stats)

    calibration_info: Dict[str, object] = {
        "used": False,
        "reason": "",
        "embedding_dim": 3,
    }

    if len(np.unique(y_val)) < 2:
        holue_calibrated_unc = test_stats.holue_entropy.copy()
        calibration_info["reason"] = "single_class_validation_error_labels"
    else:
        scaler = StandardScaler()
        X_val_scaled = scaler.fit_transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(
    class_weight=args.calibration_class_weight,
    random_state=args.calibration_random_state,
    max_iter=args.calibration_max_iter,
    solver=args.calibration_solver,
)
        clf.fit(X_val_scaled, y_val)

        holue_calibrated_unc = clf.predict_proba(X_test_scaled)[:, 1]

        calibration_info.update(
            {
                "used": True,
                "reason": "ok",
                "intercept": clf.intercept_.tolist(),
                "coef": clf.coef_.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "val_error_rate": float(np.mean(y_val)),
            }
        )

    with open(
        result_dir / "holue_calibration_info_3d.json", "w", encoding="utf-8"
    ) as f:
        json.dump(calibration_info, f, indent=2)

    # ---------------------------
    # Uncertainty scores
    # ---------------------------

    rng_eval = np.random.default_rng(args.seed + 54321)
    random_unc = rng_eval.random(len(test_stats.labels))
    oracle_unc = test_error.astype(float) + 1e-6 * rng_eval.random(len(test_error))

    # Paper-style HolUE uncertainty:
    # high KL = high identity information = low uncertainty.
    # Therefore uncertainty is -KL.
    holue_mixed_kl_unc = -test_stats.kl_total

    # Collapsed action risk is an oracle diagnostic:
    # it is not the same as mixed identity-information uncertainty.
    holue_action_risk = action_risk_from_posterior(test_stats)

    uncertainties: Dict[str, np.ndarray] = {
        # Main no-calibration HolUE score matching the paper-style mixed KL.
        "HolUE no calibration": holue_mixed_kl_unc,

        # Validation-calibrated comparison.
        "HolUE calibrated": holue_calibrated_unc,

        # Diagnostics / ablations.
        "HolUE raw KL": holue_mixed_kl_unc,
        "HolUE collapsed entropy": test_stats.holue_entropy,
        "HolUE action risk": holue_action_risk,

        # Baselines.
        "GalUE": test_stats.gal_entropy,
        "SCF": -np.log(test_stats.scf_kappa + 1e-8),
        "AccScr": -np.abs(test_stats.max_sim - tau),
        "MaxSim": -test_stats.max_sim,
        "Random": random_unc,
        "Oracle": oracle_unc,
    }

    np.savez(
        result_dir / "uncertainty_scores_3d.npz",
        **{k.replace(" ", "_").replace("-", "_"): v for k, v in uncertainties.items()},
    )

    # ---------------------------
    # Rejection curves
    # ---------------------------

    fractions = np.linspace(
    args.rejection_fraction_start,
    args.rejection_fraction_stop,
    args.rejection_fraction_num,
)

    def slugify(name: str) -> str:
        out = name.lower()
        for ch in [" ", "-", "/", "(", ")", "[", "]", "{", "}", "."]:
            out = out.replace(ch, "_")
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_")

    curves: Dict[str, pd.DataFrame] = {}

    for name, unc in uncertainties.items():
        df = rejection_curve(
            uncertainty=unc,
            pred_idx=test_stats.pred_idx,
            rejected=test_stats.rejected,
            labels=test_stats.labels,
            known_classes=known_classes,
            fractions=fractions,
        )
        curves[name] = df
        df.to_csv(result_dir / f"rejection_curve_{slugify(name)}_3d.csv", index=False)

    random_curve = curves["Random"]
    oracle_curve = curves["Oracle"]

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        average_precision_score = None
        roc_auc_score = None

    rows = []

    for name, unc in uncertainties.items():
        df = curves[name]
        prr_f1 = compute_prr(
            curve=df,
            random_curve=random_curve,
            oracle_curve=oracle_curve,
            metric="f1",
        )

        if roc_auc_score is not None and len(np.unique(test_error)) == 2:
            error_auc = float(roc_auc_score(test_error.astype(int), unc))
            error_ap = float(average_precision_score(test_error.astype(int), unc))
        else:
            error_auc = float("nan")
            error_ap = float("nan")

        rows.append(
            {
                "method": name,
                "embedding_dim": 3,
                "base_f1": test_base_metrics["f1"],
                "base_fpir": test_base_metrics["fpir"],
                "base_fnir": test_base_metrics["fnir"],
                "base_error_rate": test_base_metrics["error_rate"],
                "prr_f1": prr_f1,
                "error_roc_auc": error_auc,
                "error_average_precision": error_ap,
                "f1_after_50pct_filter": float(df.iloc[-1]["f1"]),
                "fpir_after_50pct_filter": float(df.iloc[-1]["fpir"]),
                "fnir_after_50pct_filter": float(df.iloc[-1]["fnir"]),
            }
        )

    summary_df = pd.DataFrame(rows)

    method_order = [
        "HolUE no calibration",
        "HolUE calibrated",
        "HolUE raw KL",
        "GalUE",
        "SCF",
        "AccScr",
        "MaxSim",
        "Random",
        "Oracle",
    ]

    summary_df["method"] = pd.Categorical(
        summary_df["method"],
        categories=method_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values("method").reset_index(drop=True)

    summary_df.to_csv(result_dir / "method_summary_3d.csv", index=False)

    print("\n[3D] Method summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # This plot_rejection_curves function is your PRR-enhanced version.
    plot_rejection_curves(
    curves,
    figure_dir,
    metric=args.plots.rejection_curves.metric,
    plot_cfg=args.plots.rejection_curves,
    method_order=args.methods.order,
)

    # ---------------------------
    # Plotly trained 3D teaser
    # ---------------------------

    if args.three_d.plotly.enabled:
        plot_trained_3d_holue_teaser_plotly(
            stats=test_stats,
            probe_mu=test_probe_mu,
            gallery_mu=test_gallery_mu,
            known_classes=known_classes,
            tau=tau,
            gallery_kappa=args.gallery_kappa,
            uncertainty=test_stats.holue_entropy,
            out_dir=figure_dir,
            seed=args.seed,
            n_theta=args.three_d.plotly.n_theta_surface,
            n_phi=args.three_d.plotly.n_phi_surface,
            max_probe_points=args.three_d.plotly.max_probe_points,
        )

    # ---------------------------
    # Report
    # ---------------------------

    report = f"""# 3D MNIST HolUE toy extension

This is the trained 3D version of the MNIST toy OSR experiment.

## Protocol

- Known MNIST classes: `{known_classes}`
- Unknown MNIST classes: `{unknown_classes}`
- Embedding dimension: `3`
- Embeddings are L2-normalized and live on `S^2`
- Validation-selected cosine threshold tau: `{tau:.6f}`
- Target validation FPIR: `{args.target_fpir}`
- Gallery posterior kappa: `{args.gallery_kappa}`

## Base OSR test metrics

```json
{json.dumps(test_base_metrics, indent=2)}
```

## Key outputs

- `models/tiny_arcface_mnist_3d.pt`
- `models/tiny_scf_mnist_3d.pt`
- `results_3d/method_summary_3d.csv`
- `figures_3d/rejection_curves.pdf`
- `figures_3d/stylized_3d_holue_trained_teaser.html`

The no-calibration HolUE score is the entropy of:

`p(c|x) = ∫_S² p(c|z) p(z|x) dz`

computed by deterministic quadrature on the unit sphere.
"""

    with open(out_dir / "README_3d_toy_results.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n[3D] Done.")
    print(f"[3D] Summary: {(result_dir / 'method_summary_3d.csv').resolve()}")
    print(
        f"[3D] Plotly teaser: {(figure_dir / 'stylized_3d_holue_trained_teaser.html').resolve()}"
    )

# ---------------------------------------------------------------------
# Exact oracle circle experiment
# ---------------------------------------------------------------------


def angle_to_unit_vectors(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def sample_oracle_circle_split(
    n: int,
    center_angles: np.ndarray,
    beta: float,
    gallery_kappa: float,
    kappa_high: float,
    kappa_low: float,
    low_quality_prob: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an exact synthetic OSR split on S^1.

    Known identities:
        c = 0,...,K-1
        z_true | c ~ vonMises(center_c, gallery_kappa)

    Unknown identities:
        z_true ~ Uniform(S^1)

    Observation/probabilistic embedding:
        mu_obs | z_true, kappa_x ~ vonMises(z_true, kappa_x)

    We then treat:
        p(z|x) = vMF(mu_obs, kappa_x)

    This is the clean toy setting where the HolUE assumptions are true by
    construction at the embedding level.
    """
    K = len(center_angles)

    labels = np.full(n, -1, dtype=int)
    theta_true = np.empty(n, dtype=np.float64)

    is_unknown = rng.random(n) < beta
    is_known = ~is_unknown

    known_idx = np.where(is_known)[0]
    unknown_idx = np.where(is_unknown)[0]

    if len(known_idx) > 0:
        known_labels = rng.integers(0, K, size=len(known_idx))
        labels[known_idx] = known_labels

        for out_i, c in zip(known_idx, known_labels):
            theta_true[out_i] = rng.vonmises(center_angles[c], gallery_kappa)

    if len(unknown_idx) > 0:
        theta_true[unknown_idx] = rng.uniform(0.0, 2.0 * math.pi, size=len(unknown_idx))

    low_quality = rng.random(n) < low_quality_prob
    kappa_x = np.where(low_quality, float(kappa_low), float(kappa_high))

    theta_obs = np.empty(n, dtype=np.float64)
    for i in range(n):
        theta_obs[i] = rng.vonmises(theta_true[i], kappa_x[i])

    mu_obs = angle_to_unit_vectors(theta_obs)

    return mu_obs, kappa_x, labels, theta_true, low_quality


def run_oracle_circle_experiment(args, out_dir: Path) -> None:
    """
    Synthetic exact circle experiment.

    This is the experiment that should demonstrate:

      - if p(c|z) and p(z|x) are correct,
      - and the continuous OOG prior is used,
      - then an oracle posterior score needs no MLP calibration.

    It saves outputs into:

        toy_outputs/oracle_circle_results/
        toy_outputs/oracle_circle_figures/
    """
    ocfg = cfg_get(args.full_config, "oracle_circle", None)

    if not bool(cfg_get(ocfg, "enabled", False)):
        return

    print("\n" + "=" * 80)
    print("Running exact oracle circle HolUE experiment")
    print("=" * 80)

    result_dir = out_dir / "oracle_circle_results"
    figure_dir = out_dir / "oracle_circle_figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg_get(ocfg, "seed", args.seed + 90000))
    rng = np.random.default_rng(seed)

    K = int(cfg_get(ocfg, "num_known_classes", len(args.known_classes)))
    n_val = int(cfg_get(ocfg, "n_val", 6000))
    n_test = int(cfg_get(ocfg, "n_test", 6000))

    beta = float(cfg_get(ocfg, "beta", 0.5))
    gallery_kappa = float(cfg_get(ocfg, "gallery_kappa", 18.0))
    kappa_high = float(cfg_get(ocfg, "kappa_high", 80.0))
    kappa_low = float(cfg_get(ocfg, "kappa_low", 1.5))
    low_quality_prob = float(cfg_get(ocfg, "low_quality_prob", 0.30))
    circle_grid = int(cfg_get(ocfg, "circle_grid", args.circle_grid))

    # Equally spaced known identity centers.
    center_angles = np.linspace(0.0, 2.0 * math.pi, K, endpoint=False)
    gallery_mu = angle_to_unit_vectors(center_angles)

    # Choose tau implied by beta, so Bayesian posterior and OSR threshold agree.
    tau = tau_from_beta_mixed_prior(
        dim=2,
        beta=beta,
        gallery_kappa=gallery_kappa,
        K=K,
    )

    known_classes_oracle = list(range(K))

    print(f"[Oracle circle] K={K}")
    print(f"[Oracle circle] beta={beta}")
    print(f"[Oracle circle] gallery_kappa={gallery_kappa}")
    print(f"[Oracle circle] implied tau={tau:.6f}")
    print(f"[Oracle circle] kappa_high={kappa_high}, kappa_low={kappa_low}")
    print(f"[Oracle circle] low_quality_prob={low_quality_prob}")

    val_mu, val_kappa, val_labels, _, val_lowq = sample_oracle_circle_split(
        n=n_val,
        center_angles=center_angles,
        beta=beta,
        gallery_kappa=gallery_kappa,
        kappa_high=kappa_high,
        kappa_low=kappa_low,
        low_quality_prob=low_quality_prob,
        rng=rng,
    )

    test_mu, test_kappa, test_labels, _, test_lowq = sample_oracle_circle_split(
        n=n_test,
        center_angles=center_angles,
        beta=beta,
        gallery_kappa=gallery_kappa,
        kappa_high=kappa_high,
        kappa_low=kappa_low,
        low_quality_prob=low_quality_prob,
        rng=rng,
    )

    val_stats = compute_osr_stats(
        probe_mu=val_mu,
        probe_kappa=val_kappa,
        probe_labels=val_labels,
        gallery_mu=gallery_mu,
        tau=tau,
        gallery_kappa=gallery_kappa,
        n_grid=circle_grid,
        beta=beta,
        continuous_oog=True,
    )
    setattr(val_stats, "probe_mu_for_plot", val_mu)

    test_stats = compute_osr_stats(
        probe_mu=test_mu,
        probe_kappa=test_kappa,
        probe_labels=test_labels,
        gallery_mu=gallery_mu,
        tau=tau,
        gallery_kappa=gallery_kappa,
        n_grid=circle_grid,
        beta=beta,
        continuous_oog=True,
    )
    setattr(test_stats, "probe_mu_for_plot", test_mu)

    val_error = osr_error_mask(
        pred_idx=val_stats.pred_idx,
        rejected=val_stats.rejected,
        labels=val_stats.labels,
        known_classes=known_classes_oracle,
    )

    test_error = osr_error_mask(
        pred_idx=test_stats.pred_idx,
        rejected=test_stats.rejected,
        labels=test_stats.labels,
        known_classes=known_classes_oracle,
    )

    test_base_metrics = compute_osr_metrics(
        pred_idx=test_stats.pred_idx,
        rejected=test_stats.rejected,
        labels=test_stats.labels,
        known_classes=known_classes_oracle,
    )

    print("\n[Oracle circle] Test OSR metrics:")
    for k, v in test_base_metrics.items():
        print(f"  {k}: {v}")

    # Exact posterior action risk for the current OSR decision.
    val_action_risk = action_risk_from_posterior(val_stats)
    test_action_risk = action_risk_from_posterior(test_stats)

    # Main HolUE mixed identity-information uncertainty.
    val_mixed_kl_unc = -val_stats.kl_total
    test_mixed_kl_unc = -test_stats.kl_total

    # Logistic calibration comparison.
    X_val = np.column_stack(
        [
            val_mixed_kl_unc,
            val_stats.holue_entropy,
            val_stats.gal_entropy,
            np.log(val_stats.scf_kappa + 1e-8),
            np.abs(val_stats.max_sim - tau),
            val_stats.kl_known,
            val_stats.kl_oog,
            val_stats.kl_total,
            val_stats.max_sim,
        ]
    ).astype(np.float64)

    X_test = np.column_stack(
        [
            test_mixed_kl_unc,
            test_stats.holue_entropy,
            test_stats.gal_entropy,
            np.log(test_stats.scf_kappa + 1e-8),
            np.abs(test_stats.max_sim - tau),
            test_stats.kl_known,
            test_stats.kl_oog,
            test_stats.kl_total,
            test_stats.max_sim,
        ]
    ).astype(np.float64)

    y_val = val_error.astype(int)

    if len(np.unique(y_val)) < 2:
        calibrated_unc = test_mixed_kl_unc.copy()
        calibration_info = {
            "used": False,
            "reason": "single_class_validation_error_labels",
        }
    else:
        scaler = StandardScaler()
        X_val_scaled = scaler.fit_transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(
            class_weight=args.calibration_class_weight,
            random_state=args.calibration_random_state,
            max_iter=args.calibration_max_iter,
            solver=args.calibration_solver,
        )
        clf.fit(X_val_scaled, y_val)
        calibrated_unc = clf.predict_proba(X_test_scaled)[:, 1]

        calibration_info = {
            "used": True,
            "reason": "ok",
            "intercept": clf.intercept_.tolist(),
            "coef": clf.coef_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "val_error_rate": float(np.mean(y_val)),
        }

    with open(result_dir / "oracle_circle_calibration_info.json", "w", encoding="utf-8") as f:
        json.dump(calibration_info, f, indent=2)

    rng_eval = np.random.default_rng(seed + 123)
    random_unc = rng_eval.random(len(test_labels))
    oracle_unc = test_error.astype(float) + 1e-6 * rng_eval.random(len(test_error))

    uncertainties = {
        # Exact Bayes risk of the current collapsed OSR action.
        # This is the true no-MLP oracle score for OSR error probability.
        "Bayes action risk": test_action_risk,

        # Paper-style mixed identity-information uncertainty.
        "HolUE mixed KL": test_mixed_kl_unc,

        # Validation-calibrated comparison.
        "LogReg calibrated": calibrated_unc,

        # Diagnostics.
        "Collapsed action entropy": test_stats.holue_entropy,
        "GalUE": test_stats.gal_entropy,
        "SCF": -np.log(test_stats.scf_kappa + 1e-8),
        "AccScr": -np.abs(test_stats.max_sim - tau),
        "MaxSim": -test_stats.max_sim,

        "Random": random_unc,
        "Oracle": oracle_unc,
    }

    np.savez(
        result_dir / "oracle_circle_arrays.npz",
        gallery_mu=gallery_mu,
        tau=tau,
        beta=beta,
        gallery_kappa=gallery_kappa,
        val_mu=val_mu,
        val_kappa=val_kappa,
        val_labels=val_labels,
        val_error=val_error,
        val_low_quality=val_lowq,
        val_holue_posterior=val_stats.holue_posterior,
        val_kl_total=val_stats.kl_total,
        test_mu=test_mu,
        test_kappa=test_kappa,
        test_labels=test_labels,
        test_error=test_error,
        test_low_quality=test_lowq,
        test_holue_posterior=test_stats.holue_posterior,
        test_kl_total=test_stats.kl_total,
        test_action_risk=test_action_risk,
    )

    fractions = np.linspace(
        args.rejection_fraction_start,
        args.rejection_fraction_stop,
        args.rejection_fraction_num,
    )

    def slugify_local(name: str) -> str:
        out = name.lower()
        for ch in [" ", "-", "/", "(", ")", "[", "]", "{", "}", "."]:
            out = out.replace(ch, "_")
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_")

    curves = {}
    for name, unc in uncertainties.items():
        df = rejection_curve(
            uncertainty=unc,
            pred_idx=test_stats.pred_idx,
            rejected=test_stats.rejected,
            labels=test_stats.labels,
            known_classes=known_classes_oracle,
            fractions=fractions,
        )
        curves[name] = df
        df.to_csv(result_dir / f"rejection_curve_{slugify_local(name)}.csv", index=False)

    random_curve = curves["Random"]
    oracle_curve = curves["Oracle"]

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        average_precision_score = None
        roc_auc_score = None

    rows = []
    for name, unc in uncertainties.items():
        df = curves[name]
        prr_f1 = compute_prr(
            curve=df,
            random_curve=random_curve,
            oracle_curve=oracle_curve,
            metric="f1",
        )

        if roc_auc_score is not None and len(np.unique(test_error)) == 2:
            error_auc = float(roc_auc_score(test_error.astype(int), unc))
            error_ap = float(average_precision_score(test_error.astype(int), unc))
        else:
            error_auc = float("nan")
            error_ap = float("nan")

        rows.append(
            {
                "method": name,
                "base_f1": test_base_metrics["f1"],
                "base_fpir": test_base_metrics["fpir"],
                "base_fnir": test_base_metrics["fnir"],
                "base_error_rate": test_base_metrics["error_rate"],
                "prr_f1": prr_f1,
                "error_roc_auc": error_auc,
                "error_average_precision": error_ap,
                "f1_after_50pct_filter": float(df.iloc[-1]["f1"]),
                "fpir_after_50pct_filter": float(df.iloc[-1]["fpir"]),
                "fnir_after_50pct_filter": float(df.iloc[-1]["fnir"]),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(result_dir / "oracle_circle_method_summary.csv", index=False)

    print("\n[Oracle circle] Method summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    method_order = [
        "Bayes action risk",
        "HolUE mixed KL",
        "LogReg calibrated",
        "Collapsed action entropy",
        "GalUE",
        "SCF",
        "AccScr",
        "MaxSim",
        "Random",
        "Oracle",
    ]

    plot_rejection_curves(
        curves=curves,
        out_dir=figure_dir,
        metric=args.plots.rejection_curves.metric,
        plot_cfg=args.plots.rejection_curves,
        method_order=method_order,
    )

    # Simple diagnostic scatter.
    plt.figure(figsize=(7.0, 4.8))
    plt.scatter(
        test_action_risk,
        test_mixed_kl_unc,
        c=test_error.astype(int),
        cmap="coolwarm",
        s=18,
        alpha=0.70,
        linewidths=0,
    )
    plt.xlabel("Bayes action risk")
    plt.ylabel("HolUE mixed-KL uncertainty: -KL")
    plt.title("Oracle circle: action risk vs mixed identity-information uncertainty")
    cb = plt.colorbar()
    cb.set_label("OSR error")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(figure_dir / "oracle_circle_action_risk_vs_mixed_kl.png", dpi=300)
    plt.savefig(figure_dir / "oracle_circle_action_risk_vs_mixed_kl.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    report = f""""""

def main() -> None:
    args = load_toy_config()
    seed_everything(args.seed, deterministic=args.deterministic)

    out_dir = Path(args.out_dir)

    data_dir = resolve_under_out_dir(args.data_dir, out_dir)
    model_dir = resolve_under_out_dir(args.model_dir, out_dir)
    protocol_dir = resolve_under_out_dir(args.protocol_dir, out_dir)
    result_dir = resolve_under_out_dir(args.result_dir_2d, out_dir)
    figure_dir = resolve_under_out_dir(args.figure_dir_2d, out_dir)

    for d in [data_dir, model_dir, protocol_dir, result_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    known_classes = list(args.known_classes)
    unknown_classes = list(args.unknown_classes)
    class_to_idx = {c: i for i, c in enumerate(known_classes)}

    print(f"Output dir:      {out_dir.resolve()}")
    print(f"Device:          {device}")
    print(f"Known classes:   {known_classes}")
    print(f"Unknown classes: {unknown_classes}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    mnist_train = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=args.mnist_download,
        transform=transform,
    )
    mnist_test = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=args.mnist_download,
        transform=transform,
    )

    rng = np.random.default_rng(args.seed)

    train_indices, val_protocol = build_train_val_indices(
        train_ds=mnist_train,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        train_per_known_class=args.train_per_known_class,
        val_gallery_per_class=args.val_gallery_per_class,
        val_probe_known_per_class=args.val_probe_known_per_class,
        val_probe_unknown_per_class=args.val_probe_unknown_per_class,
        corrupt_known_frac=args.corrupt_known_frac,
        corrupt_unknown_frac=args.corrupt_unknown_frac,
        rng=rng,
    )

    test_protocol = build_test_indices(
        test_ds=mnist_test,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        gallery_per_class=args.test_gallery_per_class,
        probe_known_per_class=args.test_probe_known_per_class,
        probe_unknown_per_class=args.test_probe_unknown_per_class,
        corrupt_known_frac=args.corrupt_known_frac,
        corrupt_unknown_frac=args.corrupt_unknown_frac,
        rng=rng,
    )

    np.savez(
        protocol_dir / "mnist_open_set_protocol_indices.npz",
        train_indices=train_indices,
        val_gallery=val_protocol.gallery,
        val_probe_known=val_protocol.probe_known,
        val_probe_unknown=val_protocol.probe_unknown,
        val_corrupt_probe=val_protocol.corrupt_probe,
        test_gallery=test_protocol.gallery,
        test_probe_known=test_protocol.probe_known,
        test_probe_unknown=test_protocol.probe_unknown,
        test_corrupt_probe=test_protocol.corrupt_probe,
        known_classes=np.asarray(known_classes, dtype=int),
        unknown_classes=np.asarray(unknown_classes, dtype=int),
    )
    
    train_ds_arcface = make_known_training_dataset(
        args=args,
        base=mnist_train,
        train_indices=train_indices,
        class_to_idx=class_to_idx,
        stage="arcface",
    )

    train_ds_scf = make_known_training_dataset(
        args=args,
        base=mnist_train,
        train_indices=train_indices,
        class_to_idx=class_to_idx,
        stage="scf",
    )

    if args.plots.corruption_visualization.enabled:
        save_corruption_visualizations(
            args=args,
            mnist_train=mnist_train,
            mnist_test=mnist_test,
            train_indices=train_indices,
            val_protocol=val_protocol,
            test_protocol=test_protocol,
            train_ds_arcface=train_ds_arcface,
            train_ds_scf=train_ds_scf,
            figure_dir=figure_dir,
        )

    # ---------------------------
    # Train / load ArcFace
    # ---------------------------

    arc_ckpt = model_dir / args.arcface_checkpoint_2d
    arc_model = TinyArcFaceMNIST(
    num_classes=len(known_classes),
    embedding_dim=args.embedding_dim_2d,
)

    if arc_ckpt.is_file() and not args.force_train:
        ckpt = torch.load(arc_ckpt, map_location=device, weights_only=False)
        arc_model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded ArcFace checkpoint: {arc_ckpt}")
    else:
        # train_ds = KnownMNISTDataset(
        #     base=mnist_train,
        #     indices=train_indices,
        #     class_to_idx=class_to_idx,
        # )
        train_arcface(
    model=arc_model,
    train_ds=train_ds_arcface,
    device=device,
    epochs=args.arcface_epochs,
    batch_size=args.batch_size,
    lr=args.lr_arcface,
    seed=args.seed,
    arcface_s=args.arcface_s,
    arcface_m=args.arcface_m,
    weight_decay=args.arcface_weight_decay,
)
        torch.save(
            {
                "state_dict": arc_model.state_dict(),
                "known_classes": known_classes,
                "args": to_jsonable(args),
            },
            arc_ckpt,
        )
        print(f"Saved ArcFace checkpoint: {arc_ckpt}")

    # ---------------------------
    # Train / load SCF
    # ---------------------------

    scf_ckpt = model_dir / args.scf_checkpoint_2d
    scf_model = TinySCFMNIST(
    arc_model=arc_model,
    kappa_min=args.scf_kappa_min,
    kappa_max=args.scf_kappa_max,
)

    if scf_ckpt.is_file() and not args.force_train:
        ckpt = torch.load(scf_ckpt, map_location=device, weights_only=False)
        scf_model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded SCF checkpoint: {scf_ckpt}")
    else:
        # train_ds = KnownMNISTDataset(
        #     base=mnist_train,
        #     indices=train_indices,
        #     class_to_idx=class_to_idx,
        # )
        train_scf(
    scf_model=scf_model,
    arc_model=arc_model,
    train_ds=train_ds_scf,
    device=device,
    epochs=args.scf_epochs,
    batch_size=args.batch_size,
    lr=args.lr_scf,
    seed=args.seed,
    weight_decay=args.scf_weight_decay,
    kappa_regularizer=args.scf_kappa_regularizer,
)
        torch.save(
            {
                "state_dict": scf_model.state_dict(),
                "known_classes": known_classes,
                "args": to_jsonable(args),
            },
            scf_ckpt,
        )
        print(f"Saved SCF checkpoint: {scf_ckpt}")

    # ---------------------------
    # Extract validation/test embeddings
    # ---------------------------

    print("Extracting validation gallery embeddings...")
    val_gallery_emb, val_gallery_labels = extract_arc_embeddings(
        arc_model,
        mnist_train,
        val_protocol.gallery,
        device,
        args.batch_size,
        corrupt_indices=[],
        corrupt_seed=args.seed,
        corruption_cfg=args.corruption,
    )
    val_gallery_mu = build_gallery_prototypes(
        val_gallery_emb,
        val_gallery_labels,
        known_classes,
    )

    print("Extracting validation probe embeddings + SCF kappa...")
    val_probe_mu, val_probe_kappa, val_probe_labels = extract_scf_embeddings(
        scf_model,
        mnist_train,
        val_protocol.probe,
        device,
        args.batch_size,
        corrupt_indices=val_protocol.corrupt_probe,
        corrupt_seed=args.seed + 10000,
        corruption_cfg=args.corruption,
    )

    print("Extracting test gallery embeddings...")
    test_gallery_emb, test_gallery_labels = extract_arc_embeddings(
        arc_model,
        mnist_test,
        test_protocol.gallery,
        device,
        args.batch_size,
        corrupt_indices=[],
        corrupt_seed=args.seed,
        corruption_cfg=args.corruption,
    )
    test_gallery_mu = build_gallery_prototypes(
        test_gallery_emb,
        test_gallery_labels,
        known_classes,
    )

    print("Extracting test probe embeddings + SCF kappa...")
    test_probe_mu, test_probe_kappa, test_probe_labels = extract_scf_embeddings(
        scf_model,
        mnist_test,
        test_protocol.probe,
        device,
        args.batch_size,
        corrupt_indices=test_protocol.corrupt_probe,
        corrupt_seed=args.seed + 20000,
        corruption_cfg=args.corruption,
    )

    np.savez(
        result_dir / "mnist_embeddings_for_toy_osr.npz",
        val_gallery_mu=val_gallery_mu,
        val_probe_mu=val_probe_mu,
        val_probe_kappa=val_probe_kappa,
        val_probe_labels=val_probe_labels,
        test_gallery_mu=test_gallery_mu,
        test_probe_mu=test_probe_mu,
        test_probe_kappa=test_probe_kappa,
        test_probe_labels=test_probe_labels,
    )

    # ---------------------------
    # Choose OSR threshold on validation unknown probes
    # ---------------------------

    val_sim = val_probe_mu @ val_gallery_mu.T
    val_max_sim = val_sim.max(axis=1)
    val_seen = np.isin(val_probe_labels, np.asarray(known_classes))
    tau = threshold_at_fpir(val_max_sim[~val_seen], args.target_fpir)

    print(
        f"Validation-selected cosine threshold tau={tau:.6f} for target FPIR={args.target_fpir}"
    )
    effective_beta_2d = args.osr_beta
    if effective_beta_2d is None:
        effective_beta_2d = beta_from_tau_mixed_prior(
            dim=args.embedding_dim_2d,
            tau=tau,
            gallery_kappa=args.gallery_kappa,
            K=len(known_classes),
        )

    print(
        f"Continuous-OOG HolUE: enabled={args.continuous_oog}, "
        f"effective beta={effective_beta_2d:.6g}"
    )
    
    # ---------------------------
    # Compute GalUE/HolUE stats
    # ---------------------------

    print("Computing validation HolUE/GalUE posteriors...")
    val_stats = compute_osr_stats(
        probe_mu=val_probe_mu,
        probe_kappa=val_probe_kappa,
        probe_labels=val_probe_labels,
        gallery_mu=val_gallery_mu,
        tau=tau,
        gallery_kappa=args.gallery_kappa,
        n_grid=args.circle_grid,
        beta=effective_beta_2d,
        continuous_oog=args.continuous_oog,
    )
    setattr(val_stats, "probe_mu_for_plot", val_probe_mu)

    print("Computing test HolUE/GalUE posteriors...")
    test_stats = compute_osr_stats(
        probe_mu=test_probe_mu,
        probe_kappa=test_probe_kappa,
        probe_labels=test_probe_labels,
        gallery_mu=test_gallery_mu,
        tau=tau,
        gallery_kappa=args.gallery_kappa,
        n_grid=args.circle_grid,
        beta=effective_beta_2d,
        continuous_oog=args.continuous_oog,
    )
    setattr(test_stats, "probe_mu_for_plot", test_probe_mu)

    np.savez(
        result_dir / "posteriors_and_osr_stats.npz",
        tau=tau,
        gallery_kappa=args.gallery_kappa,
        known_classes=np.asarray(known_classes, dtype=int),
        unknown_classes=np.asarray(unknown_classes, dtype=int),
        val_labels=val_stats.labels,
        val_max_sim=val_stats.max_sim,
        val_pred_idx=val_stats.pred_idx,
        val_rejected=val_stats.rejected,
        val_scf_kappa=val_stats.scf_kappa,
        val_gal_posterior=val_stats.gal_posterior,
        val_holue_posterior=val_stats.holue_posterior,
        val_gal_entropy=val_stats.gal_entropy,
        val_holue_entropy=val_stats.holue_entropy,
        val_kl_known=val_stats.kl_known,
        val_kl_oog=val_stats.kl_oog,
        val_kl_total=val_stats.kl_total,
        test_labels=test_stats.labels,
        test_max_sim=test_stats.max_sim,
        test_pred_idx=test_stats.pred_idx,
        test_rejected=test_stats.rejected,
        test_scf_kappa=test_stats.scf_kappa,
        test_gal_posterior=test_stats.gal_posterior,
        test_holue_posterior=test_stats.holue_posterior,
        test_gal_entropy=test_stats.gal_entropy,
        test_holue_entropy=test_stats.holue_entropy,
        test_kl_known=test_stats.kl_known,
        test_kl_oog=test_stats.kl_oog,
        test_kl_total=test_stats.kl_total,
        continuous_oog=args.continuous_oog,
        effective_beta=effective_beta_2d,
    )

    # ---------------------------
    # Base OSR quality
    # ---------------------------

    val_error = osr_error_mask(
        pred_idx=val_stats.pred_idx,
        rejected=val_stats.rejected,
        labels=val_stats.labels,
        known_classes=known_classes,
    )
    test_error = osr_error_mask(
        pred_idx=test_stats.pred_idx,
        rejected=test_stats.rejected,
        labels=test_stats.labels,
        known_classes=known_classes,
    )

    val_base_metrics = compute_osr_metrics(
        pred_idx=val_stats.pred_idx,
        rejected=val_stats.rejected,
        labels=val_stats.labels,
        known_classes=known_classes,
    )
    test_base_metrics = compute_osr_metrics(
        pred_idx=test_stats.pred_idx,
        rejected=test_stats.rejected,
        labels=test_stats.labels,
        known_classes=known_classes,
    )

    print("\nValidation OSR metrics:")
    for k, v in val_base_metrics.items():
        print(f"  {k}: {v}")

    print("\nTest OSR metrics:")
    for k, v in test_base_metrics.items():
        print(f"  {k}: {v}")

    with open(result_dir / "base_osr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tau": tau,
                "target_fpir": args.target_fpir,
                "gallery_kappa": args.gallery_kappa,
                "val": val_base_metrics,
                "test": test_base_metrics,
                "continuous_oog": args.continuous_oog,
                "effective_beta": effective_beta_2d,
            },
            f,
            indent=2,
        )

    # ---------------------------
    # HolUE validation calibration
    # ---------------------------

    def make_calibration_features(stats: OSRStats) -> np.ndarray:
        """
        Small feature vector for calibrated HolUE.

        Important:
          This is only for comparison. The dissertation teaser and the main
          no-calibration claim use stats.holue_entropy directly.
        """
        return np.column_stack(
            [
                -stats.kl_total,                    # main mixed-KL uncertainty
                stats.holue_entropy,                # collapsed action entropy diagnostic
                action_risk_from_posterior(stats),  # posterior action-risk diagnostic
                stats.gal_entropy,
                np.log(stats.scf_kappa + 1e-8),
                np.abs(stats.max_sim - tau),
                stats.kl_known,
                stats.kl_oog,
                stats.kl_total,
                stats.max_sim,
            ]
        ).astype(np.float64)

    X_val = make_calibration_features(val_stats)
    y_val = val_error.astype(int)
    X_test = make_calibration_features(test_stats)

    calibration_info: Dict[str, object] = {
        "used": False,
        "reason": "",
        "feature_names": [
            "negative_mixed_kl",
            "holue_collapsed_entropy",
            "holue_action_risk",
            "galue_entropy",
            "log_scf_kappa",
            "abs_maxsim_minus_tau",
            "kl_known",
            "kl_oog",
            "kl_total",
            "max_sim",
        ],
    }

    if (not args.calibration_enabled) or len(np.unique(y_val)) < 2:
        print(
            "[Calibration] Validation set has a single error label. "
            "Falling back to no-calibration HolUE entropy."
        )
        holue_calibrated_unc = test_stats.holue_entropy.copy()
        calibration_info["reason"] = "single_class_validation_error_labels"
    else:
        scaler = StandardScaler()
        X_val_scaled = scaler.fit_transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(
    class_weight=args.calibration_class_weight,
    random_state=args.calibration_random_state,
    max_iter=args.calibration_max_iter,
    solver=args.calibration_solver,
)
        clf.fit(X_val_scaled, y_val)

        # Probability of OSR error. Higher means more uncertain.
        holue_calibrated_unc = clf.predict_proba(X_test_scaled)[:, 1]

        calibration_info.update(
            {
                "used": True,
                "reason": "ok",
                "intercept": clf.intercept_.tolist(),
                "coef": clf.coef_.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "val_error_rate": float(np.mean(y_val)),
            }
        )

    with open(result_dir / "holue_calibration_info.json", "w", encoding="utf-8") as f:
        json.dump(calibration_info, f, indent=2)

    # ---------------------------
    # Uncertainty methods
    # ---------------------------

    rng_eval = np.random.default_rng(args.seed + 12345)
    random_unc = rng_eval.random(len(test_stats.labels))

    # Oracle: all erroneous predictions first, ties random.
    oracle_unc = test_error.astype(float) + 1e-6 * rng_eval.random(len(test_error))

    holue_mixed_kl_unc = -test_stats.kl_total
    holue_action_risk = action_risk_from_posterior(test_stats)

    uncertainties: Dict[str, np.ndarray] = {
        "HolUE no calibration": holue_mixed_kl_unc,
        "HolUE calibrated": holue_calibrated_unc,
        "HolUE raw KL": holue_mixed_kl_unc,
        "HolUE collapsed entropy": test_stats.holue_entropy,
        "HolUE action risk": holue_action_risk,
        "GalUE": test_stats.gal_entropy,
        "SCF": -np.log(test_stats.scf_kappa + 1e-8),
        "AccScr": -np.abs(test_stats.max_sim - tau),
        "MaxSim": -test_stats.max_sim,
        "Random": random_unc,
        "Oracle": oracle_unc,
    }

    np.savez(
        result_dir / "uncertainty_scores.npz",
        **{k.replace(" ", "_").replace("-", "_"): v for k, v in uncertainties.items()},
    )

    # ---------------------------
    # Rejection curves and PRR
    # ---------------------------

    fractions = np.linspace(
    args.rejection_fraction_start,
    args.rejection_fraction_stop,
    args.rejection_fraction_num,
)

    def slugify(name: str) -> str:
        out = name.lower()
        for ch in [" ", "-", "/", "(", ")", "[", "]", "{", "}", "."]:
            out = out.replace(ch, "_")
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_")

    curves: Dict[str, pd.DataFrame] = {}
    for name, unc in uncertainties.items():
        df = rejection_curve(
            uncertainty=unc,
            pred_idx=test_stats.pred_idx,
            rejected=test_stats.rejected,
            labels=test_stats.labels,
            known_classes=known_classes,
            fractions=fractions,
        )
        curves[name] = df
        df.to_csv(result_dir / f"rejection_curve_{slugify(name)}.csv", index=False)

    random_curve = curves["Random"]
    oracle_curve = curves["Oracle"]

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        average_precision_score = None
        roc_auc_score = None

    rows = []
    for name, unc in uncertainties.items():
        df = curves[name]
        prr_f1 = compute_prr(
            curve=df,
            random_curve=random_curve,
            oracle_curve=oracle_curve,
            metric="f1",
        )

        if roc_auc_score is not None and len(np.unique(test_error)) == 2:
            error_auc = float(roc_auc_score(test_error.astype(int), unc))
            error_ap = float(average_precision_score(test_error.astype(int), unc))
        else:
            error_auc = float("nan")
            error_ap = float("nan")

        rows.append(
            {
                "method": name,
                "base_f1": test_base_metrics["f1"],
                "base_fpir": test_base_metrics["fpir"],
                "base_fnir": test_base_metrics["fnir"],
                "base_error_rate": test_base_metrics["error_rate"],
                "prr_f1": prr_f1,
                "error_roc_auc": error_auc,
                "error_average_precision": error_ap,
                "f1_after_50pct_filter": float(df.iloc[-1]["f1"]),
                "fpir_after_50pct_filter": float(df.iloc[-1]["fpir"]),
                "fnir_after_50pct_filter": float(df.iloc[-1]["fnir"]),
            }
        )

    summary_df = pd.DataFrame(rows)

    method_order = [
        "HolUE no calibration",
        "HolUE calibrated",
        "HolUE raw KL",
        "GalUE",
        "SCF",
        "AccScr",
        "MaxSim",
        "Random",
        "Oracle",
    ]
    summary_df["method"] = pd.Categorical(
        summary_df["method"],
        categories=method_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values("method").reset_index(drop=True)

    summary_df.to_csv(result_dir / "method_summary.csv", index=False)

    print("\nMethod summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------
    # Figures
    # ---------------------------

    print("Saving rejection-curve figure...")
    plot_rejection_curves(
    curves,
    figure_dir,
    metric=args.plots.rejection_curves.metric,
    plot_cfg=args.plots.rejection_curves,
    method_order=args.methods.order,
)

    print("Saving teaser circle figure...")
    if args.plots.teaser_circle_2d.enabled:
        plot_teaser_circle(
            stats=test_stats,
            gallery_mu=test_gallery_mu,
            known_classes=known_classes,
            tau=tau,
            gallery_kappa=args.gallery_kappa,
            uncertainty=test_stats.holue_entropy,
            out_dir=figure_dir,
            seed=args.seed,
            max_points=args.plots.teaser_circle_2d.max_points,
        )
    print("Saving stylized Bayesian circle teaser...")
    if args.plots.stylized_bayesian_circle.enabled:
        plot_stylized_bayesian_circle_teaser(
            gallery_mu=test_gallery_mu,
            known_classes=known_classes,
            out_dir=figure_dir,
            kappa=args.plots.stylized_bayesian_circle.kappa,
            beta=args.plots.stylized_bayesian_circle.beta,
            unc_type=args.plots.stylized_bayesian_circle.unc_type,
            draw_oog=args.plots.stylized_bayesian_circle.draw_oog,
        )

    # Additional compact diagnostic scatter: error vs uncertainty.
    scatter_cfg = args.plots.holue_vs_scf_scatter
    plt.figure(figsize=tuple(scatter_cfg.figsize))
    plot_df = pd.DataFrame(
        {
            "HolUE no calibration": test_stats.holue_entropy,
            "SCF uncertainty": -np.log(test_stats.scf_kappa + 1e-8),
            "is_error": test_error.astype(int),
        }
    )
    plt.scatter(
        plot_df["SCF uncertainty"],
        plot_df["HolUE no calibration"],
        c=plot_df["is_error"],
        cmap="coolwarm",
        s=20,
        alpha=0.75,
        linewidths=0,
    )
    plt.title(scatter_cfg.title)
    plt.xlabel(scatter_cfg.xlabel)
    plt.ylabel(scatter_cfg.ylabel)
    cb = plt.colorbar()
    cb.set_label("OSR error")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(figure_dir / scatter_cfg.save_png, dpi=scatter_cfg.dpi)
    plt.savefig(figure_dir / scatter_cfg.save_pdf, dpi=scatter_cfg.dpi, bbox_inches="tight")

    plt.close()

    # Save a tiny markdown report.
    report = f"""# Toy MNIST HolUE OSR experiment

This directory was generated by `toy_mnist_holue.py`.

## Protocol

- Known MNIST classes: `{known_classes}`
- Unknown MNIST classes: `{unknown_classes}`
- OSR threshold selected on validation unknown probes.
- Target validation FPIR: `{args.target_fpir}`
- Selected cosine threshold tau: `{tau:.6f}`
- Gallery posterior sharpness kappa: `{args.gallery_kappa}`

## Test OSR metrics before filtering

```json
{json.dumps(test_base_metrics, indent=2)}
```

## Main files

- `models/tiny_arcface_mnist_2d.pt`
- `models/tiny_scf_mnist_2d.pt`
- `protocols/mnist_open_set_protocol_indices.npz`
- `results/method_summary.csv`
- `results/rejection_curve_*.csv`
- `figures/teaser_holue_mnist_circle.pdf`
- `figures/rejection_curves.pdf`

## Intended dissertation message

The method `HolUE no calibration` uses no validation-set calibration for
uncertainty. It is simply the entropy of the predictive posterior

`p(c|x) = ∫ p(c|z) p(z|x) dz`

computed exactly by deterministic quadrature on the 2D unit circle. The
validation-calibrated variant is included only as an extra comparison.
"""

    with open(out_dir / "README_toy_results.md", "w", encoding="utf-8") as f:
        f.write(report)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(args), f, indent=2)
    run_oracle_circle_experiment(args, out_dir)
    if not args.skip_3d:
        run_3d_extension(
            args=args,
            known_classes=known_classes,
            unknown_classes=unknown_classes,
            class_to_idx=class_to_idx,
            mnist_train=mnist_train,
            mnist_test=mnist_test,
            train_indices=train_indices,
            val_protocol=val_protocol,
            test_protocol=test_protocol,
            device=device,
            out_dir=out_dir,
        )
    print("\nDone.")
    print(f"All outputs saved to: {out_dir.resolve()}")
    print(f"Main summary:         {(result_dir / 'method_summary.csv').resolve()}")
    print(
        f"Teaser figure:        {(figure_dir / 'teaser_holue_mnist_circle.pdf').resolve()}"
    )


if __name__ == "__main__":
    main()
