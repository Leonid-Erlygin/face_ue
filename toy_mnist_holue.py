#!/usr/bin/env python3
"""
Clean toy MNIST HolUE open-set recognition example.

This script demonstrates HolUE in a small controlled MNIST OSR setting.

Core idea
---------
Known classes are digits 0..4. Unknown/OOD classes are digits 5..9.

A CNN is trained with an ArcFace-like loss to produce normalized embeddings
on S^{d-1}, where d=2 or d=3. A small SCF-like head predicts a vMF
concentration kappa(x), giving

    p(z | x) = vMF(z; mu(x), kappa(x)).

For the gallery-aware posterior we use the paper-style mixed identity prior:

    c in {1,...,K} union S^{d-1}.

Known identities:
    p(c=i) = (1-beta)/K.

OOD identities:
    p(c=psi) = beta / S_{d-1}, psi in S^{d-1}.

Class conditional likelihoods:
    p(z | c=i) = vMF(z; m_i, kappa_g).

OOD identity conditional:
    p(z | c=psi) = delta(z - psi).

Therefore

    p(z) = sum_i ((1-beta)/K) p(z|c=i) + beta/S_{d-1}.

HolUE computes

    p(c | x) = int p(c | z) p(z | x) dz

by deterministic quadrature on S^1 or S^2.

The no-calibration HolUE uncertainty score is

    u_HolUE(x) = -D_KL(p(c|x) || p(c)),

where the KL is computed over the full mixed discrete-continuous identity
space. This continuous OOD KL term is essential: collapsing all OOD identities
into one reject class discards the SCF uncertainty over OOD identity location.

Outputs are saved under `out_dir` from the YAML config.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML with: pip install pyyaml") from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gamma, ive, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Reproducibility and config helpers
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


def cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_jsonable(obj):
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
            f"Pass a config path or create toy_mnist_holue_config.yaml."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    return data


def flatten_config(raw: dict) -> SimpleNamespace:
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

    checkpoints = output.get("checkpoints", {})

    train_corr = training.get("train_corruption", {})
    arc_corr = train_corr.get("arcface", {})
    scf_corr = train_corr.get("scf", {})

    beta_raw = osr.get("beta", "auto")
    if beta_raw is None or str(beta_raw).lower() == "auto":
        osr_beta = None
    else:
        osr_beta = float(beta_raw)

    args = SimpleNamespace(
        raw_config=raw,
        seed=int(runtime.get("seed", 777)),
        deterministic=bool(runtime.get("deterministic", True)),
        device=str(runtime.get("device", "auto")),
        out_dir=str(output["out_dir"]),
        data_dir=str(output.get("data_dir", "mnist_data")),
        model_dir=str(output.get("model_dir", "models")),
        protocol_dir=str(output.get("protocol_dir", "protocols")),
        result_dir={
            2: str(output.get("result_dir_2d", "results")),
            3: str(output.get("result_dir_3d", "results_3d")),
        },
        figure_dir={
            2: str(output.get("figure_dir_2d", "figures")),
            3: str(output.get("figure_dir_3d", "figures_3d")),
        },
        arcface_checkpoint={
            2: checkpoints.get("arcface_2d", "tiny_arcface_mnist_2d.pt"),
            3: checkpoints.get("arcface_3d", "tiny_arcface_mnist_3d.pt"),
        },
        scf_checkpoint={
            2: checkpoints.get("scf_2d", "tiny_scf_mnist_2d.pt"),
            3: checkpoints.get("scf_3d", "tiny_scf_mnist_3d.pt"),
        },
        mnist_download=bool(data.get("mnist_download", True)),
        known_classes=[int(x) for x in data["known_classes"]],
        unknown_classes=[int(x) for x in data["unknown_classes"]],
        train_per_known_class=int(protocol["train_per_known_class"]),
        val_gallery_per_class=int(protocol["val_gallery_per_class"]),
        val_probe_known_per_class=int(protocol["val_probe_known_per_class"]),
        val_probe_unknown_per_class=int(protocol["val_probe_unknown_per_class"]),
        test_gallery_per_class=int(protocol["test_gallery_per_class"]),
        test_probe_known_per_class=int(protocol["test_probe_known_per_class"]),
        test_probe_unknown_per_class=int(protocol["test_probe_unknown_per_class"]),
        corrupt_known_frac=float(protocol["corrupt_known_frac"]),
        corrupt_unknown_frac=float(protocol["corrupt_unknown_frac"]),
        embedding_dim={
            2: int(model.get("embedding_dim_2d", 2)),
            3: int(model.get("embedding_dim_3d", 3)),
        },
        scf_kappa_min=float(model["scf"].get("kappa_min", 1.0)),
        scf_kappa_max=float(model["scf"].get("kappa_max", 80.0)),
        force_train=bool(training.get("force_train", False)),
        batch_size=int(training.get("batch_size", 256)),
        arcface_epochs={
            2: int(training["arcface"].get("epochs_2d", 30)),
            3: int(training["arcface"].get("epochs_3d", 30)),
        },
        arcface_lr=float(training["arcface"].get("lr", 1e-3)),
        arcface_scale=float(training["arcface"].get("scale", 16.0)),
        arcface_margin=float(training["arcface"].get("margin", 0.30)),
        arcface_weight_decay=float(training["arcface"].get("weight_decay", 1e-4)),
        scf_epochs={
            2: int(training["scf"].get("epochs_2d", 10)),
            3: int(training["scf"].get("epochs_3d", 10)),
        },
        scf_lr=float(training["scf"].get("lr", 2e-3)),
        scf_weight_decay=float(training["scf"].get("weight_decay", 1e-4)),
        scf_kappa_regularizer=float(training["scf"].get("kappa_regularizer", 1e-4)),
        train_corrupt_arcface_enabled=bool(arc_corr.get("enabled", False)),
        train_corrupt_arcface_probability=float(arc_corr.get("probability", 0.0)),
        train_corrupt_arcface_seed_shift=int(arc_corr.get("seed_shift", 30000)),
        train_corrupt_scf_enabled=bool(scf_corr.get("enabled", False)),
        train_corrupt_scf_probability=float(scf_corr.get("probability", 0.0)),
        train_corrupt_scf_seed_shift=int(scf_corr.get("seed_shift", 40000)),
        target_fpir=float(osr.get("target_fpir", 0.1)),
        gallery_kappa=float(osr.get("gallery_kappa", 25.0)),
        osr_beta=osr_beta,
        circle_grid=int(integration.get("circle_grid", 720)),
        sphere_theta_grid=int(integration.get("sphere_theta_grid", 96)),
        sphere_phi_grid=int(integration.get("sphere_phi_grid", 48)),
        calibration_enabled=bool(calibration.get("enabled", True)),
        calibration_solver=str(calibration.get("solver", "lbfgs")),
        calibration_max_iter=int(calibration.get("max_iter", 2000)),
        calibration_class_weight=calibration.get("class_weight", "balanced"),
        calibration_random_state=int(
            calibration.get("random_state", runtime.get("seed", 777))
        ),
        rejection_fraction_start=float(
            evaluation["rejection_fractions"].get("start", 0.0)
        ),
        rejection_fraction_stop=float(
            evaluation["rejection_fractions"].get("stop", 0.5)
        ),
        rejection_fraction_num=int(evaluation["rejection_fractions"].get("num", 21)),
        corruption=raw.get("corruption", {}),
        plots=raw.get("plots", {}),
        methods=raw.get("methods", {}),
        three_d=raw.get("three_d", {}),
        controlled_circle=raw.get("controlled_circle", raw.get("oracle_circle", {})),
    )

    args.method_order = list(
        cfg_get(
            args.methods,
            "order",
            [
                "HolUE no calibration",
                "HolUE calibrated",
                "HolUE collapsed entropy",
                "GalUE",
                "SCF",
                "AccScr",
                "MaxSim",
                "Random",
                "Oracle",
            ],
        )
    )

    return args


def load_toy_config() -> SimpleNamespace:
    if len(sys.argv) > 2:
        raise ValueError("Pass at most one positional YAML config path.")

    if len(sys.argv) == 2:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(
            os.environ.get("TOY_MNIST_HOLUE_CONFIG", "toy_mnist_holue_config.yaml")
        )

    raw = load_yaml_file(config_path)
    args = flatten_config(raw)
    args.config_path = str(config_path)
    print(f"Loaded config: {config_path.resolve()}")
    return args


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


class ArcFaceLoss(nn.Module):
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
    def __init__(self, num_classes: int = 5, embedding_dim: int = 2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.PReLU(),
        )

        self.embed = nn.Linear(128, self.embedding_dim)
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.embedding_dim))
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
    Negative log-likelihood of target class center under vMF(mu, kappa).

    Supports S^1 and S^2, i.e. embedding_dim 2 or 3.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim not in [2, 3]:
            raise ValueError("Only embedding_dim=2 or embedding_dim=3 are supported.")
        self.embedding_dim = int(embedding_dim)

    @staticmethod
    def log_i0(kappa: torch.Tensor) -> torch.Tensor:
        if hasattr(torch.special, "i0e"):
            return torch.log(torch.special.i0e(kappa).clamp_min(1e-12)) + kappa
        return torch.log(torch.i0(kappa).clamp_min(1e-12))

    @staticmethod
    def log_sinh_stable(kappa: torch.Tensor) -> torch.Tensor:
        small = kappa < 1e-4
        out = torch.empty_like(kappa)

        if small.any():
            out[small] = torch.log(torch.sinh(kappa[small]).clamp_min(1e-12))

        if (~small).any():
            ks = kappa[~small]
            out[~small] = ks - math.log(2.0) + torch.log1p(-torch.exp(-2.0 * ks))

        return out

    def forward(
        self, mu: torch.Tensor, kappa: torch.Tensor, class_center: torch.Tensor
    ) -> torch.Tensor:
        cos = torch.sum(mu * class_center, dim=1, keepdim=True).clamp(-1.0, 1.0)
        kappa = kappa.clamp_min(1e-8)

        if self.embedding_dim == 2:
            # C_2(kappa) = 1/(2*pi*I_0(kappa))
            nll = -kappa * cos + math.log(2.0 * math.pi) + self.log_i0(kappa)
        else:
            # C_3(kappa) = kappa/(4*pi*sinh(kappa))
            nll = (
                -kappa * cos
                + math.log(4.0 * math.pi)
                + self.log_sinh_stable(kappa)
                - torch.log(kappa)
            )

        return nll.mean()


class TinySCFMNIST(nn.Module):
    """
    Frozen ArcFace backbone plus small concentration head.
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
# Dataset protocol and corruption
# ---------------------------------------------------------------------


def get_targets(ds: datasets.MNIST) -> np.ndarray:
    targets = ds.targets
    if torch.is_tensor(targets):
        return targets.cpu().numpy().astype(int)
    return np.asarray(targets, dtype=int)


def corrupt_tensor_mnist(
    img: torch.Tensor, seed: int, corruption_cfg=None
) -> torch.Tensor:
    rng = np.random.default_rng(seed)

    x = img.clone()
    x = x * 0.5 + 0.5
    x = x.clamp(0.0, 1.0)

    enabled_modes = cfg_get(corruption_cfg, "enabled_modes", ["gaussian_noise"])
    enabled_modes = list(enabled_modes)
    if len(enabled_modes) == 0:
        return img

    mode = str(rng.choice(enabled_modes))

    if mode == "occlusion":
        ocfg = cfg_get(corruption_cfg, "occlusion", {})
        h = int(
            rng.integers(
                int(cfg_get(ocfg, "h_min", 8)), int(cfg_get(ocfg, "h_max", 15)) + 1
            )
        )
        w = int(
            rng.integers(
                int(cfg_get(ocfg, "w_min", 8)), int(cfg_get(ocfg, "w_max", 15)) + 1
            )
        )
        y = int(rng.integers(0, 28 - h + 1))
        z = int(rng.integers(0, 28 - w + 1))
        fill = float(cfg_get(ocfg, "fill_value", 0.0))
        x[:, y : y + h, z : z + w] = fill

    elif mode == "gaussian_noise":
        ncfg = cfg_get(corruption_cfg, "gaussian_noise", {})
        std = float(cfg_get(ncfg, "std", 0.35))
        noise = torch.tensor(
            rng.normal(0.0, std, size=tuple(x.shape)),
            dtype=x.dtype,
            device=x.device,
        )
        x = (x + noise).clamp(0.0, 1.0)

    elif mode == "translation":
        tcfg = cfg_get(corruption_cfg, "translation", {})
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
        ecfg = cfg_get(corruption_cfg, "erase_noise", {})
        h = int(
            rng.integers(
                int(cfg_get(ecfg, "h_min", 6)), int(cfg_get(ecfg, "h_max", 12)) + 1
            )
        )
        w = int(
            rng.integers(
                int(cfg_get(ecfg, "w_min", 6)), int(cfg_get(ecfg, "w_max", 12)) + 1
            )
        )
        y = int(rng.integers(0, 28 - h + 1))
        z = int(rng.integers(0, 28 - w + 1))
        noise_std = float(cfg_get(ecfg, "noise_std", 0.20))

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


class KnownMNISTDataset(Dataset):
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
                f"corruption_prob must be in [0,1], got {self.corruption_prob}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _should_corrupt(self, orig_idx: int) -> bool:
        if self.corruption_prob <= 0.0:
            return False
        if self.corruption_prob >= 1.0:
            return True
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

        mapped = self.class_to_idx[int(label)]
        return img, torch.tensor(mapped, dtype=torch.long)


class IndexedMNISTView(Dataset):
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
        self.corrupt_set = (
            set()
            if corrupt_indices is None
            else set(map(int, np.asarray(list(corrupt_indices)).reshape(-1)))
        )
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
        pool = rng.permutation(np.where(targets == c)[0])
        need = train_per_known_class + val_gallery_per_class + val_probe_known_per_class
        if len(pool) < need:
            raise ValueError(f"Not enough train samples for class {c}: need {need}")
        train_indices.extend(pool[:train_per_known_class])
        s = train_per_known_class
        val_gallery.extend(pool[s : s + val_gallery_per_class])
        s += val_gallery_per_class
        val_probe_known.extend(pool[s : s + val_probe_known_per_class])

    for c in unknown_classes:
        pool = rng.permutation(np.where(targets == c)[0])
        if len(pool) < val_probe_unknown_per_class:
            raise ValueError(f"Not enough train unknown samples for class {c}")
        val_probe_unknown.extend(pool[:val_probe_unknown_per_class])

    val_probe_known = np.asarray(val_probe_known, dtype=int)
    val_probe_unknown = np.asarray(val_probe_unknown, dtype=int)

    corrupt = []
    nk = int(round(corrupt_known_frac * len(val_probe_known)))
    nu = int(round(corrupt_unknown_frac * len(val_probe_unknown)))

    if nk > 0:
        corrupt.extend(rng.choice(val_probe_known, size=nk, replace=False))
    if nu > 0:
        corrupt.extend(rng.choice(val_probe_unknown, size=nu, replace=False))

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
        pool = rng.permutation(np.where(targets == c)[0])
        need = gallery_per_class + probe_known_per_class
        if len(pool) < need:
            raise ValueError(f"Not enough test samples for class {c}: need {need}")
        gallery.extend(pool[:gallery_per_class])
        probe_known.extend(
            pool[gallery_per_class : gallery_per_class + probe_known_per_class]
        )

    for c in unknown_classes:
        pool = rng.permutation(np.where(targets == c)[0])
        if len(pool) < probe_unknown_per_class:
            raise ValueError(f"Not enough test unknown samples for class {c}")
        probe_unknown.extend(pool[:probe_unknown_per_class])

    probe_known = np.asarray(probe_known, dtype=int)
    probe_unknown = np.asarray(probe_unknown, dtype=int)

    corrupt = []
    nk = int(round(corrupt_known_frac * len(probe_known)))
    nu = int(round(corrupt_unknown_frac * len(probe_unknown)))

    if nk > 0:
        corrupt.extend(rng.choice(probe_known, size=nk, replace=False))
    if nu > 0:
        corrupt.extend(rng.choice(probe_unknown, size=nu, replace=False))

    return ProtocolIndices(
        gallery=np.asarray(gallery, dtype=int),
        probe_known=probe_known,
        probe_unknown=probe_unknown,
        corrupt_probe=np.asarray(corrupt, dtype=int),
    )


def make_known_training_dataset(
    args: SimpleNamespace,
    base: datasets.MNIST,
    train_indices: np.ndarray,
    class_to_idx: Dict[int, int],
    stage: str,
) -> KnownMNISTDataset:
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

    print(f"[Train dataset] stage={stage}, corruption_enabled={enabled}, prob={prob}")

    return KnownMNISTDataset(
        base=base,
        indices=train_indices,
        class_to_idx=class_to_idx,
        corruption_cfg=args.corruption,
        corruption_prob=prob,
        corrupt_seed=seed,
    )


# ---------------------------------------------------------------------
# Training and feature extraction
# ---------------------------------------------------------------------


def train_arcface(
    model: TinyArcFaceMNIST,
    train_ds: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    arcface_s: float,
    arcface_m: float,
    weight_decay: float,
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
        losses, accs = [], []

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = (logits.argmax(dim=1) == labels).float().mean().item()

            losses.append(float(loss.item()))
            accs.append(float(acc))

        print(
            f"[ArcFace] epoch {epoch:03d}/{epochs:03d} loss={np.mean(losses):.4f} acc={np.mean(accs):.4f}"
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
    weight_decay: float,
    kappa_regularizer: float,
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
        scf_model.head.parameters(), lr=lr, weight_decay=weight_decay
    )

    class_centers = arc_model.class_centers().to(device)

    for epoch in range(1, epochs + 1):
        losses, kappas, cosines = [], [], []

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            emb, log_kappa = scf_model(imgs)
            kappa = torch.exp(log_kappa)
            wc = class_centers[labels]

            loss = criterion(emb, kappa, wc)
            loss = loss + kappa_regularizer * kappa.mean()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                cos = torch.sum(emb * wc, dim=1).mean().item()

            losses.append(float(loss.item()))
            kappas.append(float(kappa.mean().item()))
            cosines.append(float(cos))

        print(
            f"[SCF]     epoch {epoch:03d}/{epochs:03d} "
            f"loss={np.mean(losses):.4f} kappa={np.mean(kappas):.2f} cos={np.mean(cosines):.3f}"
        )


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

    embs, labels = [], []

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

    embs, kappas, labels = [], [], []

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


# ---------------------------------------------------------------------
# Geometry, mixed-prior GalUE, and HolUE
# ---------------------------------------------------------------------


def normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def build_gallery_prototypes(
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    known_classes: List[int],
) -> np.ndarray:
    gallery_embs = normalize_np(gallery_embs)
    protos = []

    for c in known_classes:
        mask = gallery_labels == c
        if not np.any(mask):
            raise ValueError(f"No gallery samples for known class {c}")
        proto = gallery_embs[mask].mean(axis=0, keepdims=True)
        proto = normalize_np(proto)[0]
        protos.append(proto)

    return np.asarray(protos, dtype=np.float64)


def sphere_area_np(dim: int) -> float:
    return float(2.0 * math.pi ** (dim / 2.0) / gamma(dim / 2.0))


def log_vmf_normalizer_np(kappa, dim: int):
    """
    log C_d(kappa) for vMF on S^{dim-1}.

    Supports dim=2 and dim=3.
    """
    scalar_input = np.isscalar(kappa)
    k = np.atleast_1d(np.asarray(kappa, dtype=np.float64))
    k = np.maximum(k, 1e-12)

    if dim == 2:
        # C_2(kappa) = 1/(2*pi*I0(kappa)).
        log_i0 = np.log(np.maximum(ive(0, k), 1e-300)) + k
        out = -math.log(2.0 * math.pi) - log_i0

    elif dim == 3:
        # C_3(kappa)=kappa/(4*pi*sinh(kappa)).
        out = np.empty_like(k)
        small = k < 1e-8
        out[small] = -math.log(4.0 * math.pi)

        ks = k[~small]
        if len(ks) > 0:
            log_sinh = ks - math.log(2.0) + np.log1p(-np.exp(-2.0 * ks))
            out[~small] = np.log(ks) - math.log(4.0 * math.pi) - log_sinh

    else:
        raise ValueError(f"Only dim=2 and dim=3 are supported, got dim={dim}")

    if scalar_input:
        return float(out[0])
    return out.reshape(np.shape(kappa))


def beta_from_tau_mixed_prior(
    dim: int, tau: float, gallery_kappa: float, K: int
) -> float:
    """
    Derive beta so that Bayesian known-vs-OOD boundary matches cosine threshold tau.

    Boundary:
        ((1-beta)/K) C_d(kappa) exp(kappa*tau) = beta / S.

    Hence:
        beta/(1-beta) = S C_d(kappa) exp(kappa*tau) / K.
    """
    S = sphere_area_np(dim)
    log_C = log_vmf_normalizer_np(gallery_kappa, dim=dim)

    log_A = math.log(S) + log_C + float(gallery_kappa) * float(tau) - math.log(K)

    if log_A >= 0:
        beta = 1.0 / (1.0 + math.exp(-log_A))
    else:
        A = math.exp(log_A)
        beta = A / (1.0 + A)

    return float(np.clip(beta, 1e-8, 1.0 - 1e-8))


def tau_from_beta_mixed_prior(
    dim: int, beta: float, gallery_kappa: float, K: int
) -> float:
    beta = float(np.clip(beta, 1e-8, 1.0 - 1e-8))
    S = sphere_area_np(dim)
    log_C = log_vmf_normalizer_np(gallery_kappa, dim=dim)

    tau = (
        math.log(beta) - math.log1p(-beta) + math.log(K) - math.log(S) - log_C
    ) / float(gallery_kappa)

    return float(tau)


def circle_quadrature_grid(n_grid: int) -> Tuple[np.ndarray, np.ndarray]:
    n_grid = int(n_grid)
    angles = np.linspace(0.0, 2.0 * math.pi, n_grid, endpoint=False)
    grid_z = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)
    log_dS = np.full(n_grid, math.log(2.0 * math.pi / n_grid), dtype=np.float64)
    return grid_z, log_dS


def sphere_quadrature_grid(n_theta: int, n_phi: int) -> Tuple[np.ndarray, np.ndarray]:
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


def make_quadrature_grid(
    dim: int, circle_grid: int, sphere_theta_grid: int, sphere_phi_grid: int
):
    if dim == 2:
        return circle_quadrature_grid(circle_grid)
    if dim == 3:
        return sphere_quadrature_grid(sphere_theta_grid, sphere_phi_grid)
    raise ValueError(f"Only dim=2 and dim=3 are supported, got dim={dim}")


def posterior_from_z_mixed_prior(
    z: np.ndarray,
    gallery_mu: np.ndarray,
    gallery_kappa: float,
    beta: float,
) -> np.ndarray:
    """
    Collapsed action posterior for deterministic z:

        [P(c=1|z), ..., P(c=K|z), P(OOD|z)]

    The OOD term is induced by a continuous uniform identity prior.
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

    known_log = log_prior_known + log_Cg + float(gallery_kappa) * (z @ gallery_mu.T)
    oog_log = np.full((z.shape[0], 1), log_prior_oog_density, dtype=np.float64)

    logits = np.concatenate([known_log, oog_log], axis=1)
    log_norm = logsumexp(logits, axis=1, keepdims=True)

    return np.exp(logits - log_norm)


def mixed_holue_posterior_quadrature(
    mu: np.ndarray,
    kappa: np.ndarray,
    gallery_mu: np.ndarray,
    gallery_kappa: float,
    beta: float,
    circle_grid: int = 720,
    sphere_theta_grid: int = 96,
    sphere_phi_grid: int = 48,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Paper-style HolUE with continuous OOD prior.

    Returns:
        action_post:
            collapsed action posterior [known class masses, total OOD mass].
            This is for diagnostics and entropy only.

        kl_known:
            sum_i P_i log(P_i / ((1-beta)/K))

        kl_oog:
            int rho(psi|x) log(rho(psi|x)/(beta/S)) dpsi

        kl_total:
            full mixed KL.
    """
    mu = normalize_np(np.asarray(mu, dtype=np.float64))
    kappa = np.asarray(kappa, dtype=np.float64).reshape(-1)
    kappa = np.maximum(kappa, 1e-12)
    gallery_mu = normalize_np(np.asarray(gallery_mu, dtype=np.float64))

    dim = mu.shape[1]
    K = gallery_mu.shape[0]
    S = sphere_area_np(dim)

    beta = float(np.clip(beta, 1e-8, 1.0 - 1e-8))

    grid_z, log_dS = make_quadrature_grid(
        dim=dim,
        circle_grid=circle_grid,
        sphere_theta_grid=sphere_theta_grid,
        sphere_phi_grid=sphere_phi_grid,
    )
    grid_z = normalize_np(grid_z)
    log_dS = np.asarray(log_dS, dtype=np.float64).reshape(-1)

    log_prior_known = math.log1p(-beta) - math.log(K)
    log_prior_oog_density = math.log(beta) - math.log(S)
    log_Cg = log_vmf_normalizer_np(gallery_kappa, dim=dim)

    cos_grid_gallery = grid_z @ gallery_mu.T
    log_known_joint = log_prior_known + log_Cg + float(gallery_kappa) * cos_grid_gallery
    log_oog_joint = np.full(
        (grid_z.shape[0], 1), log_prior_oog_density, dtype=np.float64
    )

    log_p_z = logsumexp(
        np.concatenate([log_known_joint, log_oog_joint], axis=1),
        axis=1,
    )

    action_posts = []
    kl_known_all = []
    kl_oog_all = []
    kl_total_all = []

    for start in range(0, len(mu), batch_size):
        end = min(start + batch_size, len(mu))

        mu_b = mu[start:end]
        kappa_b = kappa[start:end]
        B = len(mu_b)

        cos_to_mu = mu_b @ grid_z.T
        log_Cx = log_vmf_normalizer_np(kappa_b, dim=dim).reshape(B, 1)
        log_f_x = log_Cx + kappa_b[:, None] * cos_to_mu

        # Known posterior masses.
        log_integrand_known = (
            log_f_x[:, :, None]
            + log_known_joint[None, :, :]
            - log_p_z[None, :, None]
            + log_dS[None, :, None]
        )
        P_known = np.exp(logsumexp(log_integrand_known, axis=1))

        # Continuous OOD posterior density rho(psi|x).
        log_rho = log_prior_oog_density + log_f_x - log_p_z[None, :]
        rho_dS = np.exp(log_rho + log_dS[None, :])
        P_oog = np.sum(rho_dS, axis=1)

        # Remove tiny quadrature mass error.
        total_mass = np.maximum(P_known.sum(axis=1) + P_oog, 1e-300)
        log_total_mass = np.log(total_mass)

        P_known_n = P_known / total_mass[:, None]
        P_oog_n = P_oog / total_mass
        rho_dS_n = rho_dS / total_mass[:, None]

        action_post = np.concatenate([P_known_n, P_oog_n[:, None]], axis=1)

        P_safe = np.clip(P_known_n, 1e-300, 1.0)
        kl_known = np.sum(P_known_n * (np.log(P_safe) - log_prior_known), axis=1)

        # rho_n/q_oog = exp(log_f_x - log_p_z - log_total_mass)
        log_ratio_oog = log_f_x - log_p_z[None, :] - log_total_mass[:, None]
        kl_oog = np.sum(rho_dS_n * log_ratio_oog, axis=1)

        kl_total = np.maximum(kl_known + kl_oog, 0.0)

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


def entropy_normalized(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    return h / math.log(p.shape[1])


def threshold_at_fpir(scores_unknown: np.ndarray, target_fpir: float) -> float:
    """
    accept iff max_similarity >= tau.
    """
    scores = np.asarray(scores_unknown, dtype=np.float64).reshape(-1)
    if len(scores) == 0:
        ValueError("Need unknown validation scores to set FPIR threshold.")

    target_fpir = float(target_fpir)
    if target_fpir <= 0.0:
        return float(np.nextafter(scores.max(), np.inf))
    if target_fpir >= 1.0:
        return float(np.nextafter(scores.min(), -np.inf))

    # accept iff score >= tau. Choose tau so approximately target_fpir
    # fraction of unknown validation probes are accepted.
    n_accept = int(np.floor(target_fpir * len(scores)))
    if n_accept <= 0:
        return float(np.nextafter(scores.max(), np.inf))

    sorted_scores = np.sort(scores)
    idx = len(sorted_scores) - n_accept
    idx = int(np.clip(idx, 0, len(sorted_scores) - 1))
    return float(sorted_scores[idx])


@dataclass
class OSRStats:
    labels: np.ndarray
    sim: np.ndarray
    max_sim: np.ndarray
    pred_idx: np.ndarray
    rejected: np.ndarray
    scf_kappa: np.ndarray

    # Collapsed action posterior:
    #   [:, :K] = known identity masses
    #   [:, K]  = total OOD mass
    # This is only a diagnostic representation. The KL is computed over the
    # full mixed discrete-continuous identity space.
    gal_posterior: np.ndarray
    holue_posterior: np.ndarray

    gal_entropy: np.ndarray
    holue_entropy: np.ndarray

    # Mixed KL components.
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
    beta: float,
    circle_grid: int,
    sphere_theta_grid: int,
    sphere_phi_grid: int,
) -> OSRStats:
    """
    Compute fixed-pipeline OSR predictions plus GalUE/HolUE quantities.

    Important:
    OSR decisions are made by the fixed cosine-threshold rule:
        accept iff max_c <mu_x, m_c> >= tau.

    HolUE is used only as an uncertainty score, not to change decisions.
    """
    probe_mu = normalize_np(np.asarray(probe_mu, dtype=np.float64))
    probe_kappa = np.asarray(probe_kappa, dtype=np.float64).reshape(-1)
    gallery_mu = normalize_np(np.asarray(gallery_mu, dtype=np.float64))

    dim = probe_mu.shape[1]
    if dim not in [2, 3]:
        raise ValueError(f"Only dim=2 and dim=3 are supported, got dim={dim}")

    sim = probe_mu @ gallery_mu.T
    max_sim = np.max(sim, axis=1)
    pred_idx = np.argmax(sim, axis=1)
    rejected = max_sim < float(tau)

    gal_post = posterior_from_z_mixed_prior(
        probe_mu,
        gallery_mu=gallery_mu,
        gallery_kappa=gallery_kappa,
        beta=beta,
    )

    hol_post, kl_known, kl_oog, kl_total = mixed_holue_posterior_quadrature(
        mu=probe_mu,
        kappa=probe_kappa,
        gallery_mu=gallery_mu,
        gallery_kappa=gallery_kappa,
        beta=beta,
        circle_grid=circle_grid,
        sphere_theta_grid=sphere_theta_grid,
        sphere_phi_grid=sphere_phi_grid,
    )

    return OSRStats(
        labels=np.asarray(probe_labels, dtype=int),
        sim=sim,
        max_sim=max_sim,
        pred_idx=pred_idx,
        rejected=rejected,
        scf_kappa=probe_kappa,
        gal_posterior=gal_post,
        holue_posterior=hol_post,
        gal_entropy=entropy_normalized(gal_post),
        holue_entropy=entropy_normalized(hol_post),
        kl_known=kl_known,
        kl_oog=kl_oog,
        kl_total=kl_total,
    )


# ---------------------------------------------------------------------
# Metrics and uncertainty filtering
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
    known_arr = np.asarray(known_classes, dtype=int)
    pred_digit = known_arr[np.asarray(pred_idx, dtype=int)]

    labels = np.asarray(labels, dtype=int)
    seen = np.isin(labels, known_arr)

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
    known_arr = np.asarray(known_classes, dtype=int)
    pred_digit = known_arr[np.asarray(pred_idx, dtype=int)]

    labels = np.asarray(labels, dtype=int)
    rejected = np.asarray(rejected, dtype=bool)
    seen = np.isin(labels, known_arr)

    true_accept_true_ident = seen & (~rejected) & (pred_digit == labels)
    false_accept = (~seen) & (~rejected)
    false_reject = seen & rejected
    misidentification = seen & (~rejected) & (pred_digit != labels)

    tp = int(np.sum(true_accept_true_ident))
    fp = int(np.sum(false_accept))
    fn = int(np.sum(false_reject) + np.sum(misidentification))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    fpir = safe_div(fp, np.sum(~seen))
    fnir = 1.0 - safe_div(tp, np.sum(seen))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": float(f1),
        "fpir": float(fpir),
        "fnir": float(fnir),
        "precision": float(precision),
        "recall": float(recall),
        "error_rate": float(
            np.mean(osr_error_mask(pred_idx, rejected, labels, known_classes))
        ),
        "false_accept_count": int(np.sum(false_accept)),
        "false_reject_count": int(np.sum(false_reject)),
        "misidentification_count": int(np.sum(misidentification)),
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
    Higher uncertainty is filtered earlier.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    order = np.argsort(-uncertainty)
    n = len(uncertainty)

    rows = []
    for frac in fractions:
        frac = float(frac)
        n_drop = int(round(frac * n))

        keep = np.ones(n, dtype=bool)
        if n_drop > 0:
            keep[order[:n_drop]] = False

        m = compute_osr_metrics(
            pred_idx=np.asarray(pred_idx)[keep],
            rejected=np.asarray(rejected)[keep],
            labels=np.asarray(labels)[keep],
            known_classes=known_classes,
        )
        rows.append({"fraction": frac, **m})

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


def plot_rejection_curve_panels(
    curves: Dict[str, pd.DataFrame],
    out_dir: Path,
    metrics: List[str],
    plot_cfg: dict,
    method_order: List[str],
) -> pd.DataFrame:
    """
    Side-by-side filtering plots for several metrics, e.g. F1, FPIR, FNIR.

    PRR is computed for every metric and saved to CSV. The same PRR formula
    works for both higher-is-better metrics such as F1 and lower-is-better
    metrics such as FPIR/FNIR, because the oracle and random AUCs define the
    direction automatically.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Random" not in curves or "Oracle" not in curves:
        raise ValueError("curves must contain Random and Oracle.")

    metrics = list(metrics)

    figsize = tuple(cfg_get(plot_cfg, "panel_figsize", [15.5, 4.7]))
    dpi = int(cfg_get(plot_cfg, "dpi", 300))
    grid = bool(cfg_get(plot_cfg, "grid", True))
    legend_fontsize = float(cfg_get(plot_cfg, "legend_fontsize", 8.5))

    save_png = str(
        cfg_get(
            plot_cfg,
            "save_panel_png",
            "rejection_curves_f1_fpir_fnir.png",
        )
    )
    save_pdf = str(
        cfg_get(
            plot_cfg,
            "save_panel_pdf",
            "rejection_curves_f1_fpir_fnir.pdf",
        )
    )
    save_prr_csv = str(
        cfg_get(
            plot_cfg,
            "save_panel_prr_csv",
            "rejection_curves_f1_fpir_fnir_prr.csv",
        )
    )

    metric_labels = {
        "f1": "F1 ↑",
        "fpir": "FPIR ↓",
        "fnir": "FNIR ↓",
        "precision": "Precision ↑",
        "recall": "Recall ↑",
        "error_rate": "Error rate ↓",
    }

    metric_titles = {
        "f1": "F1 filtering curve",
        "fpir": "FPIR filtering curve",
        "fnir": "FNIR filtering curve",
        "precision": "Precision filtering curve",
        "recall": "Recall filtering curve",
        "error_rate": "Error-rate filtering curve",
    }

    names = [n for n in method_order if n in curves]
    names += [n for n in curves.keys() if n not in names]

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize,
        squeeze=False,
        sharex=True,
    )
    axes = axes[0]

    prr_rows = []

    for metric, ax in zip(metrics, axes):
        if metric not in next(iter(curves.values())).columns:
            raise ValueError(
                f"Metric '{metric}' is not present in rejection-curve DataFrames. "
                f"Available columns: {list(next(iter(curves.values())).columns)}"
            )

        prr_for_metric = {}
        for name, df in curves.items():
            prr_for_metric[name] = compute_prr(
                curve=df,
                random_curve=curves["Random"],
                oracle_curve=curves["Oracle"],
                metric=metric,
            )

        for name in names:
            df = curves[name]

            lw = 2.8 if "HolUE" in name else 1.7
            alpha = 0.75 if name in {"Random", "Oracle"} else 1.0

            ax.plot(
                df["fraction"],
                df[metric],
                label=name,
                linewidth=lw,
                alpha=alpha,
            )

        ax.set_title(metric_titles.get(metric, metric.upper()))
        ax.set_xlabel("Filtered-out probe fraction")
        ax.set_ylabel(metric_labels.get(metric, metric.upper()))

        if metric in {"fpir", "fnir", "error_rate"}:
            ax.set_ylim(bottom=0.0)

        if grid:
            ax.grid(True, linestyle="--", alpha=0.45)

        for name in names:
            row = {
                "method": name,
                "metric": metric,
                "PRR": prr_for_metric[name],
            }
            prr_rows.append(row)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        fontsize=legend_fontsize,
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )

    title = str(
        cfg_get(
            plot_cfg,
            "panel_title",
            "MNIST toy OSR: uncertainty-based filtering",
        )
    )
    fig.suptitle(title, fontsize=13)

    plt.tight_layout(rect=[0.0, 0.10, 1.0, 0.94])

    plt.savefig(out_dir / save_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(out_dir / save_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    prr_df_long = pd.DataFrame(prr_rows)
    prr_df_long.to_csv(out_dir / save_prr_csv, index=False)

    prr_df_wide = prr_df_long.pivot(
        index="method",
        columns="metric",
        values="PRR",
    ).reset_index()

    prr_df_wide.columns = [
        "method" if c == "method" else f"PRR_{c}" for c in prr_df_wide.columns
    ]
    prr_df_wide.to_csv(
        out_dir / save_prr_csv.replace(".csv", "_wide.csv"),
        index=False,
    )

    return prr_df_wide


def plot_rejection_curves(
    curves: Dict[str, pd.DataFrame],
    out_dir: Path,
    metric: str,
    plot_cfg: dict,
    method_order: List[str],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Random" not in curves or "Oracle" not in curves:
        raise ValueError("curves must contain Random and Oracle.")

    figsize = tuple(cfg_get(plot_cfg, "figsize", [8.5, 5.3]))
    dpi = int(cfg_get(plot_cfg, "dpi", 300))
    title = str(
        cfg_get(plot_cfg, "title", "MNIST toy OSR: uncertainty-based filtering")
    )
    xlabel = str(cfg_get(plot_cfg, "xlabel", "Filtered-out probe fraction"))
    ylabel = str(cfg_get(plot_cfg, "ylabel", metric.upper()))
    grid = bool(cfg_get(plot_cfg, "grid", True))
    legend_fontsize = float(cfg_get(plot_cfg, "legend_fontsize", 8.5))
    show_prr = bool(cfg_get(plot_cfg, "show_prr_in_legend", True))

    save_png = str(cfg_get(plot_cfg, "save_png", "rejection_curves.png"))
    save_pdf = str(cfg_get(plot_cfg, "save_pdf", "rejection_curves.pdf"))
    save_prr_csv = str(
        cfg_get(plot_cfg, "save_prr_csv", f"rejection_curve_prr_{metric}.csv")
    )

    prr_values = {}
    for name, df in curves.items():
        prr_values[name] = compute_prr(
            df,
            random_curve=curves["Random"],
            oracle_curve=curves["Oracle"],
            metric=metric,
        )

    names = [n for n in method_order if n in curves]
    names += [n for n in curves.keys() if n not in names]

    plt.figure(figsize=figsize)

    for name in names:
        df = curves[name]
        lw = 2.8 if "HolUE" in name else 1.7
        alpha = 0.75 if name in {"Random", "Oracle"} else 1.0

        if show_prr:
            prr = prr_values[name]
            label = (
                f"{name} (PRR={prr:.2f})" if np.isfinite(prr) else f"{name} (PRR=nan)"
            )
        else:
            label = name

        plt.plot(df["fraction"], df[metric], label=label, linewidth=lw, alpha=alpha)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    plt.savefig(out_dir / save_png, dpi=dpi)
    plt.savefig(out_dir / save_pdf, dpi=dpi, bbox_inches="tight")
    plt.close()

    prr_df = pd.DataFrame(
        [{"method": name, f"PRR_{metric}": prr_values[name]} for name in names]
    )
    prr_df.to_csv(out_dir / save_prr_csv, index=False)
    return prr_df


def plot_teaser_circle(
    stats: OSRStats,
    probe_mu: np.ndarray,
    gallery_mu: np.ndarray,
    known_classes: List[int],
    beta: float,
    gallery_kappa: float,
    uncertainty: np.ndarray,
    out_dir: Path,
    seed: int,
    max_points: int = 900,
) -> None:
    """
    2D teaser on unit circle.

    Uses the same mixed-prior GalUE posterior as the experiment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_mu = normalize_np(probe_mu)
    gallery_mu = normalize_np(gallery_mu)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)

    rng = np.random.default_rng(seed)
    err = osr_error_mask(
        stats.pred_idx,
        stats.rejected,
        stats.labels,
        known_classes,
    )

    n = len(stats.labels)
    if n > max_points:
        err_idx = np.where(err)[0]
        ok_idx = np.where(~err)[0]

        remaining = max(0, max_points - len(err_idx))
        if len(ok_idx) > remaining:
            ok_idx = rng.choice(ok_idx, size=remaining, replace=False)

        selected = np.concatenate([err_idx, ok_idx])
    else:
        selected = np.arange(n)

    selected = np.asarray(selected, dtype=int)
    ok_sel = selected[~err[selected]]
    err_sel = selected[err[selected]]

    unc_min = float(np.nanmin(uncertainty))
    unc_max = float(np.nanmax(uncertainty))
    unc_norm = (uncertainty - unc_min) / (unc_max - unc_min + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 8))

    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    z_grid = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    post_grid = posterior_from_z_mixed_prior(
        z_grid,
        gallery_mu=gallery_mu,
        gallery_kappa=gallery_kappa,
        beta=beta,
    )
    ring_unc = entropy_normalized(post_grid)

    ax.scatter(
        1.12 * z_grid[:, 0],
        1.12 * z_grid[:, 1],
        c=ring_unc,
        cmap="Blues",
        s=18,
        alpha=0.75,
        linewidths=0,
    )

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(known_classes)))

    for i, (digit, proto) in enumerate(zip(known_classes, gallery_mu)):
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
            1.34 * proto[0],
            1.34 * proto[1],
            f"digit {digit}",
            color=colors[i],
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
        )

    def point_xy(idx: np.ndarray):
        r = 0.88 + 0.16 * unc_norm[idx]
        return r * probe_mu[idx, 0], r * probe_mu[idx, 1]

    if len(ok_sel) > 0:
        x, y = point_xy(ok_sel)
        ax.scatter(
            x,
            y,
            c=uncertainty[ok_sel],
            cmap="magma",
            s=22,
            alpha=0.45,
            linewidths=0,
            marker="o",
            label="correct probes",
        )

    if len(err_sel) > 0:
        x, y = point_xy(err_sel)
        sc = ax.scatter(
            x,
            y,
            c=uncertainty[err_sel],
            cmap="magma",
            s=65,
            alpha=0.95,
            linewidths=1.6,
            marker="x",
            label="OSR errors",
        )
    else:
        sc = ax.scatter([], [], c=[], cmap="magma")

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
    ax.set_title("HolUE on MNIST OSR: mixed-prior uncertainty", fontsize=14)

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("HolUE no-calibration uncertainty: -mixed KL")

    ax.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()

    plt.savefig(out_dir / "teaser_holue_mnist_circle.png", dpi=300)
    plt.savefig(out_dir / "teaser_holue_mnist_circle.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_holue_vs_scf_scatter(
    holue_unc: np.ndarray,
    scf_unc: np.ndarray,
    error: np.ndarray,
    out_dir: Path,
    plot_cfg: dict,
) -> None:
    if not bool(cfg_get(plot_cfg, "enabled", True)):
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    figsize = tuple(cfg_get(plot_cfg, "figsize", [7, 4.5]))
    dpi = int(cfg_get(plot_cfg, "dpi", 300))
    title = str(cfg_get(plot_cfg, "title", "MNIST toy OSR: HolUE vs SCF"))
    xlabel = str(cfg_get(plot_cfg, "xlabel", "SCF uncertainty: -log kappa"))
    ylabel = str(cfg_get(plot_cfg, "ylabel", "HolUE uncertainty: -mixed KL"))

    plt.figure(figsize=figsize)
    plt.scatter(
        scf_unc,
        holue_unc,
        c=error.astype(int),
        cmap=str(cfg_get(plot_cfg, "cmap", "coolwarm")),
        s=float(cfg_get(plot_cfg, "marker_size", 20)),
        alpha=float(cfg_get(plot_cfg, "alpha", 0.75)),
        linewidths=0,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cb = plt.colorbar()
    cb.set_label("OSR error")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    plt.savefig(
        out_dir / str(cfg_get(plot_cfg, "save_png", "holue_vs_scf_error_scatter.png")),
        dpi=dpi,
    )
    plt.savefig(
        out_dir / str(cfg_get(plot_cfg, "save_pdf", "holue_vs_scf_error_scatter.pdf")),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------
# Optional corruption visualization
# ---------------------------------------------------------------------


def mnist_tensor_to_image(x: torch.Tensor) -> np.ndarray:
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
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return

    rng = np.random.default_rng(seed)
    n = min(num_examples, len(indices))
    chosen = rng.choice(indices, size=n, replace=False)

    ncols = min(ncols, n)
    nblocks = int(math.ceil(n / ncols))
    nrows = 2 * nblocks

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.55 * ncols, 1.75 * nrows), squeeze=False
    )

    for ax in axes.reshape(-1):
        ax.axis("off")

    for j, idx in enumerate(chosen):
        block = j // ncols
        col = j % ncols

        img, label = base[int(idx)]
        img_corr = corrupt_tensor_mnist(
            img,
            seed=seed + int(idx),
            corruption_cfg=corruption_cfg,
        )

        axes[2 * block, col].imshow(
            mnist_tensor_to_image(img), cmap="gray", vmin=0, vmax=1
        )
        axes[2 * block + 1, col].imshow(
            mnist_tensor_to_image(img_corr), cmap="gray", vmin=0, vmax=1
        )

        axes[2 * block, col].set_title(f"orig {int(label)}", fontsize=8)
        axes[2 * block + 1, col].set_title("corrupt", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Train/load one embedding branch and evaluate
# ---------------------------------------------------------------------


def train_or_load_models(
    args: SimpleNamespace,
    dim: int,
    known_classes: List[int],
    train_ds_arcface: Dataset,
    train_ds_scf: Dataset,
    device: torch.device,
    model_dir: Path,
) -> Tuple[TinyArcFaceMNIST, TinySCFMNIST]:
    arc = TinyArcFaceMNIST(num_classes=len(known_classes), embedding_dim=dim)
    arc_ckpt = model_dir / args.arcface_checkpoint[dim]

    if arc_ckpt.is_file() and not args.force_train:
        ckpt = torch.load(arc_ckpt, map_location=device, weights_only=False)
        arc.load_state_dict(ckpt["state_dict"])
        print(f"[{dim}D] Loaded ArcFace checkpoint: {arc_ckpt}")
    else:
        print(f"[{dim}D] Training ArcFace...")
        train_arcface(
            model=arc,
            train_ds=train_ds_arcface,
            device=device,
            epochs=args.arcface_epochs[dim],
            batch_size=args.batch_size,
            lr=args.arcface_lr,
            seed=args.seed + 1000 * dim,
            arcface_s=args.arcface_scale,
            arcface_m=args.arcface_margin,
            weight_decay=args.arcface_weight_decay,
        )
        torch.save(
            {
                "state_dict": arc.state_dict(),
                "known_classes": known_classes,
                "embedding_dim": dim,
                "args": to_jsonable(args),
            },
            arc_ckpt,
        )
        print(f"[{dim}D] Saved ArcFace checkpoint: {arc_ckpt}")

    scf = TinySCFMNIST(
        arc_model=arc,
        kappa_min=args.scf_kappa_min,
        kappa_max=args.scf_kappa_max,
    )
    scf_ckpt = model_dir / args.scf_checkpoint[dim]

    if scf_ckpt.is_file() and not args.force_train:
        ckpt = torch.load(scf_ckpt, map_location=device, weights_only=False)
        scf.load_state_dict(ckpt["state_dict"])
        print(f"[{dim}D] Loaded SCF checkpoint: {scf_ckpt}")
    else:
        print(f"[{dim}D] Training SCF...")
        train_scf(
            scf_model=scf,
            arc_model=arc,
            train_ds=train_ds_scf,
            device=device,
            epochs=args.scf_epochs[dim],
            batch_size=args.batch_size,
            lr=args.scf_lr,
            seed=args.seed + 2000 * dim,
            weight_decay=args.scf_weight_decay,
            kappa_regularizer=args.scf_kappa_regularizer,
        )
        torch.save(
            {
                "state_dict": scf.state_dict(),
                "known_classes": known_classes,
                "embedding_dim": dim,
                "args": to_jsonable(args),
            },
            scf_ckpt,
        )
        print(f"[{dim}D] Saved SCF checkpoint: {scf_ckpt}")

    return arc, scf


def make_holue_calibration_features(stats: OSRStats) -> np.ndarray:
    """
    Paper-like HolUE calibration features.

    We intentionally use only the two mixed-KL components, mirroring the
    KL1/KL2 calibration idea rather than a broad meta-classifier.
    """
    return np.column_stack([stats.kl_known, stats.kl_oog]).astype(np.float64)


def fit_holue_calibrator(
    args: SimpleNamespace,
    val_stats: OSRStats,
    val_error: np.ndarray,
    test_stats: OSRStats,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Logistic calibration of HolUE KL components on validation errors.

    Returns probability of OSR error. Higher = more uncertain.
    """
    info: Dict[str, object] = {
        "used": False,
        "feature_names": ["kl_known", "kl_oog"],
        "reason": "",
    }

    fallback_unc = -test_stats.kl_total

    if not args.calibration_enabled:
        info["reason"] = "disabled"
        return fallback_unc, info

    y_val = val_error.astype(int)
    if len(np.unique(y_val)) < 2:
        info["reason"] = "single_class_validation_error_labels"
        return fallback_unc, info

    X_val = make_holue_calibration_features(val_stats)
    X_test = make_holue_calibration_features(test_stats)

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

    unc = clf.predict_proba(X_test_scaled)[:, 1]

    info.update(
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

    return unc, info


def evaluate_uncertainty_methods(
    args: SimpleNamespace,
    dim: int,
    val_stats: OSRStats,
    test_stats: OSRStats,
    known_classes: List[int],
    tau: float,
    beta: float,
    result_dir: Path,
    figure_dir: Path,
    test_probe_mu: np.ndarray,
    test_gallery_mu: np.ndarray,
) -> pd.DataFrame:
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    val_error = osr_error_mask(
        val_stats.pred_idx,
        val_stats.rejected,
        val_stats.labels,
        known_classes,
    )
    test_error = osr_error_mask(
        test_stats.pred_idx,
        test_stats.rejected,
        test_stats.labels,
        known_classes,
    )

    val_metrics = compute_osr_metrics(
        val_stats.pred_idx,
        val_stats.rejected,
        val_stats.labels,
        known_classes,
    )
    test_metrics = compute_osr_metrics(
        test_stats.pred_idx,
        test_stats.rejected,
        test_stats.labels,
        known_classes,
    )

    print(f"\n[{dim}D] Validation OSR metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n[{dim}D] Test OSR metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    with open(result_dir / f"base_osr_metrics_{dim}d.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dim": dim,
                "tau": tau,
                "beta": beta,
                "target_fpir": args.target_fpir,
                "gallery_kappa": args.gallery_kappa,
                "val": val_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    holue_cal_unc, calibration_info = fit_holue_calibrator(
        args=args,
        val_stats=val_stats,
        val_error=val_error,
        test_stats=test_stats,
    )

    with open(
        result_dir / f"holue_calibration_info_{dim}d.json", "w", encoding="utf-8"
    ) as f:
        json.dump(calibration_info, f, indent=2)

    rng_eval = np.random.default_rng(args.seed + 12345 + dim)

    holue_unc = -test_stats.kl_total
    holue_entropy_unc = test_stats.holue_entropy

    # Paper GalUE confidence would be max_c p(c|z); uncertainty is 1-max.
    galue_unc = 1.0 - np.max(test_stats.gal_posterior, axis=1)

    scf_unc = -np.log(test_stats.scf_kappa + 1e-8)
    accscr_unc = -np.abs(test_stats.max_sim - tau)
    maxsim_unc = -test_stats.max_sim

    random_unc = rng_eval.random(len(test_stats.labels))
    oracle_unc = test_error.astype(float) + 1e-6 * rng_eval.random(len(test_error))

    uncertainties: Dict[str, np.ndarray] = {
        "HolUE no calibration": holue_unc,
        "HolUE calibrated": holue_cal_unc,
        "HolUE collapsed entropy": holue_entropy_unc,
        "GalUE": galue_unc,
        "SCF": scf_unc,
        "AccScr": accscr_unc,
        "MaxSim": maxsim_unc,
        "Random": random_unc,
        "Oracle": oracle_unc,
    }

    np.savez(
        result_dir / f"uncertainty_scores_{dim}d.npz",
        **{k.replace(" ", "_").replace("-", "_"): v for k, v in uncertainties.items()},
    )

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
        if name not in args.method_order:
            continue
        df = rejection_curve(
            uncertainty=unc,
            pred_idx=test_stats.pred_idx,
            rejected=test_stats.rejected,
            labels=test_stats.labels,
            known_classes=known_classes,
            fractions=fractions,
        )
        curves[name] = df
        df.to_csv(
            result_dir / f"rejection_curve_{slugify(name)}_{dim}d.csv", index=False
        )

    plot_cfg = cfg_get(args.plots, "rejection_curves", {})

    # Existing single-metric plot, usually F1.
    metric = str(cfg_get(plot_cfg, "metric", "f1"))
    prr_df = plot_rejection_curves(
        curves=curves,
        out_dir=figure_dir,
        metric=metric,
        plot_cfg=plot_cfg,
        method_order=args.method_order,
    )

    # New side-by-side panel: F1, FPIR, FNIR.
    panel_metrics = list(cfg_get(plot_cfg, "panel_metrics", ["f1", "fpir", "fnir"]))
    plot_rejection_curve_panels(
        curves=curves,
        out_dir=figure_dir,
        metrics=panel_metrics,
        plot_cfg=plot_cfg,
        method_order=args.method_order,
    )

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        average_precision_score = None
        roc_auc_score = None

    rows = []
    for name, unc in uncertainties.items():
        df = curves[name]
        prr_f1 = compute_prr(
            df,
            random_curve=curves["Random"],
            oracle_curve=curves["Oracle"],
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
                "embedding_dim": dim,
                "tau": tau,
                "beta": beta,
                "base_f1": test_metrics["f1"],
                "base_fpir": test_metrics["fpir"],
                "base_fnir": test_metrics["fnir"],
                "base_error_rate": test_metrics["error_rate"],
                "prr_f1": prr_f1,
                "error_roc_auc": error_auc,
                "error_average_precision": error_ap,
                "f1_after_max_filter": float(df.iloc[-1]["f1"]),
                "fpir_after_max_filter": float(df.iloc[-1]["fpir"]),
                "fnir_after_max_filter": float(df.iloc[-1]["fnir"]),
            }
        )

    summary_df = pd.DataFrame(rows)

    order = [n for n in args.method_order if n in set(summary_df["method"])]
    order += [n for n in summary_df["method"].tolist() if n not in order]

    summary_df["method"] = pd.Categorical(
        summary_df["method"], categories=order, ordered=True
    )
    summary_df = summary_df.sort_values("method").reset_index(drop=True)
    summary_df.to_csv(result_dir / f"method_summary_{dim}d.csv", index=False)

    print(f"\n[{dim}D] Method summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 2D teaser.
    if dim == 2 and bool(
        cfg_get(cfg_get(args.plots, "teaser_circle_2d", {}), "enabled", True)
    ):
        teaser_cfg = cfg_get(args.plots, "teaser_circle_2d", {})
        plot_teaser_circle(
            stats=test_stats,
            probe_mu=test_probe_mu,
            gallery_mu=test_gallery_mu,
            known_classes=known_classes,
            beta=beta,
            gallery_kappa=args.gallery_kappa,
            uncertainty=holue_unc,
            out_dir=figure_dir,
            seed=args.seed,
            max_points=int(cfg_get(teaser_cfg, "max_points", 900)),
        )

    scatter_cfg = cfg_get(args.plots, "holue_vs_scf_scatter", {})
    if dim == 2:
        plot_holue_vs_scf_scatter(
            holue_unc=holue_unc,
            scf_unc=scf_unc,
            error=test_error,
            out_dir=figure_dir,
            plot_cfg=scatter_cfg,
        )

    return summary_df


def run_mnist_branch(
    args: SimpleNamespace,
    dim: int,
    mnist_train: datasets.MNIST,
    mnist_test: datasets.MNIST,
    train_indices: np.ndarray,
    val_protocol: ProtocolIndices,
    test_protocol: ProtocolIndices,
    known_classes: List[int],
    class_to_idx: Dict[int, int],
    device: torch.device,
    out_dir: Path,
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"Running MNIST HolUE branch: {dim}D embeddings")
    print("=" * 80)

    model_dir = resolve_under_out_dir(args.model_dir, out_dir)
    result_dir = resolve_under_out_dir(args.result_dir[dim], out_dir)
    figure_dir = resolve_under_out_dir(args.figure_dir[dim], out_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

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

    arc_model, scf_model = train_or_load_models(
        args=args,
        dim=dim,
        known_classes=known_classes,
        train_ds_arcface=train_ds_arcface,
        train_ds_scf=train_ds_scf,
        device=device,
        model_dir=model_dir,
    )

    print(f"[{dim}D] Extracting validation gallery embeddings...")
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

    print(f"[{dim}D] Extracting validation probe embeddings + SCF kappa...")
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

    print(f"[{dim}D] Extracting test gallery embeddings...")
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

    print(f"[{dim}D] Extracting test probe embeddings + SCF kappa...")
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
        result_dir / f"mnist_embeddings_for_toy_osr_{dim}d.npz",
        val_gallery_mu=val_gallery_mu,
        val_probe_mu=val_probe_mu,
        val_probe_kappa=val_probe_kappa,
        val_probe_labels=val_probe_labels,
        test_gallery_mu=test_gallery_mu,
        test_probe_mu=test_probe_mu,
        test_probe_kappa=test_probe_kappa,
        test_probe_labels=test_probe_labels,
        known_classes=np.asarray(known_classes, dtype=int),
        unknown_classes=np.asarray(args.unknown_classes, dtype=int),
    )

    # Fixed OSR threshold selected on validation unknown probes.
    val_sim = val_probe_mu @ val_gallery_mu.T
    val_max_sim = np.max(val_sim, axis=1)
    val_seen = np.isin(val_probe_labels, np.asarray(known_classes, dtype=int))

    tau = threshold_at_fpir(val_max_sim[~val_seen], args.target_fpir)

    if args.osr_beta is None:
        beta = beta_from_tau_mixed_prior(
            dim=dim,
            tau=tau,
            gallery_kappa=args.gallery_kappa,
            K=len(known_classes),
        )
    else:
        beta = float(args.osr_beta)
        implied_tau = tau_from_beta_mixed_prior(
            dim=dim,
            beta=beta,
            gallery_kappa=args.gallery_kappa,
            K=len(known_classes),
        )
        if abs(implied_tau - tau) > 1e-4:
            print(
                f"[{dim}D][Warning] Fixed beta and validation tau are inconsistent: "
                f"OSR tau={tau:.6f}, posterior-implied tau={implied_tau:.6f}. "
                "For strict equivalence, set osr.beta: auto."
            )

    print(
        f"[{dim}D] Validation-selected tau={tau:.6f} at target FPIR={args.target_fpir}"
    )
    print(f"[{dim}D] Mixed-prior beta={beta:.8f}")

    print(f"[{dim}D] Computing validation GalUE/HolUE...")
    val_stats = compute_osr_stats(
        probe_mu=val_probe_mu,
        probe_kappa=val_probe_kappa,
        probe_labels=val_probe_labels,
        gallery_mu=val_gallery_mu,
        tau=tau,
        gallery_kappa=args.gallery_kappa,
        beta=beta,
        circle_grid=args.circle_grid,
        sphere_theta_grid=args.sphere_theta_grid,
        sphere_phi_grid=args.sphere_phi_grid,
    )

    print(f"[{dim}D] Computing test GalUE/HolUE...")
    test_stats = compute_osr_stats(
        probe_mu=test_probe_mu,
        probe_kappa=test_probe_kappa,
        probe_labels=test_probe_labels,
        gallery_mu=test_gallery_mu,
        tau=tau,
        gallery_kappa=args.gallery_kappa,
        beta=beta,
        circle_grid=args.circle_grid,
        sphere_theta_grid=args.sphere_theta_grid,
        sphere_phi_grid=args.sphere_phi_grid,
    )

    np.savez(
        result_dir / f"posteriors_and_osr_stats_{dim}d.npz",
        tau=tau,
        beta=beta,
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
    )

    return evaluate_uncertainty_methods(
        args=args,
        dim=dim,
        val_stats=val_stats,
        test_stats=test_stats,
        known_classes=known_classes,
        tau=tau,
        beta=beta,
        result_dir=result_dir,
        figure_dir=figure_dir,
        test_probe_mu=test_probe_mu,
        test_gallery_mu=test_gallery_mu,
    )


# ---------------------------------------------------------------------
# Controlled synthetic circle experiment
# ---------------------------------------------------------------------


def angle_to_unit_vectors(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def sample_controlled_circle_split(
    n: int,
    center_angles: np.ndarray,
    beta: float,
    gallery_kappa: float,
    kappa_high: float,
    kappa_low: float,
    low_quality_prob: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Controlled S^1 embedding-level toy.

    labels:
    0..K-1 = known identities
    -1     = OOD identity
    """
    K = len(center_angles)

    labels = np.full(n, -1, dtype=int)
    theta_true = np.empty(n, dtype=np.float64)

    is_oog = rng.random(n) < beta
    known_idx = np.where(~is_oog)[0]
    oog_idx = np.where(is_oog)[0]

    if len(known_idx) > 0:
        known_labels = rng.integers(0, K, size=len(known_idx))
        labels[known_idx] = known_labels
        for idx, c in zip(known_idx, known_labels):
            theta_true[idx] = rng.vonmises(center_angles[c], gallery_kappa)

    if len(oog_idx) > 0:
        theta_true[oog_idx] = rng.uniform(0.0, 2.0 * math.pi, size=len(oog_idx))

    low_quality = rng.random(n) < low_quality_prob
    kappa_x = np.where(low_quality, float(kappa_low), float(kappa_high))

    theta_obs = np.empty(n, dtype=np.float64)
    for i in range(n):
        theta_obs[i] = rng.vonmises(theta_true[i], kappa_x[i])

    mu_obs = angle_to_unit_vectors(theta_obs)
    return mu_obs, kappa_x, labels


def run_controlled_circle_experiment(args: SimpleNamespace, out_dir: Path) -> None:
    cfg = args.controlled_circle
    if not bool(cfg_get(cfg, "enabled", False)):
        return

    print("\n" + "=" * 80)
    print("Running controlled synthetic circle experiment")
    print("=" * 80)

    result_dir = out_dir / "controlled_circle_results"
    figure_dir = out_dir / "controlled_circle_figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg_get(cfg, "seed", args.seed + 90000))
    rng = np.random.default_rng(seed)

    K = int(cfg_get(cfg, "num_known_classes", 5))
    n_val = int(cfg_get(cfg, "n_val", 6000))
    n_test = int(cfg_get(cfg, "n_test", 6000))

    beta = float(cfg_get(cfg, "beta", 0.5))
    gallery_kappa = float(cfg_get(cfg, "gallery_kappa", 18.0))
    kappa_high = float(cfg_get(cfg, "kappa_high", 80.0))
    kappa_low = float(cfg_get(cfg, "kappa_low", 1.5))
    low_quality_prob = float(cfg_get(cfg, "low_quality_prob", 0.30))
    circle_grid = int(cfg_get(cfg, "circle_grid", args.circle_grid))

    center_angles = np.linspace(0.0, 2.0 * math.pi, K, endpoint=False)
    gallery_mu = angle_to_unit_vectors(center_angles)
    known_classes = list(range(K))

    tau = tau_from_beta_mixed_prior(
        dim=2,
        beta=beta,
        gallery_kappa=gallery_kappa,
        K=K,
    )

    val_mu, val_kappa, val_labels = sample_controlled_circle_split(
        n=n_val,
        center_angles=center_angles,
        beta=beta,
        gallery_kappa=gallery_kappa,
        kappa_high=kappa_high,
        kappa_low=kappa_low,
        low_quality_prob=low_quality_prob,
        rng=rng,
    )
    test_mu, test_kappa, test_labels = sample_controlled_circle_split(
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
        beta=beta,
        circle_grid=circle_grid,
        sphere_theta_grid=args.sphere_theta_grid,
        sphere_phi_grid=args.sphere_phi_grid,
    )
    test_stats = compute_osr_stats(
        probe_mu=test_mu,
        probe_kappa=test_kappa,
        probe_labels=test_labels,
        gallery_mu=gallery_mu,
        tau=tau,
        gallery_kappa=gallery_kappa,
        beta=beta,
        circle_grid=circle_grid,
        sphere_theta_grid=args.sphere_theta_grid,
        sphere_phi_grid=args.sphere_phi_grid,
    )

    # Temporarily use a local namespace for gallery_kappa if different.
    old_gk = args.gallery_kappa
    args.gallery_kappa = gallery_kappa
    try:
        evaluate_uncertainty_methods(
            args=args,
            dim=2,
            val_stats=val_stats,
            test_stats=test_stats,
            known_classes=known_classes,
            tau=tau,
            beta=beta,
            result_dir=result_dir,
            figure_dir=figure_dir,
            test_probe_mu=test_mu,
            test_gallery_mu=gallery_mu,
        )
    finally:
        args.gallery_kappa = old_gk

    with open(result_dir / "controlled_circle_setup.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "K": K,
                "n_val": n_val,
                "n_test": n_test,
                "beta": beta,
                "gallery_kappa": gallery_kappa,
                "kappa_high": kappa_high,
                "kappa_low": kappa_low,
                "low_quality_prob": low_quality_prob,
                "tau_implied_by_beta": tau,
                "circle_grid": circle_grid,
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------
# Main script utilities
# ---------------------------------------------------------------------


def maybe_save_corruption_visualizations(
    args: SimpleNamespace,
    mnist_train: datasets.MNIST,
    mnist_test: datasets.MNIST,
    train_indices: np.ndarray,
    val_protocol: ProtocolIndices,
    test_protocol: ProtocolIndices,
    figure_dir: Path,
) -> None:
    vis_cfg = cfg_get(args.plots, "corruption_visualization", {})
    if not bool(cfg_get(vis_cfg, "enabled", True)):
        return

    figure_dir.mkdir(parents=True, exist_ok=True)

    num_examples = int(cfg_get(vis_cfg, "num_pair_examples", 16))
    ncols = int(cfg_get(vis_cfg, "pair_grid_cols", 8))
    title_prefix = str(
        cfg_get(vis_cfg, "title_prefix", "MNIST corruption sanity check")
    )

    save_original_vs_corrupted_grid(
        base=mnist_train,
        indices=train_indices,
        out_path=figure_dir
        / str(
            cfg_get(vis_cfg, "save_train_pairs", "corruption_pairs_train_forced.png")
        ),
        corruption_cfg=args.corruption,
        seed=args.seed + 111,
        num_examples=num_examples,
        ncols=ncols,
        title=f"{title_prefix}: train original vs corrupted",
    )

    save_original_vs_corrupted_grid(
        base=mnist_train,
        indices=val_protocol.corrupt_probe,
        out_path=figure_dir
        / str(
            cfg_get(
                vis_cfg, "save_val_probe_pairs", "corruption_pairs_val_probe_forced.png"
            )
        ),
        corruption_cfg=args.corruption,
        seed=args.seed + 222,
        num_examples=num_examples,
        ncols=ncols,
        title=f"{title_prefix}: validation probe original vs corrupted",
    )

    save_original_vs_corrupted_grid(
        base=mnist_test,
        indices=test_protocol.corrupt_probe,
        out_path=figure_dir
        / str(
            cfg_get(
                vis_cfg,
                "save_test_probe_pairs",
                "corruption_pairs_test_probe_forced.png",
            )
        ),
        corruption_cfg=args.corruption,
        seed=args.seed + 333,
        num_examples=num_examples,
        ncols=ncols,
        title=f"{title_prefix}: test probe original vs corrupted",
    )


def write_markdown_report(
    args: SimpleNamespace,
    out_dir: Path,
    known_classes: List[int],
    unknown_classes: List[int],
    summaries: List[pd.DataFrame],
) -> None:
    summary_text = ""
    for df in summaries:
        if df is not None and len(df) > 0:
            dim = int(df["embedding_dim"].iloc[0])
            summary_text += f"\n## {dim}D method summary\n\n"
            summary_text += df.to_markdown(index=False)
            summary_text += "\n"

    report = f"""# Toy MNIST HolUE open-set recognition experiment

This directory was generated by `toy_mnist_holue.py`.

## Protocol

- Known MNIST classes: `{known_classes}`
- Unknown/OOD MNIST classes: `{unknown_classes}`
- Fixed OSR decision rule: accept iff max gallery cosine similarity exceeds a validation-selected threshold.
- Target validation FPIR: `{args.target_fpir}`
- Gallery vMF concentration: `{args.gallery_kappa}`
- OOD prior beta: `{args.osr_beta if args.osr_beta is not None else "auto"}`

## HolUE score

The main no-calibration HolUE uncertainty score is

```text
u_HolUE(x) = -D_KL(p(c|x) || p(c))
```

where `c` is a mixed identity variable:

```text
c in {{known identities}} union S^(d-1).
```

Known identities have prior mass `(1-beta)/K`. OOD identities have uniform
continuous prior density `beta / S_(d-1)`.

The continuous OOD KL term is essential. Collapsing all OOD identities into one
reject class would discard uncertainty over the OOD identity location and would
not properly incorporate SCF uncertainty from `p(z|x)`.

## Main outputs

- `protocols/mnist_open_set_protocol_indices.npz`
- `results/method_summary_2d.csv`
- `figures/rejection_curves.pdf`
- `figures/teaser_holue_mnist_circle.pdf`
- optional 3D outputs under `results_3d/` and `figures_3d/`
- optional controlled-circle outputs under `controlled_circle_results/`

{summary_text}
"""

    with open(out_dir / "README_toy_results.md", "w", encoding="utf-8") as f:
        f.write(report)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = load_toy_config()
    seed_everything(args.seed, deterministic=args.deterministic)

    out_dir = Path(args.out_dir)
    data_dir = resolve_under_out_dir(args.data_dir, out_dir)
    model_dir = resolve_under_out_dir(args.model_dir, out_dir)
    protocol_dir = resolve_under_out_dir(args.protocol_dir, out_dir)

    for d in [out_dir, data_dir, model_dir, protocol_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.embedding_dim[2] != 2:
        raise ValueError(
            "This clean toy script assumes the 2D branch has embedding_dim_2d: 2."
        )
    if args.embedding_dim[3] != 3:
        raise ValueError(
            "This clean toy script assumes the 3D branch has embedding_dim_3d: 3."
        )

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

    maybe_save_corruption_visualizations(
        args=args,
        mnist_train=mnist_train,
        mnist_test=mnist_test,
        train_indices=train_indices,
        val_protocol=val_protocol,
        test_protocol=test_protocol,
        figure_dir=resolve_under_out_dir(args.figure_dir[2], out_dir),
    )

    summaries: List[pd.DataFrame] = []

    # Always run 2D branch.
    summary_2d = run_mnist_branch(
        args=args,
        dim=2,
        mnist_train=mnist_train,
        mnist_test=mnist_test,
        train_indices=train_indices,
        val_protocol=val_protocol,
        test_protocol=test_protocol,
        known_classes=known_classes,
        class_to_idx=class_to_idx,
        device=device,
        out_dir=out_dir,
    )
    summaries.append(summary_2d)

    # Optional 3D branch.
    if bool(cfg_get(args.three_d, "enabled", False)):
        summary_3d = run_mnist_branch(
            args=args,
            dim=3,
            mnist_train=mnist_train,
            mnist_test=mnist_test,
            train_indices=train_indices,
            val_protocol=val_protocol,
            test_protocol=test_protocol,
            known_classes=known_classes,
            class_to_idx=class_to_idx,
            device=device,
            out_dir=out_dir,
        )
        summaries.append(summary_3d)

    # Optional controlled synthetic experiment.
    run_controlled_circle_experiment(args, out_dir)

    # Aggregate summary.
    if len(summaries) > 0:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(out_dir / "method_summary_all_mnist_branches.csv", index=False)

    with open(out_dir / "run_config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(args), f, indent=2)

    write_markdown_report(
        args=args,
        out_dir=out_dir,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        summaries=summaries,
    )

    print("\nDone.")
    print(f"All outputs saved to: {out_dir.resolve()}")
    print(
        f"2D summary: {(resolve_under_out_dir(args.result_dir[2], out_dir) / 'method_summary_2d.csv').resolve()}"
    )
    if bool(cfg_get(args.three_d, "enabled", False)):
        print(
            f"3D summary: {(resolve_under_out_dir(args.result_dir[3], out_dir) / 'method_summary_3d.csv').resolve()}"
        )


if __name__ == "__main__":
    main()
