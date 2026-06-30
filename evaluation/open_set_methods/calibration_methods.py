from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

_EPS = 1e-8


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    """Convert dict/OmegaConf-like scheduler configs into a plain dict."""
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return dict(obj)

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass

    try:
        return dict(obj)
    except Exception:
        pass

    keys = [
        "scheduler",
        "params",
        "max_lr",
        "steps_per_epoch",
        "epochs",
        "div_factor",
        "final_div_factor",
    ]
    out = {}
    for key in keys:
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
    return out


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_std(x: torch.Tensor, dim: int = 0, eps: float = _EPS) -> torch.Tensor:
    return torch.std(x, dim=dim, unbiased=False).clamp_min(eps)


def _make_x(kl_1: Any, kl_2: Any, device: torch.device) -> torch.Tensor:
    kl_1_np = _to_numpy(kl_1).reshape(-1)
    kl_2_np = _to_numpy(kl_2).reshape(-1)

    if kl_1_np.shape[0] != kl_2_np.shape[0]:
        raise ValueError(
            f"KL arrays must have the same length: "
            f"len(kl_1)={kl_1_np.shape[0]}, len(kl_2)={kl_2_np.shape[0]}"
        )

    x_np = np.stack([kl_1_np, kl_2_np], axis=1).astype(np.float32)
    return torch.tensor(x_np, dtype=torch.float32, device=device)


def _true_prediction_labels_from_error_calc(error_calc: Any) -> np.ndarray:
    """
    Build binary labels for calibration.

    Label convention:
      1 = correct OSR decision
      0 = OSR error

    `error_calc` is expected to be an instance of evaluation.metrics.FrrFarIdent
    after it has been called.
    """
    is_seen = np.asarray(error_calc.is_seen, dtype=bool)
    n = is_seen.shape[0]

    true_pred_label = np.zeros(n, dtype=bool)
    true_pred_label[is_seen] = np.asarray(error_calc.true_accept_true_ident, dtype=bool)
    true_pred_label[~is_seen] = np.asarray(error_calc.true_reject, dtype=bool)

    return true_pred_label


def _error_group_masks_from_error_calc(error_calc: Any) -> Dict[str, np.ndarray]:
    """
    Return full-length boolean masks useful for weighted calibration losses.
    """
    is_seen = np.asarray(error_calc.is_seen, dtype=bool)
    n = is_seen.shape[0]

    correct = _true_prediction_labels_from_error_calc(error_calc)
    error = ~correct

    false_accept = np.zeros(n, dtype=bool)
    false_accept[~is_seen] = np.asarray(error_calc.false_accept, dtype=bool)

    false_reject_or_ident = np.zeros(n, dtype=bool)

    seen_error = np.logical_or.reduce(
        [
            np.asarray(error_calc.true_accept_false_ident, dtype=bool),
            np.asarray(error_calc.false_reject_false_ident, dtype=bool),
            np.asarray(error_calc.false_reject_true_ident, dtype=bool),
        ]
    )
    false_reject_or_ident[is_seen] = seen_error

    return {
        "correct": correct,
        "error": error,
        "false_accept": false_accept,
        "false_reject_or_ident": false_reject_or_ident,
    }


def _to_torch_masks(
    masks: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, dtype=torch.bool, device=device)
        for key, value in masks.items()
    }


def _subset_masks(
    masks: Dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in masks.items()}


def _mean_if_nonempty(
    values: torch.Tensor, mask: torch.Tensor
) -> Optional[torch.Tensor]:
    if bool(mask.any()):
        return values[mask].mean()
    return None


# ---------------------------------------------------------------------------
# Optional boosting calibration
# ---------------------------------------------------------------------------


class BoostingCalibration:
    """
    Optional non-neural calibrator kept for compatibility.

    It maps two KL features to P(correct decision). The uncertainty returned by
    `apply_calibration_transform` is `-P(correct)`, consistent with the rest of
    the pipeline: lower uncertainty values are kept first by rejection curves.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        normalize_kl_by_test: bool = False,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        max_depth: int = 1,
        random_state: int = 0,
        vis: bool = True,
    ):
        self.log_dir = log_dir
        self.normalize_kl_by_test = normalize_kl_by_test
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.vis = vis
        self.clf = None

    def train_calibration_parameters(
        self,
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        true_pred_label: np.ndarray,
        save_name: str,
    ) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        x = np.stack(
            [np.asarray(kl_1).reshape(-1), np.asarray(kl_2).reshape(-1)], axis=1
        )
        y = np.asarray(true_pred_label).astype(bool)

        self.x_mean_val = x.mean(axis=0)
        self.x_std_val = np.maximum(x.std(axis=0), _EPS)
        x_norm = (x - self.x_mean_val) / self.x_std_val

        self.clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            validation_fraction=0.05,
        )
        self.clf.fit(x_norm, y)

        if self.vis:
            self.draw_density_plot(x_norm, y, save_name)

    def apply_calibration_transform(
        self,
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        y: Optional[np.ndarray] = None,
        save_name: str = "test",
    ) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("BoostingCalibration must be trained before use.")

        x = np.stack(
            [np.asarray(kl_1).reshape(-1), np.asarray(kl_2).reshape(-1)], axis=1
        )

        if self.normalize_kl_by_test:
            x_mean = x.mean(axis=0)
            x_std = np.maximum(x.std(axis=0), _EPS)
        else:
            x_mean = self.x_mean_val
            x_std = self.x_std_val

        x_norm = (x - x_mean) / x_std

        if self.vis and y is not None:
            self.draw_density_plot(x_norm, np.asarray(y).astype(bool), save_name)

        p_correct = self.clf.predict_proba(x_norm)[:, 1]
        return -p_correct

    def draw_density_plot(
        self, x_norm: np.ndarray, y: np.ndarray, image_name: str
    ) -> None:
        if self.log_dir is None or self.clf is None:
            return

        if sns is None:
            warnings.warn("seaborn is not available; skipping calibration plot.")
            return

        size = 300
        kl_1 = np.linspace(x_norm[:, 0].min(), x_norm[:, 0].max(), size)
        kl_2 = np.linspace(x_norm[:, 1].min(), x_norm[:, 1].max(), size)
        grid_x, grid_y = np.meshgrid(kl_1, kl_2, indexing="ij")
        product = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)

        p_correct = self.clf.predict_proba(product)[:, 1]
        z = p_correct.reshape(size, size)

        fig, ax = plt.subplots()
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z.min(), vmax=z.max())
        fig.colorbar(cs)

        sns.scatterplot(
            data=pd.DataFrame(
                {"kl_1": x_norm[:, 0], "kl_2": x_norm[:, 1], "correct": y}
            ),
            x="kl_1",
            y="kl_2",
            hue="correct",
            s=3,
            alpha=0.5,
            ax=ax,
        )

        log_dir = Path(self.log_dir) / "calibration_images_boosting"
        log_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(log_dir / f"{image_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Backward-compatible typo alias.
    def draw_dencity_plot(
        self, x_norm: np.ndarray, y: np.ndarray, image_name: str
    ) -> None:
        self.draw_density_plot(x_norm, y, image_name)


# ---------------------------------------------------------------------------
# Neural calibration models
# ---------------------------------------------------------------------------


def print_specific_params_table_terminal(model: nn.Module, iteration: int) -> None:
    """
    Print interpretable parameters of ExpTransform during optimization.
    Kept as a debugging utility.
    """
    if not isinstance(model, ExpTransform):
        return

    param_names = []
    if model.use_T:
        param_names.extend(["T1", "T2"])
    if model.use_alpha:
        param_names.extend(["alpha1", "alpha2"])
    if model.use_shift:
        param_names.extend(["shift1", "shift2"])

    rows = []
    for name in param_names:
        value = getattr(model, name)
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu())
        rows.append({"Name": name, "Value": round(float(value), 6)})

    df = pd.DataFrame(rows)
    print(f"\n--- ExpTransform parameters at iteration {iteration} ---")
    print(df.to_string(index=False))
    print("-" * 60)


class IDResBlock(nn.Module):
    """
    Small residual MLP block.

    The architecture is intentionally kept close to the original implementation
    for compatibility with previously reported HolUE calibration runs.
    """

    def __init__(self, input_size: int, hidden_size: int, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn

        self.bn0 = nn.BatchNorm1d(input_size, affine=True)

        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size, affine=True)

        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, affine=True)

        self.mlp3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size, affine=True)

        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bn:
            x = self.bn0(x)

        x = self.mlp1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.activation(x)

        identity = x

        x = self.mlp2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.activation(x)

        x = self.mlp3(x)
        if self.use_bn:
            x = self.bn3(x)

        return x + identity


class ExpTransform(nn.Module):
    """
    Interpretable calibration transform:

        sigmoid(alpha1 * exp((KL1 - shift1) / T1)
              + alpha2 * exp((KL2 - shift2) / T2))

    depending on which components are enabled.

    Parameters constrained to be positive:
      - T1, T2 > 0
      - alpha1, alpha2 >= 0

    The output is interpreted as P(correct OSR decision).
    """

    def __init__(
        self,
        use_shift: bool = False,
        use_alpha: bool = False,
        use_T: bool = True,
        min_T: float = 1e-4,
        exp_clip: float = 50.0,
    ):
        super().__init__()

        self.use_shift = use_shift
        self.use_alpha = use_alpha
        self.use_T = use_T
        self.min_T = min_T
        self.exp_clip = exp_clip
        self.name = "ExpTransform"

        if not (self.use_alpha or self.use_shift or self.use_T):
            raise ValueError("At least one of use_alpha/use_shift/use_T must be True.")

        if self.use_T:
            # raw value chosen so softplus(raw) is approximately 1.0
            raw_init = np.log(np.exp(1.0 - min_T) - 1.0)
            self.raw_T1 = nn.Parameter(torch.tensor(float(raw_init)))
            self.raw_T2 = nn.Parameter(torch.tensor(float(raw_init)))

        if self.use_alpha:
            raw_alpha_init = np.log(np.exp(1.0) - 1.0)
            self.raw_alpha1 = nn.Parameter(torch.tensor(float(raw_alpha_init)))
            self.raw_alpha2 = nn.Parameter(torch.tensor(float(raw_alpha_init)))

        if self.use_shift:
            self.shift1 = nn.Parameter(torch.tensor(0.0))
            self.shift2 = nn.Parameter(torch.tensor(0.0))

    @property
    def T1(self) -> torch.Tensor:
        if not self.use_T:
            return torch.tensor(1.0)
        return F.softplus(self.raw_T1) + self.min_T

    @property
    def T2(self) -> torch.Tensor:
        if not self.use_T:
            return torch.tensor(1.0)
        return F.softplus(self.raw_T2) + self.min_T

    @property
    def alpha1(self) -> torch.Tensor:
        if not self.use_alpha:
            return torch.tensor(1.0)
        return F.softplus(self.raw_alpha1)

    @property
    def alpha2(self) -> torch.Tensor:
        if not self.use_alpha:
            return torch.tensor(1.0)
        return F.softplus(self.raw_alpha2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kl1 = x[:, 0]
        kl2 = x[:, 1]

        if self.use_shift:
            kl1 = kl1 - self.shift1
            kl2 = kl2 - self.shift2

        if self.use_T:
            kl1 = kl1 / self.T1.to(device=x.device, dtype=x.dtype)
            kl2 = kl2 / self.T2.to(device=x.device, dtype=x.dtype)

        kl1 = torch.clamp(kl1, min=-self.exp_clip, max=self.exp_clip)
        kl2 = torch.clamp(kl2, min=-self.exp_clip, max=self.exp_clip)

        e1 = torch.exp(kl1)
        e2 = torch.exp(kl2)

        if self.use_alpha:
            a1 = self.alpha1.to(device=x.device, dtype=x.dtype)
            a2 = self.alpha2.to(device=x.device, dtype=x.dtype)
            score = a1 * e1 + a2 * e2
        else:
            score = e1 + e2

        return torch.sigmoid(score).flatten()


class MLP(nn.Module):
    """
    MLP calibrator used by HolUE.

    Input:
      [KL1_norm, KL2_norm]

    Output:
      P(correct OSR decision)
    """

    def __init__(self, hidden_size: int = 6, num_layers: int = 3, use_bn: bool = False):
        super().__init__()

        self.name = "MLP"
        base_dim = hidden_size
        prev_dim = 2
        layers = []

        for i in range(num_layers):
            new_dim = base_dim * (i + 2) * 4
            layers.append(IDResBlock(prev_dim, new_dim, use_bn=use_bn))
            prev_dim = new_dim

        if use_bn:
            layers.append(nn.BatchNorm1d(prev_dim, affine=True))

        layers.extend(
            [
                nn.Linear(prev_dim, 1),
                nn.Sigmoid(),
                nn.Flatten(start_dim=0),
            ]
        )

        self.perceptron = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.perceptron(x).flatten()


# ---------------------------------------------------------------------------
# Main neural calibrator
# ---------------------------------------------------------------------------


class NNcalibration:
    """
    Neural post-hoc calibration of HolUE KL components.

    This class learns a mapping

        (KL1, KL2) -> P(correct OSR decision)

    on a validation OSR protocol. At inference/evaluation time it returns

        uncertainty = -P(correct OSR decision),

    which is consistent with the repository convention: lower uncertainty
    values are retained first by rejection curves.

    Important fixes compared with the original version:
      - uses both correct and erroneous samples by default;
      - uses balanced BCE by default;
      - works on CPU if CUDA is unavailable;
      - clamps normalization std to avoid division by zero;
      - visualization is optional and cannot affect predictions.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        lr: float = 1e-3,
        epochs: int = 2000,
        weight: Optional[float] = None,
        weight_decay: float = 0.0,
        scheduler_params: Optional[Any] = None,
        train_weight: bool = False,
        normalize_kl_by_test: bool = False,
        random_subset_size: Optional[float] = None,
        log_dir: Optional[str] = None,
        weight_loss_types: bool = False,
        loss_type: str = "CE",
        use_norm: bool = True,
        balanced_loss: bool = True,
        device: Optional[str] = None,
        vis: bool = True,
        max_plot_points: int = 5000,
        grid_size: int = 300,
        random_state: int = 777,
        # Backward-compatible legacy arguments for old configs that created
        # NNcalibration without a nested `model`.
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        use_bn: Optional[bool] = None,
    ):
        if model is None:
            model = MLP(
                hidden_size=6 if hidden_size is None else hidden_size,
                num_layers=3 if num_layers is None else num_layers,
                use_bn=False if use_bn is None else use_bn,
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.lr = lr
        self.epochs = int(epochs)
        self.weight = weight
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params
        self.random_subset_size = random_subset_size
        self.log_dir = log_dir
        self.normalize_kl_by_test = normalize_kl_by_test
        self.train_weight = train_weight
        self.weight_loss_types = weight_loss_types
        self.loss_type = loss_type
        self.use_norm = use_norm
        self.balanced_loss = balanced_loss
        self.vis = vis
        self.max_plot_points = max_plot_points
        self.grid_size = grid_size
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

            # Hydra constructs nested model before this object. Reset here so
            # calibration does not depend on previous methods/datasets.
            def _reset(m):
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

            self.model.apply(_reset)
        self.val_ds_name = "unknown_val"

        if self.train_weight and (self.balanced_loss or self.weight_loss_types):
            warnings.warn(
                "train_weight=True is ignored because balanced_loss=True or "
                "weight_loss_types=True. Disable balanced_loss to use legacy "
                "learned class weighting.",
                RuntimeWarning,
            )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _fit_normalization(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            self.X_mean_val = x.mean(dim=0)
            self.X_std_val = _safe_std(x, dim=0)
        else:
            self.X_mean_val = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
            self.X_std_val = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)

        return (x - self.X_mean_val) / self.X_std_val

    def _apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_kl_by_test:
            mean = x.mean(dim=0)
            std = _safe_std(x, dim=0)
            return (x - mean) / std

        if not hasattr(self, "X_mean_val") or not hasattr(self, "X_std_val"):
            raise RuntimeError(
                "Calibration normalization statistics are missing. "
                "Call train_calibration_parameters before apply_calibration_transform."
            )

        return (x - self.X_mean_val) / self.X_std_val

    # ------------------------------------------------------------------
    # Optimizer/scheduler
    # ------------------------------------------------------------------

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        params = _as_plain_dict(self.scheduler_params)
        if not params:
            return None

        if "scheduler" in params and "params" in params:
            scheduler_name = params["scheduler"]
            scheduler_kwargs = dict(params["params"])
            scheduler_cls = getattr(
                importlib.import_module("torch.optim.lr_scheduler"),
                scheduler_name,
            )
            return scheduler_cls(optimizer, **scheduler_kwargs)

        # Compatibility with old configs that pass only OneCycleLR kwargs.
        scheduler_cls = torch.optim.lr_scheduler.OneCycleLR
        return scheduler_cls(
            optimizer,
            max_lr=params.get("max_lr", self.lr),
            steps_per_epoch=params.get("steps_per_epoch", 1),
            epochs=params.get("epochs", self.epochs),
            div_factor=params.get("div_factor", 10),
            final_div_factor=params.get("final_div_factor", 10),
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _loss_fn(self) -> nn.Module:
        if self.loss_type.upper() == "MSE":
            return nn.MSELoss(reduction="none")
        if self.loss_type.upper() in {"CE", "BCE", "BCELoss".upper()}:
            return nn.BCELoss(reduction="none")
        raise ValueError(f"Unknown loss_type={self.loss_type!r}")

    def _compute_weighted_loss(
        self,
        loss_elementwise: torch.Tensor,
        y: torch.Tensor,
        masks: Dict[str, torch.Tensor],
        legacy_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute calibration loss.

        Default: balanced binary loss over correct/error groups.
        """
        if self.weight_loss_types:
            components = []

            correct_loss = _mean_if_nonempty(loss_elementwise, masks["correct"])
            false_accept_loss = _mean_if_nonempty(
                loss_elementwise, masks["false_accept"]
            )
            false_reject_or_ident_loss = _mean_if_nonempty(
                loss_elementwise,
                masks["false_reject_or_ident"],
            )

            for component in [
                correct_loss,
                false_accept_loss,
                false_reject_or_ident_loss,
            ]:
                if component is not None:
                    components.append(component)

            if not components:
                return loss_elementwise.mean()

            return torch.stack(components).mean()

        if self.balanced_loss:
            # correct_mask = y > 0.5
            # error_mask = ~correct_mask

            components = []
            correct_loss = _mean_if_nonempty(loss_elementwise, masks["correct"])
            error_loss = _mean_if_nonempty(loss_elementwise, masks["error"])

            if correct_loss is not None:
                components.append(correct_loss)
            if error_loss is not None:
                components.append(error_loss)

            if not components:
                return loss_elementwise.mean()

            return torch.stack(components).mean()

        # Legacy behavior: `weight` controls the relative contribution of errors.

        correct_loss = loss_elementwise[masks["correct"]].sum()
        error_loss = loss_elementwise[masks["error"]].sum()

        if correct_loss is None and error_loss is None:
            return loss_elementwise.mean()
        if correct_loss is None:
            return error_loss  # type: ignore[return-value]
        if error_loss is None:
            return correct_loss

        if legacy_weight is None:
            w_error = torch.tensor(
                0.5 if self.weight is None else float(self.weight),
                dtype=loss_elementwise.dtype,
                device=loss_elementwise.device,
            )
        else:
            w_error = torch.sigmoid(legacy_weight)

        return (
            correct_loss * (1.0 - w_error) + error_loss * w_error
        ) / loss_elementwise.shape[0]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _subset_size(self, n: int) -> int:
        if self.random_subset_size is None:
            return n

        if self.random_subset_size <= 0:
            raise ValueError("random_subset_size must be positive.")

        if self.random_subset_size <= 1:
            return max(1, int(round(n * float(self.random_subset_size))))

        return min(n, int(self.random_subset_size))

    def train_calibration_parameters(
        self,
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        error_calc: Any,
        dataset_name: str,
        far: float,
    ) -> None:
        self.val_ds_name = str(dataset_name)

        x = _make_x(kl_1, kl_2, device=self.device)
        x_norm = self._fit_normalization(x)

        y_np = _true_prediction_labels_from_error_calc(error_calc)
        y = torch.tensor(
            y_np.astype(np.float32), dtype=torch.float32, device=self.device
        )

        masks_np = _error_group_masks_from_error_calc(error_calc)
        masks = _to_torch_masks(masks_np, self.device)

        loss_fn = self._loss_fn()

        legacy_weight_param = None
        if self.train_weight and not self.balanced_loss and not self.weight_loss_types:
            init_weight = 0.5 if self.weight is None else float(self.weight)
            legacy_weight_param = torch.nn.Parameter(
                torch.tensor(init_weight, dtype=torch.float32, device=self.device),
                requires_grad=True,
            )

        params = list(self.model.parameters())
        if legacy_weight_param is not None:
            params.append(legacy_weight_param)

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = self._build_scheduler(optimizer)

        n = x_norm.shape[0]
        subset_size = self._subset_size(n)

        self.model.train()
        for iteration in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)

            if subset_size < n:
                indices = torch.randperm(n, device=self.device)[:subset_size]
                x_batch = x_norm[indices]
                y_batch = y[indices]
                masks_batch = _subset_masks(masks, indices)
            else:
                x_batch = x_norm
                y_batch = y
                masks_batch = masks

            pred = self.model(x_batch).flatten().clamp(1e-6, 1.0 - 1e-6)
            loss_elementwise = loss_fn(pred, y_batch).flatten()
            loss = self._compute_weighted_loss(
                loss_elementwise,
                y_batch,
                masks_batch,
                legacy_weight=legacy_weight_param,
            )

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if iteration % 100 == 0 or iteration == self.epochs - 1:
                with torch.no_grad():
                    self.model.eval()
                    pred_eval = self.model(x_norm).flatten()
                    acc = ((pred_eval > 0.5) == (y > 0.5)).float().mean().item()
                    self.model.train()

                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[NNcalibration] iter={iteration:04d} "
                    f"loss={loss.item():.6f} acc={acc:.4f} lr={lr_now:.3e}"
                )
                print_specific_params_table_terminal(self.model, iteration)

        self.model.eval()
        self._maybe_draw_density_plot(
            x_norm.detach().cpu(),
            error_calc,
            dataset_name=dataset_name,
            far=far,
            is_val=True,
            model_name=getattr(self.model, "name", self.model.__class__.__name__),
        )

    def train_calibration_parameters_joint(
        self,
        kl_forward_fn,
        extra_params: Sequence[torch.nn.Parameter],
        extra_params_lr: float,
        error_calc: Any,
        dataset_name: str,
        far: float,
    ) -> None:
        """
        Jointly optimize the calibration model and differentiable extra parameters
        such as HolUE predict_T.

        `kl_forward_fn()` must return `(kl_1, kl_2)` as tensors whose graph
        includes `extra_params`.
        """
        self.val_ds_name = str(dataset_name)

        y_np = _true_prediction_labels_from_error_calc(error_calc)
        y = torch.tensor(
            y_np.astype(np.float32), dtype=torch.float32, device=self.device
        )

        masks_np = _error_group_masks_from_error_calc(error_calc)
        masks = _to_torch_masks(masks_np, self.device)

        with torch.no_grad():
            kl1_0, kl2_0 = kl_forward_fn()
            x0 = torch.stack(
                [
                    kl1_0.detach().float().to(self.device),
                    kl2_0.detach().float().to(self.device),
                ],
                dim=1,
            )

            if self.use_norm:
                self.X_mean_val = x0.mean(dim=0)
                self.X_std_val = _safe_std(x0, dim=0)
            else:
                self.X_mean_val = torch.zeros(2, device=self.device)
                self.X_std_val = torch.ones(2, device=self.device)

        loss_fn = self._loss_fn()

        extra_params = list(extra_params)

        legacy_weight_param = None
        if self.train_weight and not self.balanced_loss and not self.weight_loss_types:
            init_weight = 0.5 if self.weight is None else float(self.weight)
            legacy_weight_param = torch.nn.Parameter(
                torch.tensor(init_weight, dtype=torch.float32, device=self.device),
                requires_grad=True,
            )

        param_groups = [
            {
                "params": list(self.model.parameters()),
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
        ]

        if extra_params:
            param_groups.append(
                {
                    "params": extra_params,
                    "lr": extra_params_lr,
                    "weight_decay": 0.0,
                }
            )

        if legacy_weight_param is not None:
            param_groups.append(
                {
                    "params": [legacy_weight_param],
                    "lr": self.lr,
                    "weight_decay": 0.0,
                }
            )

        optimizer = torch.optim.Adam(param_groups)

        # A robust OneCycleLR setup for several parameter groups.
        scheduler = None
        sp = _as_plain_dict(self.scheduler_params)
        if sp:
            max_lr_base = float(sp.get("max_lr", self.lr))
            max_lr = []
            for group in optimizer.param_groups:
                if group["lr"] == extra_params_lr:
                    max_lr.append(float(extra_params_lr) * 10.0)
                else:
                    max_lr.append(max_lr_base)

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=int(sp.get("steps_per_epoch", 1)),
                epochs=self.epochs,
                div_factor=float(sp.get("div_factor", 10)),
                final_div_factor=float(sp.get("final_div_factor", 10)),
            )

        self.model.train()

        for iteration in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)

            kl1_t, kl2_t = kl_forward_fn()
            x = torch.stack(
                [
                    kl1_t.float().to(self.device),
                    kl2_t.float().to(self.device),
                ],
                dim=1,
            )
            x_norm = (x - self.X_mean_val) / self.X_std_val

            pred = self.model(x_norm).flatten().clamp(1e-6, 1.0 - 1e-6)
            loss_elementwise = loss_fn(pred, y).flatten()
            loss = self._compute_weighted_loss(
                loss_elementwise,
                y,
                masks,
                legacy_weight=legacy_weight_param,
            )

            loss.backward()
            optimizer.step()

            # If extra parameter is a direct temperature, keep it positive.
            with torch.no_grad():
                for param in extra_params:
                    param.clamp_(min=1e-4)

            if scheduler is not None:
                scheduler.step()

            if iteration % 100 == 0 or iteration == self.epochs - 1:
                extra_values = [
                    float(p.detach().cpu().reshape(-1)[0]) for p in extra_params
                ]
                print(
                    f"[NNcalibration:joint] iter={iteration:04d} "
                    f"loss={loss.item():.6f} extra={extra_values}"
                )

        with torch.no_grad():
            kl1_f, kl2_f = kl_forward_fn()
            x_final = torch.stack(
                [
                    kl1_f.float().to(self.device),
                    kl2_f.float().to(self.device),
                ],
                dim=1,
            )
            x_norm_final = (x_final - self.X_mean_val) / self.X_std_val

        self.model.eval()
        self._maybe_draw_density_plot(
            x_norm_final.detach().cpu(),
            error_calc,
            dataset_name=dataset_name,
            far=far,
            is_val=True,
            model_name=getattr(self.model, "name", self.model.__class__.__name__),
        )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply_calibration_transform(
        self,
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        error_calc: Optional[Any] = None,
        dataset_name: str = "test",
        far: Optional[float] = None,
    ) -> np.ndarray:
        x = _make_x(kl_1, kl_2, device=self.device)
        x_norm = self._apply_normalization(x)

        self.model.eval()
        with torch.no_grad():
            p_correct = self.model(x_norm).flatten()

        if error_calc is not None:
            self._maybe_draw_density_plot(
                x_norm.detach().cpu(),
                error_calc,
                dataset_name=dataset_name,
                far=far,
                is_val=False,
                model_name=getattr(self.model, "name", self.model.__class__.__name__),
            )

        uncertainty = -p_correct.detach().cpu().numpy()
        return uncertainty

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _maybe_draw_density_plot(
        self,
        x_norm: torch.Tensor,
        error_calc: Any,
        dataset_name: str,
        far: Optional[float],
        is_val: bool,
        model_name: str,
    ) -> None:
        if not self.vis or self.log_dir is None:
            return

        try:
            self.draw_density_plot(
                x_norm,
                error_calc,
                dataset_name=dataset_name,
                far=far,
                is_val=is_val,
                model_name=model_name,
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to draw calibration density plot: {exc}")

    def draw_density_plot(
        self,
        x_norm: torch.Tensor,
        error_calc: Any,
        dataset_name: str,
        far: Optional[float],
        is_val: bool,
        model_name: str,
    ) -> None:
        if self.log_dir is None:
            return

        if sns is None:
            warnings.warn("seaborn is not available; skipping calibration plot.")
            return

        x_np = _to_numpy(x_norm)
        n = x_np.shape[0]

        labels = _true_prediction_labels_from_error_calc(error_calc)
        masks = _error_group_masks_from_error_calc(error_calc)

        pred_kind = np.empty(n, dtype=object)
        pred_kind[masks["correct"]] = "no error"
        pred_kind[masks["false_accept"]] = "false accept"
        pred_kind[masks["false_reject_or_ident"]] = "false ident or reject"

        # Fallback for pathological masks.
        pred_kind[pred_kind == None] = "error"  # noqa: E711

        # Limit scatter points for readability and speed.
        if self.max_plot_points is not None and n > self.max_plot_points:
            rng = np.random.default_rng(0)
            keep = rng.choice(n, size=self.max_plot_points, replace=False)
            x_plot = x_np[keep]
            pred_kind_plot = pred_kind[keep]
        else:
            x_plot = x_np
            pred_kind_plot = pred_kind

        size = self.grid_size
        x1 = torch.linspace(
            float(x_np[:, 0].min()),
            float(x_np[:, 0].max()),
            size,
            device=self.device,
        )
        x2 = torch.linspace(
            float(x_np[:, 1].min()),
            float(x_np[:, 1].max()),
            size,
            device=self.device,
        )

        grid_x, grid_y = np.meshgrid(
            x1.detach().cpu().numpy(),
            x2.detach().cpu().numpy(),
            indexing="ij",
        )

        product = torch.cartesian_prod(x1, x2)

        self.model.eval()
        with torch.no_grad():
            p_correct = self.model(product).detach().cpu().numpy()

        z = p_correct.reshape(size, size)
        z_min, z_max = float(z.min()), float(z.max())

        fig, ax = plt.subplots(figsize=(7, 6))
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z_min, vmax=z_max)
        fig.colorbar(cs, ax=ax, label="P(correct)")

        hue_order = ["no error", "false accept", "false ident or reject", "error"]

        kl_data = pd.DataFrame(
            {
                "kl_1": x_plot[:, 0],
                "kl_2": x_plot[:, 1],
                "prediction kind": pred_kind_plot,
            }
        )
        kl_data["prediction kind"] = pd.Categorical(
            kl_data["prediction kind"],
            categories=hue_order,
            ordered=True,
        )

        sns.scatterplot(
            data=kl_data.sort_values("prediction kind"),
            x="kl_1",
            y="kl_2",
            hue="prediction kind",
            hue_order=hue_order,
            s=15,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.3,
            ax=ax,
        )

        ax.set_xlabel("KL1 (normalized)")
        ax.set_ylabel("KL2 (normalized)")
        ax.set_title(f"{model_name} calibration: {dataset_name}, FPIR={far}")

        log_dir = (
            Path(self.log_dir) / "calibration_images" / self.val_ds_name / str(far)
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        prefix = "val" if is_val else str(dataset_name)
        fig.savefig(
            log_dir / f"{prefix}_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Backward-compatible typo alias.
    def draw_dencity_plot(
        self,
        x_norm: torch.Tensor,
        error_calc: Any,
        dataset_name: str,
        far: Optional[float],
        is_val: bool,
        model_name: str,
    ) -> None:
        self.draw_density_plot(
            x_norm, error_calc, dataset_name, far, is_val, model_name
        )


# ---------------------------------------------------------------------------
# Simple standardization/sum calibrator
# ---------------------------------------------------------------------------


class Standartization:
    """
    Backward-compatible class name preserving the original typo.

    It computes

        uncertainty = -(KL1_norm + KL2_norm)

    so that larger normalized KL sum means higher confidence and therefore
    lower uncertainty. This class is useful for ablation.
    """

    def __init__(
        self,
        normalize_kl_by_test: bool = False,
        log_dir: Optional[str] = None,
        vis: bool = True,
        use_norm: bool = True,
        max_plot_points: int = 5000,
        grid_size: int = 300,
    ):
        self.normalize_kl_by_test = normalize_kl_by_test
        self.log_dir = log_dir
        self.vis = vis
        self.use_norm = use_norm
        self.max_plot_points = max_plot_points
        self.grid_size = grid_size
        self.device = torch.device("cpu")
        self.val_ds_name = "unknown_val"

    def train_calibration_parameters(
        self,
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        error_calc: Any,
        dataset_name: str,
        far: float,
    ) -> None:
        self.val_ds_name = str(dataset_name)

        x = _make_x(kl_1, kl_2, device=self.device)

        if self.use_norm:
            self.X_mean_val = x.mean(dim=0)
            self.X_std_val = _safe_std(x, dim=0)
        else:
            self.X_mean_val = torch.zeros(2, dtype=x.dtype)
            self.X_std_val = torch.ones(2, dtype=x.dtype)

        x_norm = (x - self.X_mean_val) / self.X_std_val

        if self.vis and self.log_dir is not None:
            self._maybe_draw_density_plot(
                x_norm,
                error_calc,
                dataset_name=dataset_name,
                far=far,
                is_val=True,
            )

    def apply_calibration_transform(
        self,
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        error_calc: Optional[Any] = None,
        dataset_name: str = "test",
        far: Optional[float] = None,
    ) -> np.ndarray:
        x = _make_x(kl_1, kl_2, device=self.device)

        if self.normalize_kl_by_test:
            mean = x.mean(dim=0)
            std = _safe_std(x, dim=0)
        else:
            if not hasattr(self, "X_mean_val") or not hasattr(self, "X_std_val"):
                raise RuntimeError(
                    "Standartization must be trained before use unless "
                    "normalize_kl_by_test=True."
                )
            mean = self.X_mean_val
            std = self.X_std_val

        x_norm = (x - mean) / std

        if self.vis and self.log_dir is not None and error_calc is not None:
            self._maybe_draw_density_plot(
                x_norm,
                error_calc,
                dataset_name=dataset_name,
                far=far,
                is_val=False,
            )

        kl_sum = torch.sum(x_norm, dim=1)
        return -kl_sum.detach().cpu().numpy()

    def _maybe_draw_density_plot(
        self,
        x_norm: torch.Tensor,
        error_calc: Any,
        dataset_name: str,
        far: Optional[float],
        is_val: bool,
    ) -> None:
        try:
            self.draw_density_plot(
                x_norm=x_norm,
                error_calc=error_calc,
                dataset_name=dataset_name,
                far=far,
                is_val=is_val,
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to draw standardization plot: {exc}")

    def draw_density_plot(
        self,
        x_norm: torch.Tensor,
        error_calc: Any,
        dataset_name: str,
        far: Optional[float],
        is_val: bool,
    ) -> None:
        if self.log_dir is None:
            return

        if sns is None:
            warnings.warn("seaborn is not available; skipping calibration plot.")
            return

        x_np = _to_numpy(x_norm)
        n = x_np.shape[0]

        masks = _error_group_masks_from_error_calc(error_calc)

        pred_kind = np.empty(n, dtype=object)
        pred_kind[masks["correct"]] = "no error"
        pred_kind[masks["false_accept"]] = "false accept"
        pred_kind[masks["false_reject_or_ident"]] = "false ident or reject"
        pred_kind[pred_kind == None] = "error"  # noqa: E711

        if self.max_plot_points is not None and n > self.max_plot_points:
            rng = np.random.default_rng(0)
            keep = rng.choice(n, size=self.max_plot_points, replace=False)
            x_plot = x_np[keep]
            pred_kind_plot = pred_kind[keep]
        else:
            x_plot = x_np
            pred_kind_plot = pred_kind

        size = self.grid_size
        x1 = torch.linspace(float(x_np[:, 0].min()), float(x_np[:, 0].max()), size)
        x2 = torch.linspace(float(x_np[:, 1].min()), float(x_np[:, 1].max()), size)

        grid_x, grid_y = np.meshgrid(x1.numpy(), x2.numpy(), indexing="ij")
        product = torch.cartesian_prod(x1, x2)
        score = torch.sum(product, dim=1).numpy().reshape(size, size)

        fig, ax = plt.subplots(figsize=(7, 6))
        cs = ax.contourf(
            grid_x,
            grid_y,
            score,
            cmap=cm.PuBu_r,
            vmin=float(score.min()),
            vmax=float(score.max()),
        )
        fig.colorbar(cs, ax=ax, label="KL1_norm + KL2_norm")

        hue_order = ["no error", "false accept", "false ident or reject", "error"]

        kl_data = pd.DataFrame(
            {
                "kl_1": x_plot[:, 0],
                "kl_2": x_plot[:, 1],
                "prediction kind": pred_kind_plot,
            }
        )
        kl_data["prediction kind"] = pd.Categorical(
            kl_data["prediction kind"],
            categories=hue_order,
            ordered=True,
        )

        sns.scatterplot(
            data=kl_data.sort_values("prediction kind"),
            x="kl_1",
            y="kl_2",
            hue="prediction kind",
            hue_order=hue_order,
            s=10,
            alpha=0.5,
            ax=ax,
        )

        ax.set_xlabel("KL1 (normalized)")
        ax.set_ylabel("KL2 (normalized)")
        ax.set_title(f"KL sum: {dataset_name}, FPIR={far}")

        log_dir = (
            Path(self.log_dir) / "calibration_images" / self.val_ds_name / str(far)
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        prefix = "val" if is_val else str(dataset_name)
        fig.savefig(
            log_dir / f"{prefix}_standardization.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    # Backward-compatible typo alias.
    def draw_dencity_plot(
        self,
        x_norm: torch.Tensor,
        error_calc: Any,
        dataset_name: str,
        far: Optional[float],
        is_val: bool,
    ) -> None:
        self.draw_density_plot(x_norm, error_calc, dataset_name, far, is_val)

class ScalarRiskCalibration:
    """
    Monotone scalar calibration for already meaningful risk scores.

    This calibrator is intended for MPRisk.

    Input convention:
      - kl_1 is interpreted as a scalar risk score.
      - kl_2 is ignored and kept only for API compatibility.

    It learns

        P(error | risk) = sigmoid(b + a * standardized_risk),

    with a >= 0.

    Therefore the calibrated uncertainty preserves the raw MPRisk ranking.
    The returned value is

        uncertainty = P(error) - 1 = -P(correct),

    which matches the repository convention used by NNcalibration:
    smaller values are more confident and are kept first by rejection curves.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        epochs: int = 1000,
        weight_decay: float = 0.0,
        normalize_kl_by_test: bool = False,
        use_norm: bool = True,
        balanced_loss: bool = True,
        min_slope: float = 1e-6,
        log_dir: Optional[str] = None,
        random_state: int = 777,
        verbose: bool = True,
    ):
        self.lr = lr
        self.epochs = int(epochs)
        self.weight_decay = weight_decay
        self.normalize_kl_by_test = normalize_kl_by_test
        self.use_norm = use_norm
        self.balanced_loss = balanced_loss
        self.min_slope = min_slope
        self.log_dir = log_dir
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device("cpu")
        self.is_trained = False

    def _prepare_risk(self, risk: np.ndarray) -> np.ndarray:
        risk = np.asarray(risk, dtype=np.float64).reshape(-1)
        risk = np.nan_to_num(
            risk,
            nan=0.0,
            posinf=np.finfo(np.float64).max / 100.0,
            neginf=np.finfo(np.float64).min / 100.0,
        )
        return risk

    def _fit_normalization(self, risk: np.ndarray) -> np.ndarray:
        if self.use_norm:
            self.risk_mean_val = float(np.mean(risk))
            self.risk_std_val = float(max(np.std(risk), _EPS))
        else:
            self.risk_mean_val = 0.0
            self.risk_std_val = 1.0

        return (risk - self.risk_mean_val) / self.risk_std_val

    def _apply_normalization(self, risk: np.ndarray) -> np.ndarray:
        if self.normalize_kl_by_test:
            mean = float(np.mean(risk))
            std = float(max(np.std(risk), _EPS))
            return (risk - mean) / std

        if not hasattr(self, "risk_mean_val") or not hasattr(self, "risk_std_val"):
            raise RuntimeError(
                "ScalarRiskCalibration must be trained before use."
            )

        return (risk - self.risk_mean_val) / self.risk_std_val

    def train_calibration_parameters(
        self,
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        error_calc: Any,
        dataset_name: str,
        far: float,
    ) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        risk = self._prepare_risk(kl_1)
        x_np = self._fit_normalization(risk)

        correct_np = _true_prediction_labels_from_error_calc(error_calc)
        y_error_np = (~correct_np).astype(np.float64)

        if len(np.unique(y_error_np)) < 2:
            # Degenerate validation set. Use monotone identity-like calibration.
            self.slope_ = 1.0
            rate = float(np.clip(np.mean(y_error_np), 1e-4, 1.0 - 1e-4))
            self.bias_ = float(np.log(rate / (1.0 - rate)))
            self.is_trained = True
            return

        x = torch.tensor(x_np, dtype=torch.float64, device=self.device)
        y = torch.tensor(y_error_np, dtype=torch.float64, device=self.device)

        # Positive slope parameterization.
        raw_slope_init = np.log(np.exp(1.0 - self.min_slope) - 1.0)
        raw_slope = torch.nn.Parameter(
            torch.tensor(raw_slope_init, dtype=torch.float64, device=self.device)
        )

        error_rate = float(np.clip(np.mean(y_error_np), 1e-4, 1.0 - 1e-4))
        bias = torch.nn.Parameter(
            torch.tensor(
                np.log(error_rate / (1.0 - error_rate)),
                dtype=torch.float64,
                device=self.device,
            )
        )

        params = [raw_slope, bias]
        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.balanced_loss:
            n_pos = float(np.sum(y_error_np == 1))
            n_neg = float(np.sum(y_error_np == 0))

            weights_np = np.zeros_like(y_error_np, dtype=np.float64)
            if n_pos > 0:
                weights_np[y_error_np == 1] = 0.5 / n_pos
            if n_neg > 0:
                weights_np[y_error_np == 0] = 0.5 / n_neg

            weights = torch.tensor(weights_np, dtype=torch.float64, device=self.device)
        else:
            weights = torch.ones_like(y) / max(float(len(y_error_np)), 1.0)

        for iteration in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)

            slope = F.softplus(raw_slope) + self.min_slope
            logits = bias + slope * x

            loss_elementwise = F.binary_cross_entropy_with_logits(
                logits,
                y,
                reduction="none",
            )
            loss = torch.sum(loss_elementwise * weights)

            loss.backward()
            optimizer.step()

            if self.verbose and (iteration % 100 == 0 or iteration == self.epochs - 1):
                with torch.no_grad():
                    p_error = torch.sigmoid(logits)
                    pred = p_error > 0.5
                    acc = (pred == (y > 0.5)).double().mean().item()
                    print(
                        f"[ScalarRiskCalibration] iter={iteration:04d} "
                        f"loss={loss.item():.6f} acc={acc:.4f} "
                        f"slope={slope.item():.6f} bias={bias.item():.6f}"
                    )

        with torch.no_grad():
            self.slope_ = float((F.softplus(raw_slope) + self.min_slope).cpu())
            self.bias_ = float(bias.cpu())

        self.is_trained = True

    def apply_calibration_transform(
        self,
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        error_calc: Optional[Any] = None,
        dataset_name: str = "test",
        far: Optional[float] = None,
    ) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(
                "ScalarRiskCalibration must be trained before use."
            )

        risk = self._prepare_risk(kl_1)
        x = self._apply_normalization(risk)

        logits = self.bias_ + self.slope_ * x
        logits = np.clip(logits, -50.0, 50.0)

        p_error = 1.0 / (1.0 + np.exp(-logits))

        # Repository convention:
        # predicted_unc is sorted ascending; smaller = more confident.
        # Returning -P(correct) = P(error)-1 keeps this convention and also
        # makes CalibrationPlot compute predicted_conf = -predicted_unc.
        return p_error - 1.0
    
# Correctly spelled alias for new code.
Standardization = Standartization
