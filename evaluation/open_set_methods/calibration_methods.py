import torch
import torch.nn as nn
import numpy as np
import importlib
from matplotlib import cm, ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier


def print_specific_params_table_terminal(model, iteration):
    """
    Accesses specific parameters from ExpTransform and prints them in a pandas
    DataFrame table to the terminal.
    """
    param_data = []

    # Base parameters that always exist
    params_to_track = ["T1", "T2"]

    # Conditionally add parameters based on model flags
    if model.use_alpha:
        params_to_track.extend(["alpha1", "alpha2"])

    if model.use_shift:
        params_to_track.extend(["shift1", "shift2"])

    # Iterate through the list of names and retrieve their values
    for name in params_to_track:
        # Use getattr() for safe access in case a name somehow ends up in the list incorrectly
        if hasattr(model, name):
            value = getattr(model, name).item()
            param_data.append({"Name": name, "Value": round(value, 4)})

    # Create and print the DataFrame
    df = pd.DataFrame(param_data)

    print(f"\n--- Parameter Table at Iteration {iteration} ---")
    # Using to_string to ensure clean terminal output without index column
    print(df.to_string(index=False))
    print("-" * (len(df.to_string().split("\n")[0]) + 20) + "\n")


class BoostingCalibration:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def train_calibration_parameters(self, kl_1, kl_2, true_pred_label, save_name):
        X = np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T
        y = true_pred_label
        self.X_mean_val = np.mean(X, axis=0)
        self.X_std_val = np.std(X, axis=0)
        X_norm = (X - self.X_mean_val) / self.X_std_val
        self.clf = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=1.0,
            max_depth=1,
            random_state=0,
            validation_fraction=0.05,
        ).fit(X_norm, y)

        self.draw_dencity_plot(X_norm, y, save_name)

    def apply_calibration_transform(self, kl_1, kl_2, y, save_name):
        X = np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T
        self.X_mean_test = np.mean(X, axis=0)
        self.X_std_test = np.std(X, axis=0)
        X_norm = (X - self.X_mean_test) / self.X_std_test
        self.draw_dencity_plot(X_norm, y, save_name)
        print(f"Score: {self.clf.score(X_norm, y)}")
        predictions_boosting = self.clf.predict_proba(X_norm)[:, 1]
        unc = -predictions_boosting
        return unc

    def draw_dencity_plot(self, X_norm, y, image_name):
        size = 500
        kl_1 = torch.linspace(X_norm[:, 0].min(), X_norm[:, 0].max(), size)
        kl_2 = torch.linspace(X_norm[:, 1].min(), X_norm[:, 1].max(), size)
        grid_x, grid_y = np.meshgrid(
            kl_1.cpu().numpy(), kl_2.cpu().numpy(), indexing="ij"
        )
        product = torch.cartesian_prod(kl_1, kl_2)

        predict_prob = self.clf.predict_proba(product.detach().cpu().numpy())[:, 1]
        z = np.reshape(predict_prob, (size, size)).T
        z_min, z_max = z.min(), z.max()

        fig, ax = plt.subplots()
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z_min, vmax=z_max)
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        cbar = fig.colorbar(cs)
        sns.scatterplot(
            data={
                "kl_1": X_norm[:, 0],
                "kl_2": X_norm[:, 1],
                "true_pred_label": y,
            },
            x="kl_1",
            y="kl_2",
            hue="true_pred_label",
            s=1,
            alpha=0.5,
        )
        log_dir = Path(self.log_dir) / "calibration_images_boosting"
        log_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(log_dir / f"{image_name}.png")


class IDResBlock(nn.Module):
    def __init__(self, input_size, hidden_size: int, use_bn=True):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(input_size, affine=True)

        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size, affine=True)
        self.activ = nn.Sigmoid()

        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, affine=True)

        self.mlp3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size, affine=True)

        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn0(x)
        x = self.mlp1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.activ(x)
        identity = x
        x = self.mlp2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.activ(x)
        x = self.mlp3(x)
        if self.use_bn:
            x = self.bn3(x)
        out = x + identity
        return out


class ExpTransform(nn.Module):
    def __init__(self, use_shift=False, use_alpha=False, use_T=True):
        super().__init__()
        self.use_shift = use_shift
        self.use_alpha = use_alpha
        self.use_T = use_T
        self.name = "exp"
        assert self.use_alpha or self.use_shift or self.use_T
        if self.use_T:
            self.T1 = nn.Parameter(torch.tensor(1.0))
            self.T2 = nn.Parameter(torch.tensor(1.0))
        if self.use_alpha:
            self.alpha1 = nn.Parameter(torch.tensor(1.0))
            self.alpha2 = nn.Parameter(torch.tensor(1.0))
        if self.use_shift:
            self.shift1 = nn.Parameter(torch.tensor(0.0))
            self.shift2 = nn.Parameter(torch.tensor(0.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        kl1 = x[:, 0]
        kl2 = x[:, 1]
        if self.use_shift:
            kl1 = kl1 - self.shift1
            kl2 = kl2 - self.shift2
        if self.use_T:
            kl1 = kl1 / self.T1
            kl2 = kl2 / self.T2
        if self.use_alpha:
            return self.sigmoid(
                self.alpha1 * torch.exp(kl1) + self.alpha2 * torch.exp(kl2)
            )
        else:
            return self.sigmoid(torch.exp(kl1) + torch.exp(kl2))
        # if self.use_shift:
        #     kl1 = kl1 - self.shift1
        #     kl2 = kl2 - self.shift2
        # return self.sigmoid(
        #     self.alpha1 * torch.exp(kl1 / self.T1)
        #     + self.alpha2 * torch.exp(kl2 / self.T2)
        # )


class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers, use_bn):
        super().__init__()
        num_layers = num_layers
        base_dim = hidden_size
        layers = []
        prev_dim = 2
        use_bn = use_bn
        self.name = "MLP"
        for i in range(num_layers):
            new_dim = base_dim * (i + 2) * 4
            layers.extend(
                [
                    IDResBlock(prev_dim, new_dim, use_bn),
                ]
            )
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

    def forward(self, x):
        return self.perceptron(x)


class NNcalibration:
    def __init__(
        self,
        model,
        lr,
        epochs,
        weight,
        weight_decay,
        scheduler_params,
        train_weight=True,
        normalize_kl_by_test=False,
        random_subset_size=None,
        log_dir=None,
        weight_loss_types=False,
        loss_type="CE",
        use_norm=True,
    ):
        self.device = torch.device("cuda")
        self.model = model
        self.model.to(self.device)
        self.lr = lr
        self.epochs = epochs
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

    def train_calibration_parameters(self, kl_1, kl_2, error_calc, dataset_name, far):
        self.val_ds_name = dataset_name
        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        true_pred_label = np.zeros(error_calc.is_seen.shape[0])
        true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred_label[~error_calc.is_seen] = error_calc.true_reject

        false_reject_or_ident = (
            error_calc.true_accept_false_ident
            + error_calc.false_reject_false_ident
            + error_calc.false_reject_true_ident
        )
        true_index_weight = 1
        if self.weight_loss_types:

            false_accept_count = np.sum(error_calc.false_accept)
            false_reject_or_ident_count = np.sum(false_reject_or_ident) + 500  # reg
            false_accept_weight = (
                false_accept_count + false_reject_or_ident_count
            ) / false_accept_count
            false_reject_or_ident_weight = (
                false_accept_count + false_reject_or_ident_count
            ) / false_reject_or_ident_count
            # false_accept_weight *= 1 - true_index_weight
            # false_reject_or_ident_weight *= 1 - true_index_weight
        else:
            false_accept_weight = 1
            false_reject_or_ident_weight = 1

        # save validation normalization parameters
        self.X_mean_val = torch.mean(X, dim=0)
        self.X_std_val = torch.std(X, dim=0)
        X_norm = (X - self.X_mean_val) / self.X_std_val
        y = torch.tensor(
            true_pred_label.astype("bool"), dtype=torch.float32, device=self.device
        )

        if self.weight is None:
            true_pred_ratio = y.sum() / y.shape[0]
            print(true_pred_ratio.item())
            self.weight = true_pred_ratio.item()
        # weight = torch.tensor(self.weight, device=self.device)
        if self.train_weight:
            weight = torch.nn.Parameter(
                torch.tensor(self.weight, device=self.device), requires_grad=True
            )
        else:
            weight = torch.tensor(self.weight, device=self.device)

        scheduler_params = {
            "scheduler": "OneCycleLR",
            "params": {
                "max_lr": self.scheduler_params.max_lr,
                "steps_per_epoch": self.scheduler_params.steps_per_epoch,
                "epochs": self.epochs,
                "div_factor": self.scheduler_params.div_factor,
                "final_div_factor": self.scheduler_params.final_div_factor,
            },
            "interval": "epoch",
            "frequency": 1,
        }

        if self.loss_type == "MSE":
            loss_fn = nn.MSELoss(reduce=False)
            pass
        else:
            loss_fn = nn.BCELoss(reduce=False)
        self.model.train()

        if self.random_subset_size is not None:
            weights = torch.zeros(
                int(X_norm.shape[0] * self.random_subset_size), device=self.device
            )
        else:
            pass

        if self.train_weight:
            optimizer = torch.optim.Adam(
                [*self.model.parameters()] + [weight],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                [*self.model.parameters()],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        scheduler = getattr(
            importlib.import_module("torch.optim.lr_scheduler"),
            scheduler_params["scheduler"],
        )(optimizer, **scheduler_params["params"])

        true_index = y == 1.0
        for iter in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            # sample train ds subset
            if self.random_subset_size is not None:
                indices = torch.randperm(X_norm.shape[0])[
                    : int(X_norm.shape[0] * self.random_subset_size)
                ]
                X_norm_subset = X_norm[indices]
                y_subset = y[indices]
                weights[y_subset == 1.0] = 1 - weight
                weights[y_subset == 0.0] = weight
                pred = self.model(X_norm_subset)
                loss = (loss_fn(pred, y_subset) * weights).mean()
            elif self.weight_loss_types is False:
                pred = self.model(X_norm)
                loss_element_wise = loss_fn(pred, y)
                loss = loss_element_wise[true_index].mean() * (
                    1 - torch.sigmoid(weight)
                ) + loss_element_wise[~true_index].mean() * torch.sigmoid(weight)
            else:
                pred = self.model(X_norm)
                loss_element_wise = loss_fn(pred, y)
                loss_false_accept = (
                    loss_element_wise[~error_calc.is_seen][
                        error_calc.false_accept
                    ].mean()
                    * false_accept_weight
                )
                loss_false_reject_or_ident = (
                    loss_element_wise[error_calc.is_seen][false_reject_or_ident].mean()
                    * false_reject_or_ident_weight
                )
                loss_true_ident = (
                    loss_element_wise[true_index].mean() * true_index_weight
                )
                loss = loss_false_accept + loss_false_reject_or_ident + loss_true_ident
                # loss = loss_element_wise[true_index].mean() * (
                #     1 - torch.sigmoid(weight)
                # ) + loss_element_wise[~true_index].mean() * torch.sigmoid(weight)
                # loss = (loss_fn(pred, y)* weights).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if iter % 100 == 0:
                print(
                    f"Iteration {iter}, Loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}"
                )
                # Clear previous output in notebook environments for cleaner updates
                # if self.model.__class__.__name__ == "ExpTransform":
                #     print_specific_params_table_terminal(self.model, iter)
            self.model.eval()
            pred_eval = self.model(X_norm)
            accuracy = np.mean(
                (pred_eval.detach().cpu().numpy() > 0.5) == y.cpu().numpy()
            )
            # print(
            #     f"Iteration {iter}, Loss: {loss.item()}, accuracy: {accuracy.item()}, lr: {optimizer.param_groups[0]['lr']}"
            # )
            # print(torch.sigmoid(weight).item())
        # draw probs
        self.draw_dencity_plot(
            X_norm.cpu(),
            error_calc,
            dataset_name,
            far,
            is_val=True,
            model_name=self.model.name,
        )

    def train_calibration_parameters_joint(
        self,
        kl_forward_fn,
        extra_params,
        extra_params_lr,
        error_calc,
        dataset_name,
        far,
    ):
        """
        Joint optimization of MLP calibration weights and `extra_params`
        (e.g. predict_T). `kl_forward_fn()` must return differentiable
        (kl_1, kl_2) tensors w.r.t. the current `extra_params`.
        """
        self.val_ds_name = dataset_name

        # --- Labels (same as in the non-joint method) -----------------------
        true_pred_label = np.zeros(error_calc.is_seen.shape[0])
        true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred_label[~error_calc.is_seen] = error_calc.true_reject
        false_reject_or_ident = (
            error_calc.true_accept_false_ident
            + error_calc.false_reject_false_ident
            + error_calc.false_reject_true_ident
        )

        y = torch.tensor(
            true_pred_label.astype("bool"), dtype=torch.float32, device=self.device
        )
        true_index = y == 1.0

        # --- Loss-type weights (copy from original) -------------------------
        true_index_weight = 1
        if self.weight_loss_types:
            false_accept_count = np.sum(error_calc.false_accept)
            false_reject_or_ident_count = np.sum(false_reject_or_ident) + 500
            false_accept_weight = (
                false_accept_count + false_reject_or_ident_count
            ) / false_accept_count
            false_reject_or_ident_weight = (
                false_accept_count + false_reject_or_ident_count
            ) / false_reject_or_ident_count
        else:
            false_accept_weight = 1
            false_reject_or_ident_weight = 1

        # --- Get a first KL snapshot to fit normalization stats -------------
        with torch.no_grad():
            kl1_0, kl2_0 = kl_forward_fn()
            X0 = torch.stack([kl1_0.float(), kl2_0.float()], dim=1).to(self.device)
            self.X_mean_val = X0.mean(dim=0).detach()
            self.X_std_val = X0.std(dim=0).detach().clamp_min(1e-8)

        # --- Weight for balanced BCE ----------------------------------------
        if self.weight is None:
            self.weight = (y.sum() / y.shape[0]).item()
        if self.train_weight:
            weight = torch.nn.Parameter(
                torch.tensor(self.weight, device=self.device), requires_grad=True
            )
        else:
            weight = torch.tensor(self.weight, device=self.device)

        # --- Loss -----------------------------------------------------------
        loss_fn = (
            nn.MSELoss(reduce=False)
            if self.loss_type == "MSE"
            else nn.BCELoss(reduce=False)
        )

        # --- Optimizer: MLP + weight + extra_params -------------------------
        param_groups = [
            {"params": self.model.parameters(), "lr": self.lr},
            {"params": list(extra_params), "lr": extra_params_lr, "weight_decay": 0.0},
        ]
        if self.train_weight:
            param_groups.append({"params": [weight], "lr": self.lr})

        optimizer = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)

        scheduler_params = {
            "max_lr": (
                [
                    self.scheduler_params.max_lr,
                    extra_params_lr * 10,
                    self.scheduler_params.max_lr,
                ]
                if self.train_weight
                else [self.scheduler_params.max_lr, extra_params_lr * 10]
            ),
            "steps_per_epoch": self.scheduler_params.steps_per_epoch,
            "epochs": self.epochs,
            "div_factor": self.scheduler_params.div_factor,
            "final_div_factor": self.scheduler_params.final_div_factor,
        }
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)

        for iteration in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()

            # (1) Recompute KLs under current T.
            kl1_t, kl2_t = kl_forward_fn()  # differentiable in T
            X = torch.stack([kl1_t.float(), kl2_t.float()], dim=1).to(self.device)
            X_norm = (X - self.X_mean_val) / self.X_std_val  # stats frozen after iter 0

            # (2) Forward MLP + loss (same structure as original).
            pred = self.model(X_norm)
            loss_element_wise = loss_fn(pred, y)

            if self.weight_loss_types:
                loss_fa = (
                    loss_element_wise[~error_calc.is_seen][
                        error_calc.false_accept
                    ].mean()
                    * false_accept_weight
                )
                loss_fri = (
                    loss_element_wise[error_calc.is_seen][false_reject_or_ident].mean()
                    * false_reject_or_ident_weight
                )
                loss_ti = loss_element_wise[true_index].mean() * true_index_weight
                loss = loss_fa + loss_fri + loss_ti
            else:
                w = torch.sigmoid(weight) if self.train_weight else weight
                loss = (
                    loss_element_wise[true_index].mean() * (1 - w)
                    + loss_element_wise[~true_index].mean() * w
                )

            loss.backward()

            # Guard T > 0  (simple, no reparameterization).
            optimizer.step()
            with torch.no_grad():
                for p in extra_params:
                    p.clamp_(min=1e-3)

            scheduler.step()

            if iteration % 100 == 0:
                T_now = float(extra_params[0].detach())
                print(
                    f"[joint] it {iteration}  loss {loss.item():.4f}  "
                    f"T {T_now:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}"
                )

        # --- Final diagnostics plot (reuse existing drawer) ----------------
        with torch.no_grad():
            kl1_f, kl2_f = kl_forward_fn()
            X_final = torch.stack([kl1_f.float(), kl2_f.float()], dim=1).to(self.device)
            X_norm_final = (X_final - self.X_mean_val) / self.X_std_val
            self.draw_dencity_plot(
                X_norm_final.cpu(),
                error_calc,
                dataset_name,
                far,
                is_val=True,
                model_name=self.model.name,
            )

    def apply_calibration_transform(self, kl_1, kl_2, error_calc, dataset_name, far):

        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        if self.normalize_kl_by_test:
            self.X_mean_test = torch.mean(X, dim=0)
            self.X_std_test = torch.std(X, dim=0)
            X_norm = (X - self.X_mean_test) / self.X_std_test
        else:
            X_norm = (X - self.X_mean_val) / self.X_std_val
        self.model.eval()
        predictions_perceptron = self.model(X_norm)
        self.draw_dencity_plot(
            X_norm.cpu(),
            error_calc,
            dataset_name,
            far,
            is_val=False,
            model_name=self.model.name,
        )
        unc = -predictions_perceptron.detach().cpu().numpy()
        return unc

    def draw_dencity_plot(
        self, X_norm, error_calc, dataset_name, far, is_val, model_name
    ):
        true_pred = np.zeros(error_calc.is_seen.shape[0])
        true_pred[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred[~error_calc.is_seen] = error_calc.true_reject
        false_accept = np.zeros(error_calc.is_seen.shape[0])
        false_accept[~error_calc.is_seen] = error_calc.false_accept
        false_ident_or_false_reject = np.zeros(error_calc.is_seen.shape[0])
        false_ident_or_false_reject[error_calc.is_seen] = (
            ~error_calc.true_accept_true_ident
        )

        size = 500
        kl_1 = torch.linspace(
            X_norm[:, 0].min(), X_norm[:, 0].max(), size, device=self.device
        )
        kl_2 = torch.linspace(
            X_norm[:, 1].min(), X_norm[:, 1].max(), size, device=self.device
        )
        grid_x, grid_y = np.meshgrid(
            kl_1.cpu().numpy(), kl_2.cpu().numpy(), indexing="ij"
        )
        product = torch.cartesian_prod(kl_1, kl_2)

        self.model.eval()
        predict_prob = self.model(product)
        z = np.reshape(predict_prob.detach().cpu().numpy(), (size, size))
        z_min, z_max = z.min(), z.max()

        fig, ax = plt.subplots()
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z_min, vmax=z_max)
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        cbar = fig.colorbar(cs)
        pred_kind = []
        for i in range(error_calc.is_seen.shape[0]):
            if true_pred[i]:
                pred_kind.append("no error")
            elif false_accept[i]:
                pred_kind.append("false accept")
            elif false_ident_or_false_reject[i]:
                pred_kind.append("false ident or reject")
        assert len(pred_kind) == error_calc.is_seen.shape[0]
        hue_order = ["no error", "false accept", "false ident or reject"]
        kl_data = pd.DataFrame(
            list(zip(X_norm[:, 0].numpy(), X_norm[:, 1].numpy(), pred_kind)),
            columns=["kl_1", "kl_2", "prediction kind"],
        )
        sns.scatterplot(
            data=kl_data.sort_values(
                "prediction kind", key=np.vectorize(hue_order.index)
            ),
            x="kl_1",
            y="kl_2",
            hue="prediction kind",
            hue_order=hue_order,
            s=20,
            alpha=0.5,
            edgecolor="black",  # Set the border color to black
            linewidth=1,
        )
        log_dir = (
            Path(self.log_dir) / "calibration_images" / self.val_ds_name / str(far)
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        if is_val:
            plt.savefig(log_dir / f"val_{model_name}.png", dpi=300)
        else:
            plt.savefig(log_dir / f"{dataset_name}_{model_name}.png", dpi=300)


class Standartization:
    def __init__(self, normalize_kl_by_test=False, log_dir=None, vis=True):
        self.normalize_kl_by_test = normalize_kl_by_test
        self.log_dir = log_dir
        self.vis = vis
        self.device = torch.device("cpu")

    def train_calibration_parameters(self, kl_1, kl_2, error_calc, dataset_name, far):
        self.val_ds_name = dataset_name
        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        true_pred_label = np.zeros(error_calc.is_seen.shape[0])
        true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred_label[~error_calc.is_seen] = error_calc.true_reject
        # save validation normalization parameters
        self.X_mean_val = torch.mean(X, dim=0)
        self.X_std_val = torch.std(X, dim=0)
        X_norm = (X - self.X_mean_val) / self.X_std_val

        # draw probs
        if self.vis:
            self.draw_dencity_plot(
                X_norm.cpu(), error_calc, dataset_name, far, is_val=False
            )

    def apply_calibration_transform(self, kl_1, kl_2, error_calc, dataset_name, far):

        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        if self.normalize_kl_by_test:
            self.X_mean_test = torch.mean(X, dim=0)
            self.X_std_test = torch.std(X, dim=0)
            X_norm = (X - self.X_mean_test) / self.X_std_test
        else:
            X_norm = (X - self.X_mean_val) / self.X_std_val
        if self.vis:
            self.draw_dencity_plot(
                X_norm.cpu(), error_calc, dataset_name, far, is_val=True
            )
        kl_sum = torch.sum(X_norm, dim=1)
        unc = -kl_sum.detach().cpu().numpy()
        return unc

    def draw_dencity_plot(self, X_norm, error_calc, dataset_name, far, is_val):
        true_pred = np.zeros(error_calc.is_seen.shape[0])
        true_pred[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred[~error_calc.is_seen] = error_calc.true_reject
        false_accept = np.zeros(error_calc.is_seen.shape[0])
        false_accept[~error_calc.is_seen] = error_calc.false_accept
        false_ident_or_false_reject = np.zeros(error_calc.is_seen.shape[0])
        false_ident_or_false_reject[error_calc.is_seen] = (
            ~error_calc.true_accept_true_ident
        )

        size = 500
        kl_1 = torch.linspace(
            X_norm[:, 0].min(), X_norm[:, 0].max(), size, device=self.device
        )
        kl_2 = torch.linspace(
            X_norm[:, 1].min(), X_norm[:, 1].max(), size, device=self.device
        )
        grid_x, grid_y = np.meshgrid(
            kl_1.cpu().numpy(), kl_2.cpu().numpy(), indexing="ij"
        )
        product = torch.cartesian_prod(kl_1, kl_2)

        predict_prob = torch.sum(product, dim=1)
        z = np.reshape(predict_prob.detach().cpu().numpy(), (size, size)).T
        z_min, z_max = z.min(), z.max()

        fig, ax = plt.subplots()
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z_min, vmax=z_max)
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        cbar = fig.colorbar(cs)
        pred_kind = []
        for i in range(error_calc.is_seen.shape[0]):
            if true_pred[i]:
                pred_kind.append("no error")
            elif false_accept[i]:
                pred_kind.append("false accept")
            elif false_ident_or_false_reject[i]:
                pred_kind.append("false ident or reject")
        assert len(pred_kind) == error_calc.is_seen.shape[0]
        hue_order = ["no error", "false accept", "false ident or reject"]
        kl_data = pd.DataFrame(
            list(zip(X_norm[:, 0].numpy(), X_norm[:, 1].numpy(), pred_kind)),
            columns=["kl_1", "kl_2", "prediction kind"],
        )
        sns.scatterplot(
            data=kl_data.sort_values(
                "prediction kind", key=np.vectorize(hue_order.index)
            ),
            x="kl_1",
            y="kl_2",
            hue="prediction kind",
            hue_order=hue_order,
            s=10,
            alpha=0.5,
        )
        log_dir = (
            Path(self.log_dir) / "calibration_images" / self.val_ds_name / str(far)
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        if is_val:
            plt.savefig(log_dir / f"val.png")
        else:
            plt.savefig(log_dir / f"{dataset_name}.png")
