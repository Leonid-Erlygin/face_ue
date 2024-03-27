import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import spectral_norm
from face_lib.models import FaceModule
from face_lib import models as mlib


class SCFHead(nn.Module):
    def __init__(self, convf_dim, latent_vector_size):
        super().__init__()

        self.convf_dim = convf_dim
        self.latent_vector_size = latent_vector_size

        self._log_kappa = nn.Sequential(
            nn.Linear(self.convf_dim, self.latent_vector_size),
            nn.BatchNorm1d(self.latent_vector_size, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_vector_size, self.latent_vector_size),
            nn.BatchNorm1d(self.latent_vector_size, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_vector_size, 1),
        )

        # Trying to increase number of parameters
        # self._log_kappa = nn.Sequential(
        #     nn.Linear(self.convf_dim, self.convf_dim // 2),
        #     nn.BatchNorm1d(self.convf_dim // 2, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.convf_dim // 2, self.convf_dim // 4),
        #     nn.BatchNorm1d(self.convf_dim // 4, affine=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.convf_dim // 4, 1),
        # )

    def forward(self, convf):
        log_kappa = self._log_kappa(convf)
        log_kappa = torch.log(1e-6 + torch.exp(log_kappa))

        return log_kappa


class PFEHead(FaceModule):
    def __init__(self, in_feat=512, **kwargs):
        super(PFEHead, self).__init__(**kwargs)
        self.fc1 = nn.Linear(in_feat * 6 * 7, in_feat)
        self.bn1 = nn.BatchNorm1d(in_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_feat, in_feat)
        self.bn2 = nn.BatchNorm1d(in_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1e-4]))
        self.beta = Parameter(torch.Tensor([-7.0]))

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        x = x / x.norm(dim=-1, keepdim=True)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))
        return {"log_sigma": x}


class PFEHeadAdjustable(FaceModule):
    def __init__(self, in_feat=512, out_feat=512, **kwargs):
        super(PFEHeadAdjustable, self).__init__(**kwargs)
        self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
        self.bn1 = nn.BatchNorm1d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm1d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        x = self.relu(self.bn1(F.linear(x, F.normalize(self.fc1))))
        x = self.bn2(F.linear(x, F.normalize(self.fc2)))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return {"log_sigma": x}


class PFEHeadAdjustableLightning(FaceModule):
    def __init__(self, in_feat=512, out_feat=512, **kwargs):
        super(PFEHeadAdjustableLightning, self).__init__(**kwargs)
        self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
        self.bn1 = nn.BatchNorm1d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm1d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

    def forward(self, bottleneck_feature: torch.Tensor):
        x = self.relu(self.bn1(F.linear(bottleneck_feature, F.normalize(self.fc1))))
        x = self.bn2(F.linear(x, F.normalize(self.fc2)))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return x


class PFEHeadAdjustableSpectralSimple(FaceModule):
    def __init__(self, in_feat=512, out_feat=512, n_power_iterations=3, **kwargs):
        super(PFEHeadAdjustableSpectralSimple, self).__init__(**kwargs)

        # self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
        self.fc1 = nn.Linear(in_feat, out_feat, bias=False)
        self.bn1 = nn.BatchNorm1d(out_feat, affine=True)
        self.relu = nn.ReLU()
        # self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.fc2 = nn.Linear(out_feat, out_feat, bias=False)
        self.bn2 = nn.BatchNorm1d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.fc1 = spectral_norm(self.fc1, n_power_iterations=n_power_iterations)
        self.fc2 = spectral_norm(self.fc2, n_power_iterations=n_power_iterations)

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return {"log_sigma": x}


class ProbHead(FaceModule):
    def __init__(self, in_feat=512, **kwargs):
        super(ProbHead, self).__init__(kwargs)
        # TODO: remove hard coding here
        self.fc1 = nn.Linear(in_feat * 7 * 7, in_feat)
        self.bn1 = nn.BatchNorm1d(in_feat, affine=True)
        self.relu = nn.ReLU(in_feat)
        self.fc2 = nn.Linear(in_feat, 1)
        self.bn2 = nn.BatchNorm1d(1, affine=False)
        self.gamma = Parameter(torch.Tensor([1e-4]))
        self.beta = Parameter(torch.Tensor([-7.0]))

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))
        return {"log_sigma": x}
