import torch
import torch.nn.functional as F
from face_lib import models as mlib
import sys

sys.path.append("/app/sandbox/happy_whale/kaggle-happywhale-1st-place")


# from src.train import SphereClassifier


# class EfficientNet(torch.nn.Module):
#     def __init__(self, checkpoint_path: str, learnable: bool) -> None:
#         super().__init__()
#         self.backbone = SphereClassifier.load_from_checkpoint(
#             checkpoint_path=checkpoint_path
#         )
#         delattr(self.backbone, "head_species")
#         if learnable is False:
#             for p in self.backbone.modules():
#                 p.requires_grad = False

#     def forward(self, x):
#         bottleneck_feat = self.backbone.get_bottleneck_feature(x)
#         feats = self.backbone.backbone_head_bn(
#             self.backbone.backbone_head(bottleneck_feat)
#         )
#         feats = F.normalize(feats, p=2.0, dim=1)
#         return {"bottleneck_feature": bottleneck_feat, "feature": feats}


class ResNet(torch.nn.Module):
    def __init__(self, resnet_name: str, weights: str, learnable: bool) -> None:
        super().__init__()
        self.backbone = mlib.model_dict[resnet_name](learnable=learnable)

        if weights is not None:
            backbone_dict = torch.load(weights)
            self.backbone.load_state_dict(backbone_dict)

    def forward(self, x):
        return self.backbone(x)
