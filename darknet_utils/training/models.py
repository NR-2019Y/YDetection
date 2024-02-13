import torch
from torch import nn
from enum import Enum
import torchvision


def get_resnet18_backbone() -> nn.Sequential:
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    module_to_remove = ("avgpool", "fc")
    backbone = nn.Sequential()
    for name, module in resnet18.named_children():
        if name not in module_to_remove:
            backbone.add_module(name, module)
    return backbone


class _CBL(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, ksize: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, (ksize, ksize), (stride, stride), padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)


class YoloResnet18(nn.Module):
    def __init__(self, num: int, num_classes: int):
        super().__init__()
        self.backbone = get_resnet18_backbone()
        num_features = num_classes + num * 5
        self.head = nn.Sequential(
            _CBL(512, 1024, 3, 1, 1),
            _CBL(1024, 512, 3, 1, 1),
            nn.Conv2d(512, num_features, (1, 1), (1, 1), 0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1)
