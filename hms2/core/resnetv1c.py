import torch
import torch.nn as nn
import torchvision


__all__ = [
    'ResNetV1c',
    'resnetv1c50',
]


class ResNetV1c(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self._make_stem_layers(3, 64)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def _make_stem_layers(self, in_channels, stem_channels):
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnetv1c50():
    """Constructs a ResNetV1c-50 model.

    ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`.
    """
    model = ResNetV1c(torchvision.models.resnet50())

    return model
