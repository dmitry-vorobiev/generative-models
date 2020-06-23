import torch
import torch.nn.functional as F
import torchvision

from torch import nn, Tensor
from typing import List


def pretty_print(t: Tensor) -> str:
    values = list(map(lambda x: "%.6g" % x, t.flatten().tolist()))
    return "[{}]".format(", ".join(values))


def normalize(x: Tensor, eps=1e-7) -> Tensor:
    norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + eps
    return x / norm


class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        self.layers = vgg16.features
        self.features = []

        for i in [4, 9, 16, 23, 30]:
            self.layers[i].register_forward_hook(self.save_features)

    def save_features(self, module: nn.Module, inp: Tensor, output: Tensor) -> None:
        self.features.append(output)

    def forward(self, x: Tensor) -> List[Tensor]:
        self.features = []
        _ = self.layers(x)
        return self.features

    @property
    def channels(self):
        return [64, 128, 256, 512, 512]


class ImageNorm(nn.Module):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [-0.030, -0.088, -0.188]

        if std is None:
            std = [0.458, 0.448, 0.450]

        super(ImageNorm, self).__init__()
        self.register_buffer('mean', torch.tensor(mean)[:, None, None])
        self.register_buffer('std', torch.tensor(std)[:, None, None])

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def extra_repr(self) -> str:
        return "mean={}, std={}".format(*map(pretty_print, [self.mean, self.std]))


class ReduceFeatures(nn.Sequential):
    def __init__(self, in_channels: int, out_channels=1, p_drop=0.5):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=1, padding=0, bias=False)
        layers = [nn.Dropout(), conv] if p_drop > 0.0 else [conv]
        super(ReduceFeatures, self).__init__(*layers)


class PNetLin(nn.Module):
    """
    Learned perceptual metric

    Original:
    https://github.com/richzhang/PerceptualSimilarity/blob/master/models/networks_basic.py
    """
    def __init__(self, spatial_diff=False, norm_mean=None, norm_std=None, p_drop=0.5):
        super(PNetLin, self).__init__()
        self.spatial_diff = spatial_diff
        self.norm = ImageNorm(norm_mean, norm_std)
        self.net = VGG16(pretrained=True)
        self.features = nn.ModuleList([ReduceFeatures(ch, p_drop=p_drop)
                                       for ch in self.net.channels])

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        y0 = self.net(self.norm(x0))
        y1 = self.net(self.norm(x1))
        size = x0.shape[-2:]  # H, W

        res = []
        for i, layer in enumerate(self.features):
            feat0, feat1 = list(map(normalize, [y0[i], y1[i]]))
            res_i = layer((feat0 - feat1).pow(2))

            if self.spatial_diff:
                res_i = F.interpolate(res_i, size=size, mode='bilinear', align_corners=False)
            else:
                res_i = torch.mean(res_i, dim=(2, 3), keepdim=True)
            res.append(res_i)
        return sum(res)
