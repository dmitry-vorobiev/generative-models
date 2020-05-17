import math

import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Any, Callable, Optional

from .ops import minibatch_stddev


def equalized_lr_init(weight, bias, scale_weights=True, lr_mult=1.0, transposed=False):
    # type: (Tensor, Tensor, Optional[bool], Optional[float], Optional[bool]) -> float
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    he_std = 1.0 / math.sqrt(fan_out if transposed else fan_in)

    if scale_weights:
        init_std = 1.0 / lr_mult
        scale = he_std * lr_mult
    else:
        init_std = he_std / lr_mult
        scale = lr_mult

    nn.init.normal_(weight, mean=0.0, std=init_std)
    if bias is not None:
        nn.init.zeros_(bias)
    return scale


class EqualizedLRLeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False, gain=math.sqrt(2)):
        super(EqualizedLRLeakyReLU, self).__init__(negative_slope, inplace)
        self.register_buffer('gain', torch.tensor(gain))

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x = x.mul_(self.gain)
        else:
            x = x * self.gain
        return F.leaky_relu(x, self.negative_slope, self.inplace)


class AddRandomNoise(nn.Module):
    def __init__(self):
        super(AddRandomNoise, self).__init__()
        self.gain = nn.Parameter(torch.empty(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gain)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        noise = x.new_empty(N, 1, H, W).normal_()
        return x + noise * self.gain


class Input(nn.Module):
    def __init__(self, channels, size=4):
        super(Input, self).__init__()
        self.weight = nn.Parameter(torch.empty(1, channels, size, size),
                                   requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, n: int) -> Tensor:
        x = self.weight.repeat(n, 1, 1, 1)
        return x


class AddBias(nn.Module):
    def __init__(self, features: int, lr_mult=1.0):
        super(AddBias, self).__init__()
        self.bias = nn.Parameter(torch.empty(features), requires_grad=True)
        self.lr_mult = lr_mult
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x.add(self.lr_mult, self.bias[:, None, None])


class EqualizedLRLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 scale_weights=True, lr_mult=1.0):
        self.scale_weights = scale_weights
        self.lr_mult = lr_mult
        self.w_mult = 1.0
        super(EqualizedLRLinear, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        self.w_mult = equalized_lr_init(
            self.weight, self.bias, self.scale_weights, self.lr_mult)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight * self.w_mult
        bias = self.bias
        if bias is not None:
            bias = bias * self.lr_mult
        return F.linear(x, weight, bias)


class EqualizedLRConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 scale_weights=True, lr_mult=1.0):
        self.scale_weights = scale_weights
        self.lr_mult = lr_mult
        self.w_mult = 1.0
        super(EqualizedLRConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode='zeros')

    def reset_parameters(self):
        self.w_mult = equalized_lr_init(
            self.weight, self.bias, self.scale_weights, self.lr_mult)

    def conv2d_forward(self, x, weight, bias):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        return F.conv2d(x, weight, bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight * self.w_mult
        bias = self.bias
        if bias is not None:
            bias = bias * self.lr_mult
        return self.conv2d_forward(x, weight, bias)


class Normalize(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        return x * norm


class ConcatLabels(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConcatLabels, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight)

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        y = self.linear(y)
        return torch.cat([z, y], dim=1)


class ConcatMiniBatchStddev(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super(ConcatMiniBatchStddev, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x: Tensor) -> Tensor:
        y = minibatch_stddev(x, group_size=self.group_size,
                             num_new_features=self.num_new_features)
        return torch.cat([x, y], dim=1)


class Lambda(nn.Module):
    def __init__(self, fn: Callable[[Any], Tensor]):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)


class Flatten(nn.Module):
    def forward(self, x: Tensor):
        return x.flatten(1)
