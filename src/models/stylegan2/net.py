import math
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from .layers import EqualLeakyReLU, EqualLinear, InputNoise, \
    ModulatedConv2d, RandomGaussianNoise


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(ToRGB, self).__init__()
        self.style = EqualLinear(style_dim, in_channels, bias=True)
        nn.init.ones_(self.style.bias)

        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=1,
                                    stride=1, padding=0, demodulate=False)

    def forward(self, x, w):
        y = self.style(w)
        x = self.conv(x, y)
        return x


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, up=False):
        super(Layer, self).__init__()
        if up:
            self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upscale = None

        self.style = EqualLinear(style_dim, in_channels, bias=True)
        nn.init.ones_(self.style.bias)

        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1)
        self.add_noise = RandomGaussianNoise()
        self.act_fn = EqualLeakyReLU(inplace=True)

    def forward(self, x, w):
        if self.upscale:
            x = self.upscale(x)
        y = self.style(w)
        x = self.conv(x, y)
        x = self.act_fn(self.add_noise(x))
        return x


def upscale(x, factor):
    return F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)


class SynthesisNet(nn.Module):
    def __init__(self, img_res=1024, img_channels=3, style_dim=512,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512):
        super(SynthesisNet, self).__init__()

        if img_res <= 4:
            raise AttributeError("Image resolution must be greater than 4")

        res_log2 = int(math.log2(img_res))
        if img_res != 2 ** res_log2:
            raise AttributeError("Image resolution must be a power of 2")

        self.res_log2 = res_log2

        def nf(stage):
            fmaps = int(fmap_base / (2.0 ** (stage * fmap_decay)))
            return np.clip(fmaps, fmap_min, fmap_max)

        main = [Layer(nf(1), nf(1), style_dim)]
        outs = [ToRGB(nf(1), img_channels, style_dim)]

        for res in range(1, res_log2 - 1):
            inp_ch, out_ch = nf(res), nf(res + 1)
            main += [Layer(inp_ch, out_ch, style_dim, up=True),
                     Layer(out_ch, out_ch, style_dim)]
            outs += [ToRGB(out_ch, img_channels, style_dim)]

        self.input = InputNoise(nf(1), size=4)
        self.main = nn.ModuleList(main)
        self.outs = nn.ModuleList(outs)

    def forward(self, n):
        w = torch.rand(len(self.main) + 1, n, 512)
        x = self.input(n).clone()
        y = None

        for i, layer in enumerate(self.main):
            x = layer(x, w[i])

            if not i % 2:
                out = self.outs[i // 2]
                if not i:
                    y = out(x, w[i + 1])
                else:
                    y = upscale(y, 2) + out(x, w[i+1])

        return y
