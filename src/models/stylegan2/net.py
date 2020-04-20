import math
import numpy as np
import random
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional

from .layers import AddRandomNoise, ConcatMiniBatchStddev, Input, EqualConv2d, EqualLinear, \
    EqualLeakyReLU, Flatten, ModulatedConv2d, Normalize, EmbedLabels

Latent = Tensor
Label = Optional[Tensor]
DLatent = Tensor


def upsample(x, factor):
    return F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)


def conv_lrelu(in_ch: int, out_ch: int):
    return [EqualConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            EqualLeakyReLU(inplace=True)]


def style_transform(in_features, out_features):
    lin = EqualLinear(in_features, out_features, bias=True)
    nn.init.ones_(lin.bias)
    return lin


class StyledLayer(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, up=False):
        super(StyledLayer, self).__init__()
        if up:
            self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upscale = None

        self.style = style_transform(style_dim, in_channels)
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1)
        self.add_noise = AddRandomNoise()
        self.act_fn = EqualLeakyReLU(inplace=True)

    def forward(self, x, w):
        if self.upscale:
            x = self.upscale(x)
        y = self.style(w)
        x = self.conv(x, y)
        x = self.act_fn(self.add_noise(x))
        return x


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(ToRGB, self).__init__()
        self.style = style_transform(style_dim, in_channels)
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=1,
                                    stride=1, padding=0, demodulate=False)

    def forward(self, x, w):
        y = self.style(w)
        x = self.conv(x, y)
        return x


class FromRGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FromRGB, self).__init__()
        self.conv = EqualConv2d(in_channels, out_channels, kernel_size=1,
                                stride=1, padding=0, bias=True)
        self.act_fn = EqualLeakyReLU(inplace=True)

    def forward(self, x):
        return self.act_fn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *conv_lrelu(in_channels, in_channels),
            *conv_lrelu(in_channels, out_channels),
            nn.AvgPool2d(2))
        self.down = nn.Sequential(
            EqualConv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2))

    def forward(self, x: Tensor):
        x = self.conv(x) + self.down(x)
        return x * (1 / math.sqrt(2))


class MappingNet(nn.Module):
    def __init__(self, latent_dim=512, label_dim=0, style_dim=512,
                 num_layers=8, hidden_dim=512, lr_mult=0.01, normalize=True):
        super(MappingNet, self).__init__()
        in_fmaps = latent_dim
        self.embed_labels = None

        if label_dim > 0:
            self.embed_labels = EmbedLabels(label_dim, latent_dim)
            in_fmaps = latent_dim * 2

        layers = [Normalize()] if normalize else []
        features = [in_fmaps] + [hidden_dim] * (num_layers - 1) + [style_dim]
        for i in range(num_layers):
            layers += [EqualLinear(features[i], features[i + 1], lr_mult=lr_mult),
                       EqualLeakyReLU(inplace=True)]
        self.mapping = nn.Sequential(*layers)

    def forward(self, z: Latent, label: Label = None):
        if self.embed_labels:
            z = self.embed_labels(z, label)
        return self.mapping(z)


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

        main = [StyledLayer(nf(1), nf(1), style_dim)]
        outs = [ToRGB(nf(1), img_channels, style_dim)]

        for res in range(1, res_log2 - 1):
            inp_ch, out_ch = nf(res), nf(res + 1)
            main += [StyledLayer(inp_ch, out_ch, style_dim, up=True),
                     StyledLayer(out_ch, out_ch, style_dim)]
            outs += [ToRGB(out_ch, img_channels, style_dim)]

        self.input = Input(nf(1), size=4)
        self.main = nn.ModuleList(main)
        self.outs = nn.ModuleList(outs)

    @property
    def num_layers(self):
        return len(self.main) + 1

    def forward(self, w: DLatent):
        x = self.input(w.size(1))
        y = None
        for i, layer in enumerate(self.main):
            x = layer(x, w[i])
            if not i % 2:
                out = self.outs[i//2]
                if not i:
                    y = out(x, w[i+1])
                else:
                    y = upsample(y, 2) + out(x, w[i+1])
        return y


class Generator(nn.Module):
    def __init__(self, mapping: MappingNet, synthesis: SynthesisNet, p_style_mix=0.9):
        super(Generator, self).__init__()
        self.mapping = mapping
        self.synthesis = synthesis
        self.p_style_mix = p_style_mix

    @property
    def num_layers(self):
        return self.synthesis.num_layers

    def mix_styles(self, z1: Tensor, y: Tensor, w1: Tensor):
        num_layers = self.num_layers

        if random.uniform(0, 1) < self.p_style_mix:
            mix_cutoff = int(random.uniform(1, num_layers))
        else:
            mix_cutoff = num_layers

        z2 = torch.randn_like(z1)
        w2 = self.mapping(z2, y)
        mask = torch.arange(num_layers)[:, None, None] < mix_cutoff
        return torch.where(mask, w1, w2)

    def forward(self, z, label=None):
        w = self.mapping(z, label)
        if w.ndim == 2:
            w = w.expand(self.num_layers, -1, -1)
        if self.p_style_mix is not None:
            w = self.mix_styles(z, label, w)
        out = self.synthesis(w)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_res=1024, img_channels=3, label_dim=0,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512,
                 mbstd_group_size=4, mbstd_num_features=1):
        super(Discriminator, self).__init__()

        if img_res <= 4:
            raise AttributeError("Image resolution must be greater than 4")

        res_log2 = int(math.log2(img_res))
        if img_res != 2 ** res_log2:
            raise AttributeError("Image resolution must be a power of 2")

        def nf(stage: int):
            fmaps = int(fmap_base / (2.0 ** (stage * fmap_decay)))
            return int(np.clip(fmaps, fmap_min, fmap_max))

        inp = FromRGB(img_channels, nf(res_log2 - 1))
        main = [ResidualBlock(nf(res - 1), nf(res - 2))
                for res in range(res_log2, 2, -1)]

        mbstd_ch = mbstd_num_features * int(mbstd_group_size > 1)
        out = [*conv_lrelu(nf(1) + mbstd_ch, nf(1)),
               Flatten(),
               EqualLinear(nf(1) * 4 ** 2, nf(0), bias=True),
               EqualLeakyReLU(inplace=True),
               EqualLinear(nf(0), max(label_dim, 1), bias=True)]
        if mbstd_ch:
            mbstd = ConcatMiniBatchStddev(mbstd_group_size, mbstd_num_features)
            out = [mbstd] + out
        self.layers = nn.Sequential(inp, *main, *out)

    def forward(self, image: Tensor, label=None):
        x = self.layers(image)
        if label is not None:
            x = torch.sum(x * label, dim=1, keepdim=True)
        return x
