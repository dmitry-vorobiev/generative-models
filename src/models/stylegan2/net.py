import math
import numpy as np
import random
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional, Tuple

from .layers import AddRandomNoise, ConcatMiniBatchStddev, Input, EqualConv2d, EqualLinear, \
    EqualLeakyReLU, Flatten, Normalize, ConcatLabels
from .mod_conv import ModulatedConv2d

Latent = Tensor
Label = Optional[Tensor]
DLatent = Tensor


def _upsample(x, factor):
    return F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)


def conv_lrelu(in_ch: int, out_ch: int):
    return [EqualConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            EqualLeakyReLU(inplace=True)]


def style_transform(in_features, out_features):
    lin = EqualLinear(in_features, out_features, bias=True)
    nn.init.ones_(lin.bias)
    return lin


class StyledLayer(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=False, impl="torch"):
        super(StyledLayer, self).__init__()
        self.style = style_transform(style_dim, in_channels)
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=3,
                                    upsample=upsample, upsample_impl=impl)
        self.add_noise = AddRandomNoise()
        self.act_fn = EqualLeakyReLU(inplace=True)

    def forward(self, x: Tensor, w: DLatent) -> Tensor:
        y = self.style(w)
        x = self.conv(x, y)
        x = self.act_fn(self.add_noise(x))
        return x


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(ToRGB, self).__init__()
        self.style = style_transform(style_dim, in_channels)
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=1, demodulate=False)

    def forward(self, x: Tensor, w: DLatent) -> Tensor:
        y = self.style(w)
        x = self.conv(x, y)
        return x


class FromRGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FromRGB, self).__init__()
        self.conv = EqualConv2d(in_channels, out_channels, kernel_size=1,
                                stride=1, padding=0, bias=True)
        self.act_fn = EqualLeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x) + self.down(x)
        return x * (1 / math.sqrt(2))


class MappingNet(nn.Module):
    def __init__(self, latent_dim=512, label_dim=0, style_dim=512,
                 num_layers=8, hidden_dim=512, lr_mult=0.01, normalize=True):
        super(MappingNet, self).__init__()
        in_fmaps = latent_dim
        self.embed_labels = None
        if label_dim > 0:
            self.embed_labels = ConcatLabels(label_dim, latent_dim)
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
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512, impl="ref"):
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
            main += [StyledLayer(inp_ch, out_ch, style_dim, upsample=True, impl=impl),
                     StyledLayer(out_ch, out_ch, style_dim, impl=impl)]
            outs += [ToRGB(out_ch, img_channels, style_dim)]

        self.input = Input(nf(1), size=4)
        self.main = nn.ModuleList(main)
        self.outs = nn.ModuleList(outs)

    @property
    def num_layers(self):
        return len(self.main) + 1

    def forward(self, w: DLatent) -> Tensor:
        x = self.input(w.size(1))
        y = None
        for i, layer in enumerate(self.main):
            x = layer(x, w[i])
            if not i % 2:
                out = self.outs[i//2]
                if not i:
                    y = out(x, w[i+1])
                else:
                    y = _upsample(y, 2) + out(x, w[i+1])
        return y


class Generator(nn.Module):
    def __init__(self, img_res=1024, img_channels=3, latent_dim=512, label_dim=0, style_dim=512,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512,
                 num_mapping_layers=8, mapping_hidden_dim=512, normalize_latent=True,
                 p_style_mix=0.9, w_ema_decay=0.995, truncation_psi=0.5, truncation_cutoff=None):
        super(Generator, self).__init__()
        if w_ema_decay >= 1.0 or w_ema_decay <= 0.0:
            w_ema_decay = None
        if p_style_mix <= 0.0:
            p_style_mix = None
        if truncation_psi >= 1.0:
            truncation_psi = None

        self.latent_dim = latent_dim
        self.label_dim = label_dim

        self.mapping = MappingNet(
            latent_dim, label_dim, style_dim, num_layers=num_mapping_layers,
            hidden_dim=mapping_hidden_dim, lr_mult=0.01, normalize=normalize_latent)

        self.synthesis = SynthesisNet(
            img_res, img_channels, style_dim, fmap_base, fmap_decay, fmap_min, fmap_max)

        self.p_style_mix = p_style_mix

        self.w_ema_decay = w_ema_decay
        self.register_buffer('w_avg', torch.zeros(style_dim))

        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

    @property
    def num_layers(self) -> int:
        return self.synthesis.num_layers

    def w_ema_step(self, w: Tensor):
        with torch.no_grad():
            self.w_avg = torch.lerp(w.mean(0), self.w_avg, self.w_ema_decay)

    def mix_styles(self, z1: Latent, label: Label, w1: DLatent) -> DLatent:
        num_layers = self.num_layers
        if random.uniform(0, 1) < self.p_style_mix:
            mix_cutoff = int(random.uniform(1, num_layers))
        else:
            mix_cutoff = num_layers

        z2 = torch.randn_like(z1)
        w2 = self.mapping(z2, label)
        layer_idx = torch.arange(num_layers, device=z1.device)
        mask = (layer_idx < mix_cutoff)[:, None, None]
        return torch.where(mask, w1, w2)

    def truncate(self, w: DLatent) -> DLatent:
        assert w.ndim == 2, "w: layer axis will be added by this op"
        layer_psi = torch.ones(self.num_layers, device=w.device)
        if self.truncation_cutoff is None:
            layer_psi *= self.truncation_psi
        else:
            layer_idx = torch.arange(self.num_layers, device=w.device)
            mask = layer_idx < self.truncation_cutoff
            layer_psi = torch.where(mask, layer_psi * self.truncation_psi, layer_psi)
        w = torch.lerp(self.w_avg[None, None, :], w[None, :], weight=layer_psi[:, None, None])
        return w

    def forward(self, z: Latent, label: Label = None) -> Tuple[Tensor, Tensor]:
        w = self.mapping(z, label)

        if self.training:
            if self.w_ema_decay:
                self.w_ema_step(w)
            if self.p_style_mix:
                w = self.mix_styles(z, label, w)
        else:
            if self.truncation_psi:
                w = self.truncate(w)

        if w.ndim == 2:
            w = w.expand(self.num_layers, -1, -1)
        return self.synthesis(w), w


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

    def forward(self, image: Tensor, label: Label = None) -> Tensor:
        x = self.layers(image)
        if label is not None:
            x = torch.sum(x * label, dim=1, keepdim=True)
        return x
