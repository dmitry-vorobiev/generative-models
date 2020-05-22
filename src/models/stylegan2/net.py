import math
import random
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List, Optional, Tuple

from .layers import AddBias, AddConstNoise, AddRandomNoise, ConcatMiniBatchStddev, Input, \
    EqualizedLRConv2d, EqualizedLRLinear, EqualizedLRLeakyReLU, Flatten, Normalize, ConcatLabels
from .mod_conv import upfirdn_2d_opt, setup_blur_weights, ModulatedConv2d, Conv2d_Downsample

try:
    from .ops.upfirdn_2d import upfirdn_2d_op_cuda
except ImportError:
    pass


def conv_lrelu(in_ch, out_ch, kernel=3, bias=True):
    # type: (int, int, Optional[int], Optional[bool]) -> List[nn.Module]
    padding = kernel // 2
    return [EqualizedLRConv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=bias),
            EqualizedLRLeakyReLU(inplace=True)]


def conv_down_torch(in_ch, out_ch, kernel=3, bias=True):
    # type: (int, int, Optional[int], Optional[bool]) -> List[nn.Module]
    padding = kernel // 2
    layers = [EqualizedLRConv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
              nn.AvgPool2d(2)]
    if bias:
        layers.append(AddBias(out_ch))
    return layers


def style_transform(in_features, out_features):
    # type: (int, int) -> EqualizedLRLinear
    lin = EqualizedLRLinear(in_features, out_features, bias=True)
    nn.init.ones_(lin.bias)
    return lin


class StyledLayer(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=False,
                 impl="ref", blur_kernel=None, noise=None):
        super(StyledLayer, self).__init__()
        self.style = style_transform(style_dim, in_channels)
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=3, upsample=upsample,
                                    impl=impl, blur_kernel=blur_kernel)
        if noise is not None:
            self.add_noise = AddConstNoise(noise)
        else:
            self.add_noise = AddRandomNoise()
        self.act_fn = EqualizedLRLeakyReLU(inplace=True)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        y = self.style(w)
        x = self.conv(x, y)
        x = self.act_fn(self.add_noise(x))
        return x


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(ToRGB, self).__init__()
        self.style = style_transform(style_dim, in_channels)
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=1, demodulate=False)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        y = self.style(w)
        x = self.conv(x, y)
        return x


class FromRGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FromRGB, self).__init__()
        self.conv = EqualizedLRConv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                      padding=0, bias=True)
        self.act_fn = EqualizedLRLeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act_fn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, impl="ref", blur_kernel=None):
        super(ResidualBlock, self).__init__()

        if impl == "ref":
            conv_layers = [Conv2d_Downsample(in_channels, out_channels, kernel_size=3,
                                             bias=True, blur_kernel=blur_kernel)]
            skip_layers = [Conv2d_Downsample(in_channels, out_channels, kernel_size=1,
                                             bias=False, blur_kernel=blur_kernel)]
        else:
            conv_layers = conv_down_torch(in_channels, out_channels, kernel=3, bias=True)
            skip_layers = conv_down_torch(in_channels, out_channels, kernel=1, bias=False)

        self.conv = nn.Sequential(
            *conv_lrelu(in_channels, in_channels, kernel=3, bias=True),
            *conv_layers,
            EqualizedLRLeakyReLU(inplace=True))
        self.skip = nn.Sequential(*skip_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x) + self.skip(x)
        return x * (1 / math.sqrt(2))


class MappingNet(nn.Module):
    r"""
    Mapping network.
    Transforms the input latent code (z) to the disentangled latent code (w).
    Used in configs B-F (Table 1).

    Args:
        latent_dim: Latent vector (Z) dimensionality.
        num_classes: Label dimensionality, 0 if no labels.
        style_dim: Disentangled latent (W) dimensionality.
        num_layers: Number of mapping layers.
        hidden_dim: Number of activations in the mapping layers.
        lr_mult: Learning rate multiplier for the mapping layers.
        normalize_latent: Normalize latent vectors (Z) before feeding them to the mapping layers?
    """
    def __init__(self, latent_dim=512, num_classes=0, style_dim=512,
                 num_layers=8, hidden_dim=512, lr_mult=0.01, normalize_latent=True):
        super(MappingNet, self).__init__()
        in_fmaps = latent_dim
        self.cat_label = None
        if num_classes > 0:
            self.cat_label = ConcatLabels(num_classes, latent_dim)
            in_fmaps = latent_dim * 2

        layers = [Normalize()] if normalize_latent else []
        features = [in_fmaps] + [hidden_dim] * (num_layers - 1) + [style_dim]
        for i in range(num_layers):
            layers += [EqualizedLRLinear(features[i], features[i + 1], lr_mult=lr_mult),
                       EqualizedLRLeakyReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, z, label=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if self.cat_label is not None:
            z = self.cat_label(z, label)
        return self.layers(z)


class SynthesisNet(nn.Module):
    r"""
    StyleGAN2 synthesis network (Figure 7).
    Implements skip connections (Figure 7), but no progressive growing.
    Used in configs E-F (Table 1).

    Args:
        img_res: Output resolution.
        img_channels: Number of output color channels.
        style_dim: Disentangled latent (W) dimensionality.
        fmap_base: Overall multiplier for the number of feature maps.
        fmap_decay: log2 feature map reduction when doubling the resolution.
        fmap_min: Minimum number of feature maps in any layer.
        fmap_max: Maximum number of feature maps in any layer.
        randomize_noise: True = randomize noise inputs every time (non-deterministic),
            False = read noise inputs from variables.
        impl: Implementation of upsample_conv ops
        blur_kernel: Low-pass filter to apply when resampling activations (only for `ref` impl).
    """
    def __init__(self, img_res=1024, img_channels=3, style_dim=512,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512,
                 randomize_noise=True, impl="ref", blur_kernel=None):
        super(SynthesisNet, self).__init__()

        if img_res <= 4:
            raise AttributeError("Image resolution must be greater than 4")

        res_log2 = int(math.log2(img_res))
        if img_res != 2 ** res_log2:
            raise AttributeError("Image resolution must be a power of 2")

        if impl not in ["torch", "ref", "cuda"]:
            raise AttributeError("impl should be one of [torch, ref, cuda]")

        if impl in ["ref", "cuda"]:
            if blur_kernel is None:
                blur_kernel = [1, 3, 3, 1]

            weight_blur = setup_blur_weights(blur_kernel, up=2, impl=impl)
            self.register_buffer("weight_blur", weight_blur)

            p = weight_blur.size(-1) - 2
            self.pad0 = (p + 1) // 2 + 1
            self.pad1 = p // 2
            self._upsample = getattr(self, "_upsample_" + impl)
        else:
            self.weight_blur = None

        self._upsample = getattr(self, "_upsample_" + impl)
        self.res_log2 = res_log2

        def nf(stage: int) -> int:
            fmaps = fmap_base / (2.0 ** (stage * fmap_decay))
            return int(min(max(fmaps, fmap_min), fmap_max))

        main = [StyledLayer(nf(1), nf(1), style_dim)]
        outs = [ToRGB(nf(1), img_channels, style_dim)]

        for res in range(1, res_log2 - 1):
            inp_ch, out_ch = nf(res), nf(res + 1)
            noise = [None, None]
            if not randomize_noise:
                size = 2 ** (2 + res)
                shape = (1, 1, size, size)
                noise = [torch.randn(*shape) for _ in range(2)]
            main += [StyledLayer(inp_ch, out_ch, style_dim, upsample=True, impl=impl,
                                 blur_kernel=lambda: self.weight_blur, noise=noise[0]),
                     StyledLayer(out_ch, out_ch, style_dim, impl=impl, noise=noise[1])]
            outs += [ToRGB(out_ch, img_channels, style_dim)]

        self.input = Input(nf(1), size=4)
        self.main = nn.ModuleList(main)
        self.outs = nn.ModuleList(outs)

    @property
    def num_layers(self) -> int:
        return len(self.main) + 1

    @staticmethod
    def _upsample_torch(x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def _upsample_ref(self, x: Tensor) -> Tensor:
        return upfirdn_2d_opt(x, self.weight_blur, up=2, pad0=self.pad0, pad1=self.pad1)

    def _upsample_cuda(self, x: Tensor) -> Tensor:
        return upfirdn_2d_op_cuda(x, self.weight_blur, up=2, pad0=self.pad0, pad1=self.pad1)

    def forward(self, w: Tensor) -> Tensor:
        x = self.input(w.size(1))
        y = None
        for i, layer in enumerate(self.main):
            x = layer(x, w[i])
            if not i % 2:
                out = self.outs[i//2]
                if not i:
                    y = out(x, w[i+1])
                else:
                    y = self._upsample(y) + out(x, w[i+1])
        return y


class Generator(nn.Module):
    r"""
    Main generator network.
    Composed of two sub-networks (mapping and synthesis).
    Used in configs B-F (Table 1).

    Args:
        img_res: Output resolution.
        img_channels: Number of output color channels.
        num_classes: Label dimensionality, 0 if no labels.
        latent_dim: Latent vector (Z) dimensionality.
        style_dim: Disentangled latent (W) dimensionality.
        fmap_base: Overall multiplier for the number of feature maps.
        fmap_decay: log2 feature map reduction when doubling the resolution.
        fmap_min: Minimum number of feature maps in any layer.
        fmap_max: Maximum number of feature maps in any layer.
        num_mapping_layers: Number of mapping layers.
        mapping_hidden_dim: Number of activations in the mapping layers.
        mapping_lr_mult: Learning rate multiplier for the mapping layers.
        normalize_latent: Normalize latent vectors (Z) before feeding them to the mapping layers?
        p_style_mix: Probability of mixing styles during training. None = disable.
        w_avg_beta: Decay for tracking the moving average of W during training. None = disable.
        truncation_psi: Style strength multiplier for the truncation trick. None = disable.
        truncation_cutoff: Number of layers for which to apply the truncation trick. None = all layers.
        randomize_noise: True = randomize noise inputs every time (non-deterministic),
            False = read noise inputs from variables.
        impl: Implementation of upsample_conv ops
        blur_kernel: Low-pass filter to apply when resampling activations (only for `ref` impl).
    """
    def __init__(self, img_res=1024, img_channels=3, num_classes=0, latent_dim=512, style_dim=512,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512, num_mapping_layers=8,
                 mapping_hidden_dim=512, mapping_lr_mult=0.01, normalize_latent=True,
                 p_style_mix=0.9, w_avg_beta=0.995, truncation_psi=0.5, truncation_cutoff=None,
                 randomize_noise=True, impl="ref", blur_kernel=None):
        super(Generator, self).__init__()

        if w_avg_beta >= 1.0 or w_avg_beta <= 0.0:
            w_avg_beta = None

        if p_style_mix <= 0.0:
            p_style_mix = None

        if truncation_psi >= 1.0:
            truncation_psi = None

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.mapping = MappingNet(
            latent_dim, num_classes, style_dim, num_mapping_layers, mapping_hidden_dim,
            mapping_lr_mult, normalize_latent)

        self.synthesis = SynthesisNet(
            img_res, img_channels, style_dim, fmap_base, fmap_decay, fmap_min, fmap_max,
            randomize_noise, impl, blur_kernel)

        self.p_style_mix = p_style_mix

        self.w_avg_beta = w_avg_beta
        self.register_buffer('w_avg', torch.zeros(style_dim))

        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

    @property
    def num_layers(self) -> int:
        return self.synthesis.num_layers

    def update_w_avg(self, w: Tensor):
        with torch.no_grad():
            self.w_avg = torch.lerp(w.mean(0), self.w_avg, self.w_avg_beta)

    def mix_styles(self, z1, w1, label=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
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

    def truncate(self, w: Tensor) -> Tensor:
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

    def forward(self, z, label=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        w = self.mapping(z, label)

        if self.training:
            if self.w_avg_beta is not None:
                self.update_w_avg(w)
            if self.p_style_mix is not None:
                w = self.mix_styles(z, w, label)
        else:
            if self.truncation_psi is not None:
                w = self.truncate(w)

        if w.ndim == 2:
            w = w.expand(self.num_layers, -1, -1)
        return self.synthesis(w), w


class Discriminator(nn.Module):
    r"""
    StyleGAN2 discriminator (Figure 7).
    Implements residual nets (Figure 7), but no progressive growing.
    Used in configs E-F (Table 1).

    Args:
        img_res: Input resolution.
        img_channels: Number of input color channels.
        num_classes: Dimensionality of the labels, 0 if no labels.
        fmap_base: Overall multiplier for the number of feature maps.
        fmap_decay: log2 feature map reduction when doubling the resolution.
        fmap_min: Minimum number of feature maps in any layer.
        fmap_max: Maximum number of feature maps in any layer.
        mbstd_group_size: Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features: Number of features for the minibatch standard deviation layer.
        impl: Implementation of conv_downsample ops
        blur_kernel: Low-pass filter to apply when resampling activations. (only for `ref` impl)
    """
    def __init__(self, img_res=1024, img_channels=3, num_classes=0, fmap_base=16 << 10,
                 fmap_decay=1.0, fmap_min=1, fmap_max=512, mbstd_group_size=4,
                 mbstd_num_features=1, impl="ref", blur_kernel=None):
        super(Discriminator, self).__init__()

        if img_res <= 4:
            raise AttributeError("Image resolution must be greater than 4")

        res_log2 = int(math.log2(img_res))
        if img_res != 2 ** res_log2:
            raise AttributeError("Image resolution must be a power of 2")

        if impl not in ["torch", "ref"]:
            raise AttributeError("impl should be one of [torch, ref]")

        self.num_classes = num_classes

        if impl == "ref":
            if blur_kernel is None:
                blur_kernel = [1, 3, 3, 1]
            weight_blur = setup_blur_weights(blur_kernel, down=2)
            self.register_buffer("weight_blur", weight_blur)
        else:
            self.weight_blur = None

        def nf(stage: int) -> int:
            fmaps = fmap_base / (2.0 ** (stage * fmap_decay))
            return int(min(max(fmaps, fmap_min), fmap_max))

        inp = FromRGB(img_channels, nf(res_log2 - 1))
        main = [ResidualBlock(nf(res - 1), nf(res - 2), impl=impl,
                              blur_kernel=lambda: self.weight_blur)
                for res in range(res_log2, 2, -1)]

        mbstd_ch = mbstd_num_features * int(mbstd_group_size > 1)
        out = [*conv_lrelu(nf(1) + mbstd_ch, nf(1)),
               Flatten(),
               EqualizedLRLinear(nf(1) * 4 ** 2, nf(0), bias=True),
               EqualizedLRLeakyReLU(inplace=True),
               EqualizedLRLinear(nf(0), max(num_classes, 1), bias=True)]
        if mbstd_ch:
            mbstd = ConcatMiniBatchStddev(mbstd_group_size, mbstd_num_features)
            out = [mbstd] + out
        self.layers = nn.Sequential(inp, *main, *out)

    def forward(self, image, label=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        x = self.layers(image)
        if label is not None:
            label = F.one_hot(label, num_classes=self.num_classes)
            x = torch.sum(x * label, dim=1, keepdim=True)
        return x
