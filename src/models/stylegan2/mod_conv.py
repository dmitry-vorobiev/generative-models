import torch
import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import List, Sequence

from .layers import equalized_lr_init


def _upfirdn_2d_ref(x, w, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    # type: (Tensor, Tensor, int, int, int, int, int, int, int, int) -> Tensor
    N, C, H, W = x.shape
    Hk, Wk = w.shape[-2:]
    assert H > 0 and W > 0
    # left only runtime asserts

    # Upsample (insert zeros).
    if upx > 1 or upy > 1:
        x = x.view(-1, C, H, 1, W, 1)
        x = F.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
        x = x.view(-1, C, H * upy, W * upx)

    # Pad (crop if negative).
    pads = [padx0, padx1, pady0, pady1]
    if any(pads):
        x = F.pad(x, pads)

    # Convolve with filter.
    _, _, H1, W1 = x.shape
    x = x.view(-1, 1, H1, W1)
    if w.ndim == 2:
        # no need to flip since we use symmetric kernels
        w = torch.flip(w, dims=(0, 1))[None, None, :]
    x = F.conv2d(x, w, stride=1, padding=0)
    x = x.view(N, C, H1 - Hk + 1, W1 - Wk + 1)

    # Downsample (throw away pixels).
    if downx > 1 or downy > 1:
        x = x[:, :, ::downy, ::downx]
    return x


def _upfirdn_2d_opt(x, w, up=1, down=1, pad0=0, pad1=0):
    # type: (Tensor, Tensor, int, int, int, int) -> Tensor
    N, C, H, W = x.shape
    assert H > 0 and W > 0
    assert w.ndim == 4
    # left only runtime asserts

    # Upsample (insert zeros).
    if up > 1:
        x = x.view(-1, C, H, 1, W, 1)
        x = F.pad(x, [0, up - 1, 0, 0, 0, up - 1])
        x = x.view(-1, C, H * up, W * up)

        # Same results:
        # w_up = torch.ones(1, device=x.device).expand(C, 1, 1, -1)
        # x = F.conv_transpose2d(x, w_up, stride=up, output_padding=1, groups=C)

    # Pad (crop if negative).
    # for padding like (1,1), (2,2), etc. use conv2d argument
    padding = pad0
    if pad0 != pad1 or pad0 < 0 or pad1 < 0:
        x = F.pad(x, [pad0, pad1] * 2)
        padding = 0

    # Convolve with filter.
    w = w.expand(C, -1, -1, -1)
    x = F.conv2d(x, w, stride=1, padding=padding, groups=C)

    # Downsample (throw away pixels).
    if down > 1:
        x = x[:, :, ::down, ::down]
    return x


def _setup_kernel(k: Sequence[int], device=None) -> Tensor:
    k = torch.tensor(k, dtype=torch.float32, device=device)
    if k.ndim == 1:
        k = k[:, None] * k[None, :]
    k /= k.sum()
    return k


def setup_blur_weights(k: Sequence[int], up_factor=2) -> Tensor:
    if k is None:
        k = [1] * up_factor
    if not isinstance(k, list):
        raise AttributeError("lpf_kernel must be of type List or None")
    k = _setup_kernel(k) * (up_factor ** 2)
    w = torch.flip(k, dims=(0, 1))[None, None, :]
    return w


def _same(k: Sequence) -> bool:
    for i in range(len(k)-1):
        if k[i] != k[i+1]:
            return False
    return True


class ModulatedConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, demodulate=True,
                 upsample=False, upsample_impl="ref", lp_kernel=None,
                 scale_weights=True, lr_mult=1.0):
        if upsample_impl not in ["torch", "ref"]:
            raise AttributeError("impl should be one of [torch, ref]")

        if isinstance(kernel_size, Sequence):
            if not _same(kernel_size):
                raise AttributeError("Only square kernels are supported")

        kernel_size = _pair(kernel_size)
        padding = _pair(kernel_size[0] // 2)
        stride = _pair(2 if upsample else 1)
        dilation = _pair(1)

        self.demodulate = demodulate
        self.scale_weights = scale_weights
        self.lr_mult = lr_mult
        self.w_mult = 1.0

        transposed = upsample and upsample_impl == "ref"
        super(ModulatedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            transposed, _pair(0), groups=1, bias=bias, padding_mode='zeros')

        if upsample:
            if lp_kernel is None:
                lp_kernel = [1, 3, 3, 1]
            if isinstance(lp_kernel, torch.Tensor):
                C1, C0, Hb, Wb = lp_kernel.shape
                assert C1 == 1 and C0 == 1 and Hb == Wb
                # shared LPF weights, don't want to save as buffer here
                self.lp_weight = lp_kernel
            else:
                self.register_buffer("lp_weight", setup_blur_weights(lp_kernel, 2))

            Hb = len(lp_kernel)
            Hc = kernel_size[0]
            p = (Hb - 2) - (Hc - 1)
            self.pad0 = (p + 1) // 2 + 1
            self.pad1 = p // 2 + 1

            if upsample_impl == "ref":
                self._exec = self._upsample_conv2d_ref
            else:
                self._exec = self._upsample_conv2d_torch
        else:
            self._exec = self._conv2d

        self.upsample = upsample

    def reset_parameters(self):
        self.w_mult = equalized_lr_init(self.weight, self.bias, self.scale_weights,
                                        self.lr_mult, self.transposed)

    def _modulate(self, weight: Tensor, style: Tensor) -> Tensor:
        w = weight[None, :]  # batch dim
        # out channels dim: 1 - normal, 2 - transposed
        C_out = 1 + int(self.transposed)
        s = style.unsqueeze(C_out)
        s = s[:, :, :, None, None]
        return w * s

    def _demodulate(self, w: Tensor, eps=1e-8) -> Tensor:
        # in channels dim: 2 - normal, 1 - transposed
        C_in = 2 - int(self.transposed)
        d = torch.rsqrt(w.pow(2).sum(dim=(C_in, 3, 4), keepdim=True) + eps)
        return w * d

    def _upsample_conv2d_ref(self, x, w):
        # type: (Tensor, Tensor) -> Tensor
        N, C0, C1, Hc, Wc = w.shape
        w = w.view(N * C0, C1, Hc, Wc)
        x = F.conv_transpose2d(x, w, stride=2, padding=0, groups=N)
        _, _, H1, W1, = x.shape
        x = x.view(N, C1, H1, W1)
        return _upfirdn_2d_opt(x, self.lp_weight, pad0=self.pad0, pad1=self.pad1)

    def _upsample_conv2d_torch(self, x, w):
        # type: (Tensor, Tensor) -> Tensor
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self._conv2d(x, w)

    def _conv2d(self, x, w):
        # type: (Tensor, Tensor) -> Tensor
        N, C1, C0, Hc, Wc = w.shape
        w = w.view(N * C1, C0, Hc, Wc)
        x = F.conv2d(x, w, stride=1, padding=self.padding, groups=N)
        _, _, H1, W1 = x.shape
        return x.view(N, C1, H1, W1)

    def conv2d_forward(self, x, style, weight, bias):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
        w = self._modulate(weight, style)
        if self.demodulate:
            w = self._demodulate(w)

        N, C0, H0, W0 = x.shape
        x = x.view(1, N * C0, H0, W0)
        out = self._exec(x, w)
        del x, w

        if bias is not None:
            out = out + bias[:, None, None]
        return out

    def forward(self, x, style):
        weight = self.weight * self.w_mult
        bias = self.bias
        if bias is not None:
            bias = bias * self.lr_mult
        return self.conv2d_forward(x, style, weight, bias)
