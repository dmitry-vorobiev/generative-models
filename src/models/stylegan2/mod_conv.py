import torch
import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import List, Sequence, Tuple

from .layers import equalized_lr_init, EqualizedLRConv2d


def upfirdn_2d_ref(x, w, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
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


def upfirdn_2d_opt(x, w, up=1, down=1, pad0=0, pad1=0):
    # type: (Tensor, Tensor, int, int, int, int) -> Tensor
    N, C, H, W = x.shape
    assert H > 0 and W > 0
    assert w.ndim == 4
    # left only runtime asserts

    # Upsample (insert zeros).
    if up > 1:
        x = x[:, :, :, None, :, None]
        x = F.pad(x, [0, up - 1, 0, 0, 0, up - 1])
        x = x.view(N, C, H * up, W * up)

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


def setup_blur_weights(k: Sequence[int], up=0, down=0) -> Tensor:
    assert not (up and down)
    if k is None:
        k = [1] * (up or down)
    if not isinstance(k, Sequence):
        raise AttributeError("blur_kernel must be of type List, Tuple or None")
    k = _setup_kernel(k)
    if up:
        k *= (up ** 2)
    # from _upfirdn_2d_ref:
    # w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
    return torch.flip(k, dims=(0, 1))[None, None, :]


def _same(k: Sequence) -> bool:
    for i in range(len(k)-1):
        if k[i] != k[i+1]:
            return False
    return True


class BlurWeightsMixin(object):
    def _init_blur_weights(self, blur_kernel, up=0, down=0):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]
        if isinstance(blur_kernel, torch.Tensor):
            C1, C0, Hb, Wb = blur_kernel.shape
            assert C1 == 1 and C0 == 1 and Hb == Wb
            # shared LPF weights, don't want to save as buffer here
            self._weight_blur = blur_kernel
        elif hasattr(blur_kernel, "__call__"):
            # a way to get a proper reference to the shared buffer
            # after it has been moved to GPU
            self._weight_blur_func = blur_kernel
        else:
            # noinspection PyUnresolvedReferences
            self.register_buffer("_weight_blur", setup_blur_weights(blur_kernel, up=up, down=down))

    @property
    def weight_blur(self):
        if hasattr(self, '_weight_blur_func'):
            return self._weight_blur_func()
        return self._weight_blur


class ModulatedConv2d(_ConvNd, BlurWeightsMixin):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, demodulate=True,
                 upsample=False, upsample_impl="ref", blur_kernel=None,
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
        self.weight_mult = 1.0

        transposed = upsample and upsample_impl == "ref"
        super(ModulatedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            transposed, _pair(0), groups=1, bias=bias, padding_mode='zeros')

        if upsample:
            if upsample_impl == "ref":
                self._init_blur_weights(blur_kernel, up=2)
                W_blur = self.weight_blur.size(-1)
                W_conv = kernel_size[0]
                p = (W_blur - 2) - (W_conv - 1)
                self.pad0 = (p + 1) // 2 + 1
                self.pad1 = p // 2 + 1
                self._exec = self._upsample_conv2d_ref
            else:
                self._exec = self._upsample_conv2d_torch
        else:
            self._exec = self._conv2d

    def reset_parameters(self):
        self.weight_mult = equalized_lr_init(self.weight, self.bias, self.scale_weights,
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
        return upfirdn_2d_opt(x, self.weight_blur, pad0=self.pad0, pad1=self.pad1)

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

    def conv2d_forward(self, x, style, weight):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        w = self._modulate(weight, style)
        if self.demodulate:
            w = self._demodulate(w)

        N, C0, H0, W0 = x.shape
        x = x.view(1, N * C0, H0, W0)
        return self._exec(x, w)

    def forward(self, x, style):
        x = self.conv2d_forward(x, style, self.weight * self.weight_mult)
        if self.bias is not None:
            bias = self.bias * self.lr_mult
            x += bias[:, None, None]
        return x


# noinspection PyPep8Naming
class Conv2d_Downsample(nn.Module, BlurWeightsMixin):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, blur_kernel=None):
        super(Conv2d_Downsample, self).__init__()
        self.conv = EqualizedLRConv2d(in_channels, out_channels, kernel_size, stride=2, bias=bias)

        self._init_blur_weights(blur_kernel, down=2)
        W_blur = self.weight_blur.size(-1)
        W_conv = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        p = (W_blur - 2) + (W_conv - 1)
        self.pad0 = (p + 1) // 2
        self.pad1 = p // 2

    def forward(self, x):
        x = upfirdn_2d_opt(x, self.weight_blur, pad0=self.pad0, pad1=self.pad1)
        return self.conv(x)
