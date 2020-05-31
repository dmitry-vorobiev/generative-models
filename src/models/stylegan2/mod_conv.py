import torch
import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import Sequence

from .layers import equalized_lr_init, EqualizedLRConv2d
from .ops import upfirdn_2d_cuda, upfirdn_2d_opt


def _setup_kernel(k: Sequence[int], device=None) -> Tensor:
    k = torch.tensor(k, dtype=torch.float32, device=device)
    if k.ndim == 1:
        k = k[:, None] * k[None, :]
    k /= k.sum()
    return k


def setup_blur_weights(k: Sequence[int], up=0, down=0, is_ref=True) -> Tensor:
    assert not (up and down)
    if k is None:
        k = [1] * (up or down)
    k = _setup_kernel(k) * max(1, up ** 2)
    if is_ref:
        # from _upfirdn_2d_ref:
        # w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
        k = torch.flip(k, dims=(0, 1))[None, None, :]
    return k


def _same(k: Sequence) -> bool:
    for i in range(len(k)-1):
        if k[i] != k[i+1]:
            return False
    return True


class BlurWeightsMixin(object):
    def _init_blur_weights(self, blur_kernel, up=0, down=0, is_ref=True):
        if blur_kernel is None:
            blur_kernel = [1, 3, 3, 1]

        if isinstance(blur_kernel, Sequence):
            blur_kernel = setup_blur_weights(blur_kernel, up=up, down=down, is_ref=is_ref)

        if isinstance(blur_kernel, Tensor):
            Hb, Wb = blur_kernel.shape[-2:]
            assert Hb == Wb, "only square kernels are supported"
            if is_ref:
                C1, C0 = blur_kernel.shape[:2]
                assert C1 == 1 and C0 == 1
            # noinspection PyUnresolvedReferences
            self.register_buffer("_weight_blur", blur_kernel)
        elif hasattr(blur_kernel, "__call__"):
            # a way to get a proper reference to the shared buffer after it has been moved to GPU
            self._weight_blur_func = blur_kernel
        else:
            raise AttributeError("blur_kernel must be of type Callable, List, Tuple or None")

    @property
    def weight_blur(self):
        if hasattr(self, '_weight_blur_func'):
            return self._weight_blur_func()
        return self._weight_blur


class ModulatedConv2d(_ConvNd, BlurWeightsMixin):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, demodulate=True,
                 upsample=False, impl="ref", blur_kernel=None, scale_weights=True, lr_mult=1.0):
        if impl not in ["torch", "ref", "cuda", "cuda_full"]:
            raise AttributeError("impl should be one of [torch, ref, cuda, cuda_full]")

        if isinstance(kernel_size, Sequence):
            if not _same(kernel_size):
                raise AttributeError("Only square kernels are supported")

        kernel_size = _pair(kernel_size)
        padding = _pair(kernel_size[0] // 2)
        stride = _pair(2 if upsample else 1)
        dilation = _pair(1)

        self.demodulate = demodulate
        self.upsample = upsample
        self.impl = impl
        self.scale_weights = scale_weights
        self.lr_mult = lr_mult
        self.weight_mult = 1.0

        transposed = upsample and impl != "torch"
        super(ModulatedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            transposed, _pair(0), groups=1, bias=bias, padding_mode='zeros')

        if upsample:
            if impl != "torch":
                self._init_blur_weights(blur_kernel, up=2, is_ref=(impl == "ref"))
                W_blur = self.weight_blur.size(-1)
                W_conv = kernel_size[0]
                p = (W_blur - 2) - (W_conv - 1)
                self.pad0 = (p + 1) // 2 + 1
                self.pad1 = p // 2 + 1
            self._exec = getattr(self, "_upsample_conv2d_" + impl)
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
        d = torch.rsqrt_(w.pow(2).sum(dim=(C_in, 3, 4), keepdim=True).add_(eps))
        return w * d

    def _upsample_conv2d_cuda(self, x, w):
        # type: (Tensor, Tensor) -> Tensor
        x = self._conv_transpose2d(x, w)
        return upfirdn_2d_cuda(x, self.weight_blur, pad0=self.pad0, pad1=self.pad1)

    _upsample_conv2d_cuda_full = _upsample_conv2d_cuda

    def _upsample_conv2d_ref(self, x, w):
        # type: (Tensor, Tensor) -> Tensor
        x = self._conv_transpose2d(x, w)
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

    @staticmethod
    def _conv_transpose2d(x, w):
        # type: (Tensor, Tensor) -> Tensor
        N, C0, C1, Hc, Wc = w.shape
        w = w.view(N * C0, C1, Hc, Wc)
        x = F.conv_transpose2d(x, w, stride=2, padding=0, groups=N)
        H1, W1, = x.shape[-2:]
        return x.view(N, C1, H1, W1)

    def conv2d_forward(self, x, style, weight):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        w = self._modulate(weight, style)
        if self.demodulate:
            w = self._demodulate(w)

        N, C0, H0, W0 = x.shape
        x = x.view(1, N * C0, H0, W0)
        return self._exec(x, w)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        x = self.conv2d_forward(x, style, self.weight * self.weight_mult)
        if self.bias is not None:
            x = x.add(self.lr_mult, self.bias[:, None, None])
        return x

    def extra_repr(self) -> str:
        s = ', impl={impl}'
        if not self.demodulate:
            s += ', demodulate=False'
        if self.upsample:
            s += ', upsample=True'
        if not self.scale_weights:
            s += ', scale_weights=False'
        if self.lr_mult != 1.0:
            s += ', lr_mult={lr_mult}'
        return super().extra_repr() + s.format(**self.__dict__)


# noinspection PyPep8Naming
class Conv2d_Downsample(nn.Module, BlurWeightsMixin):
    upfirdn_2d_fn = dict(ref=upfirdn_2d_opt,
                         cuda=upfirdn_2d_cuda,
                         cuda_full=upfirdn_2d_cuda)

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, impl="ref",
                 blur_kernel=None):
        super(Conv2d_Downsample, self).__init__()

        if impl not in ["ref", "cuda", "cuda_full"]:
            raise AttributeError("impl should be one of [ref, cuda, cuda_full]")

        down = 2
        self.conv = EqualizedLRConv2d(in_channels, out_channels, kernel_size, stride=down,
                                      bias=bias)
        self._init_blur_weights(blur_kernel, down=down, is_ref=(impl == "ref"))
        W_blur = self.weight_blur.size(-1)
        W_conv = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        p = (W_blur - down) + (W_conv - 1)
        self.pad0 = (p + 1) // 2
        self.pad1 = p // 2
        self.upfirdn_2d = self.upfirdn_2d_fn[impl]

    def forward(self, x: Tensor) -> Tensor:
        x = self.upfirdn_2d(x, self.weight_blur, pad0=self.pad0, pad1=self.pad1)
        return self.conv(x)
