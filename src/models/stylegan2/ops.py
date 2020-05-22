import torch
import torch.nn.functional as F

from torch import Tensor


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


def minibatch_stddev(x: Tensor, group_size: int, num_new_features: int, eps=1e-8):
    N, C, H, W = x.shape
    assert C % num_new_features == 0, 'C must be divisible by n'
    # Minibatch must be divisible by (or smaller than) group_size.
    G = min(group_size, N)
    assert N % G == 0, 'Batch size must be divisible by group_size'
    # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y = x.reshape(G, -1, num_new_features, C//num_new_features, H, W)
    # [GMncHW] Subtract mean over group.
    y = y - y.mean(dim=0, keepdim=True)
    # [MncHW]  Calc stddev over group.
    y = torch.sqrt(y.pow(2).mean(dim=0) + eps)
    # [Mn11] Split channels into n channel groups
    y = y.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
    # [NnHW]  Replicate over group and pixels.
    y = y.repeat(G, 1, H, W)
    return y
