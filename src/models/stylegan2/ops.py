import torch
from torch import Tensor


def minibatch_stddev(x: Tensor, group_size: int, num_new_features: int, eps=1e-8):
    N, C, H, W = x.shape
    assert C % num_new_features == 0, 'C must be divisible by n'
    # Minibatch must be divisible by (or smaller than) group_size.
    G = min(group_size, N)
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
