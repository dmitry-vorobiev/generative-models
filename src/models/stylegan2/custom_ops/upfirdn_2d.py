import torch

import upfirdn_2d_op

from torch import Tensor
from typing import Any


class UpFirDn2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, kernel, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
        # type: (Any, Tensor, Tensor, int, int, int, int, int, int, int, int) -> Tensor
        out = upfirdn_2d_op.execute(inp, kernel, upx, upy, downx, downy,
                                    padx0, padx1, pady0, pady1)
        # saving vars for backward
        inH, inW = inp.shape[-2:]
        # H1, W1 = out.shape[-2:]
        kernelH, kernelW = kernel.shape
        outW = (inW * upx + padx0 + padx1 - kernelW) // downx + 1
        outH = (inH * upy + pady0 + pady1 - kernelH) // downy + 1

        ctx.upx = downx
        ctx.upy = downy
        ctx.downx = upx
        ctx.downy = upy
        ctx.padx0 = kernelW - padx0 - 1
        ctx.pady0 = kernelH - pady0 - 1
        ctx.padx1 = inW * upx - outW * downx + padx0 - upx + 1
        ctx.pady1 = inH * upy - outH * downy + pady0 - upy + 1

        grad_kernel = torch.flip(kernel, dims=(0, 1))
        ctx.save_for_backward(grad_kernel)
        return out

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor):
        kernel, = ctx.saved_tensors
        d_inp: Tensor = upfirdn_2d_op.execute(grad_y, kernel, ctx.upx, ctx.upy,
                                              ctx.downx, ctx.downy,
                                              ctx.padx0, ctx.padx1, ctx.pady0, ctx.pady1)
        return d_inp, None, None, None, None, None, None, None, None, None


def upfirdn_2d_cuda(x, w, up=1, down=1, pad0=0, pad1=0):
    # type: (Tensor, Tensor, int, int, int, int) -> Tensor
    N, C, H, W = x.shape
    assert H > 0 and W > 0
    return UpFirDn2D.apply(x, w, up, up, down, down, pad0, pad1, pad0, pad1)
