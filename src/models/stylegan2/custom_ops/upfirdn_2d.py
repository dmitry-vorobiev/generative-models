"""
The backprop solution comes from another PyTorch StyleGAN2 implementation:
https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn2d.py
Previous naive port of TF grad function caused the gradient penalty (R1)
to be orders of magnitude higher, than it should, which significantly crippled training.


MIT License

Copyright (c) 2019 Kim Seonghyeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch

from torch import Tensor
from typing import Any, Tuple

try:
    import upfirdn_2d_op
except ImportError:
    import os
    from torch.utils import cpp_extension
    module_dir = os.path.dirname(__file__)
    sources = [os.path.join(module_dir, 'upfirdn_2d.cpp'),
               os.path.join(module_dir, 'upfirdn_2d_kernel.cu')]
    upfirdn_2d_op = cpp_extension.load('upfirdn_2d_op', sources)


Tuple2Int = Tuple[int, int]
Tuple4Int = Tuple[int, int, int, int]


class UpFirDn2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel, up, down, pad):
        # type: (Any, Tensor, Tensor, Tuple2Int, Tuple2Int, Tuple4Int) -> Tensor
        y = upfirdn_2d_op.call(x, kernel, *up, *down, *pad)
        inH, inW = x.shape[-2:]
        # outH, outW = y.shape[-2:]
        kernelH, kernelW = kernel.shape

        upx, upy = up
        downx, downy = down
        padx0, padx1, pady0, pady1 = pad

        outW = (inW * upx + padx0 + padx1 - kernelW) // downx + 1
        outH = (inH * upy + pady0 + pady1 - kernelH) // downy + 1

        ctx.up = up
        ctx.down = down
        ctx.pad = pad

        gpadx0 = kernelW - padx0 - 1
        gpady0 = kernelH - pady0 - 1
        gpadx1 = inW * upx - outW * downx + padx0 - upx + 1
        gpady1 = inH * upy - outH * downy + pady0 - upy + 1
        ctx.gpad = (gpadx0, gpadx1, gpady0, gpady1)

        dkernel = torch.flip(kernel, dims=(0, 1))
        ctx.save_for_backward(kernel, dkernel)
        return y

    @staticmethod
    def backward(ctx: Any, dy: Tensor):
        kernel, dkernel = ctx.saved_tensors
        dx: Tensor = UpFirDn2DBackward.apply(dy, kernel, dkernel, ctx.up,
                                             ctx.down, ctx.pad, ctx.gpad)
        return dx, None, None, None, None


class UpFirDn2DBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, kernel, dkernel, up, down, pad, dpad):
        # type: (Any, Tensor, Tensor, Tensor, Tuple2Int, Tuple2Int, Tuple4Int, Tuple4Int) -> Tensor
        dx = upfirdn_2d_op.call(dy, dkernel, *down, *up, *dpad)
        ctx.up = up
        ctx.down = down
        ctx.pad = pad
        ctx.save_for_backward(kernel)
        return dx

    @staticmethod
    def backward(ctx: Any, ddy: Tensor):
        kernel, = ctx.saved_tensors
        ddx: Tensor = upfirdn_2d_op.call(ddy, kernel, *ctx.up, *ctx.down, *ctx.pad)
        return ddx, None, None, None, None, None, None
