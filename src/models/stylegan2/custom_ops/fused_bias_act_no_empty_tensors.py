import math
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Mapping, Optional, Tuple, Union

try:
    import fused_bias_act_op
except ImportError:
    import os
    from torch.utils import cpp_extension
    module_dir = os.path.dirname(__file__)
    sources = [os.path.join(module_dir, 'fused_bias_act.cpp'),
               os.path.join(module_dir, 'fused_bias_act.cu')]
    fused_bias_act_op = cpp_extension.load('fused_bias_act_op', sources)


FUNCS = {
    'linear':   dict(func=lambda x, **_:        x,                       alpha=None, gain=1.0,          cuda_idx=1, ref='y', zero_2nd_grad=True),
    'relu':     dict(func=lambda x, **_:        torch.relu_(x),          alpha=None, gain=math.sqrt(2), cuda_idx=2, ref='y', zero_2nd_grad=True),
    'lrelu':    dict(func=lambda x, alpha, **_: F.leaky_relu_(x, alpha), alpha=0.2,  gain=math.sqrt(2), cuda_idx=3, ref='y', zero_2nd_grad=True),
    'tanh':     dict(func=lambda x, **_:        torch.tanh_(x),          alpha=None, gain=1.0,          cuda_idx=4, ref='y', zero_2nd_grad=False),
    'sigmoid':  dict(func=lambda x, **_:        torch.sigmoid_(x),       alpha=None, gain=1.0,          cuda_idx=5, ref='y', zero_2nd_grad=False),
    'elu':      dict(func=lambda x, **_:        F.elu_(x),               alpha=None, gain=1.0,          cuda_idx=6, ref='y', zero_2nd_grad=False),
    'selu':     dict(func=lambda x, **_:        torch.selu_(x),          alpha=None, gain=1.0,          cuda_idx=7, ref='y', zero_2nd_grad=False),
    'softplus': dict(func=lambda x, **_:        F.softplus(x),           alpha=None, gain=1.0,          cuda_idx=8, ref='y', zero_2nd_grad=False),
    'swish':    dict(func=lambda x, **_:        torch.sigmoid(x) * x,    alpha=None, gain=math.sqrt(2), cuda_idx=9, ref='x', zero_2nd_grad=False),
}


class FusedBiasAct(torch.autograd.Function):
    r"""Fused bias and activation function.
    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can have any shape, but if `b` is defined, the
                dimension corresponding to `axis`, as well as the rank, must be known.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `axis`.
        axis:   The dimension in `x` corresponding to the elements of `b`.
                The value of `axis` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying `1.0`.

    Returns:
        Tensor of the same shape and datatype as `x`.

    original:
    https://github.com/NVlabs/stylegan2/blob/master/dnnlib/tflib/ops/fused_bias_act.py
    """
    @staticmethod
    def forward(ctx, x, b=None, axis=1, act='linear', alpha=None, gain=None):
        # type: (Any, Tensor, Optional[Tensor], int, str, Optional[float], Optional[float]) -> Tensor
        if b is not None:
            assert b.ndim == 1 and (b.shape[0] == 0 or b.shape[0] == x.shape[axis])
            assert b.shape[0] == 0 or 0 <= axis < x.ndim

        spec = FUNCS[act]
        if alpha is None:
            alpha = spec['alpha'] or 0.0
        if gain is None:
            gain = spec['gain'] or 1.0

        # Special cases.
        if act == 'linear' and b is None and gain == 1.0:
            return x
        if spec['cuda_idx'] is None:
            raise NotImplementedError("sorry, bro...")
        if not spec['zero_2nd_grad']:
            raise NotImplementedError("sorry, bro...")

        empty = x.new_empty([0])
        kwargs = dict(axis=axis, act=spec['cuda_idx'], alpha=alpha, gain=gain)
        bias = empty if b is None else b
        y = fused_bias_act_op.call(x=x, b=bias, ref=empty, grad=0, **kwargs)

        ref = {'x': x, 'y': y}[spec['ref']]
        ctx.save_for_backward(x, b, ref)
        ctx.kwargs = kwargs
        return y

    @staticmethod
    def backward(ctx: Any, *args: Tensor):
        dy = args[0]
        x, b, ref = ctx.saved_tensors
        grads = FusedBiasActBackward.apply(dy, x, b, ref, ctx.kwargs)
        if b is None:
            dx, db = grads, None
        else:
            dx, db = grads
        return dx, db, None, None, None, None


class FusedBiasActBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, x, b, ref, kwargs):
        # type: (Any, Tensor, Tensor, Tensor, Tensor, Mapping[str, Any]) -> Union[Tensor, Tuple[Tensor, Tensor]]
        ctx.save_for_backward(ref)
        ctx.kwargs = kwargs

        dx = fused_bias_act_op.call(x=dy, b=dy.new_empty([0]), ref=ref, grad=1, **kwargs)
        if b is None:
            return dx

        axis = kwargs['axis']
        dims = list(filter(lambda ax: ax != axis, range(x.ndim)))
        db = torch.sum(dx, dims)
        return dx, db

    @staticmethod
    def backward(ctx: Any, *grads: Tensor):
        if len(grads) > 1:
            d_dx, d_db = grads
        else:
            d_dx, d_db = grads[0], grads[0].new_empty([0])
        ref, = ctx.saved_tensors
        d_dy: Tensor = fused_bias_act_op.call(x=d_dx, b=d_db, ref=ref, grad=1, **ctx.kwargs)
        return d_dy, None, None, None, None
