import math
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Mapping, Optional, Tuple

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
        empty_tensor = x.new_empty([0])
        if b is None:
            b = empty_tensor

        act_spec = FUNCS[act]
        assert b.ndim == 1 and (b.shape[0] == 0 or b.shape[0] == x.shape[axis])
        assert b.shape[0] == 0 or 0 <= axis < x.ndim
        if alpha is None:
            alpha = act_spec['alpha']
        if gain is None:
            gain = act_spec['gain']

        # Special cases.
        if act == 'linear' and b is None and gain == 1.0:
            return x
        if act_spec['cuda_idx'] is None:
            raise NotImplementedError("sorry, bro...")
        if not act_spec['zero_2nd_grad']:
            raise NotImplementedError("sorry, bro...")

        kwargs = dict(axis=axis, act=act_spec['cuda_idx'], alpha=alpha, gain=gain)
        y = fused_bias_act_op.call(x, b, ref=empty_tensor, grad=0, **kwargs)
        ctx.save_for_backward(y, x, b)
        ctx.kwargs = kwargs
        return y

    @staticmethod
    def backward(ctx: Any, dy: Tensor):
        y, x, b = ctx.saved_tensors
        dx, db = FusedBiasActBackward.apply(dy, y, x, b, ctx.kwargs)
        return dx, db, None, None, None, None


class FusedBiasActBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, y, x, b, kwargs):
        # type: (Any, Tensor, Tensor, Tensor, Tensor, Mapping[str, Any]) -> Tuple[Tensor, Tensor]
        cls = FusedBiasActBackward
        dx = cls.grad_dx(dy, x, y, kwargs)
        db = cls.grad_db(dx, x, b, kwargs)
        ctx.kwargs = kwargs
        return dx, db

    @staticmethod
    def backward(ctx: Any, d_dx: Tensor, d_db: Tensor, x: Tensor, y: Tensor):
        kwargs = ctx.kwargs
        ref = {'x': x, 'y': y}[kwargs['ref']]
        d_dy: Tensor = fused_bias_act_op(d_dx, d_db, ref, grad=1, **kwargs)
        return d_dy, None, None, None, None

    @staticmethod
    def grad_dx(dy, x, y, kwargs):
        # type: (Tensor, Tensor, Tensor, Mapping[str, Any]) -> Tensor
        ref = {'x': x, 'y': y}[kwargs['ref']]
        return fused_bias_act_op.call(dy, x.new_empty([0]), ref, grad=1, **kwargs)

    @staticmethod
    def grad_db(dx, x, b, kwargs):
        # type: (Tensor, Tensor, Tensor, Mapping[str, Any]) -> Tensor
        if b.shape[0] == 0:
            return b.new_empty([0])
        axis = kwargs['axis']
        db = dx
        if axis < x.ndim - 1:
            db = torch.sum(db, list(range(axis + 1, x.ndim)))
        if axis > 0:
            db = torch.sum(db, list(range(axis)))
        return db
