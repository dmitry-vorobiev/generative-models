import torch
import torch.nn.functional as F

from functools import partial
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Any, Mapping, Optional, Sequence, Tuple

from .net import Discriminator, Generator
from my_types import Batch, Device, DLossFunc, FloatDict, GLossFunc, SnapshotFunc, TrainFunc

Options = Optional[Mapping[str, Any]]


# TODO: how to properly handle buffers?
def ema_step(G, G_ema, weight=0.001):
    # type: (Generator, Generator, float) -> None
    curr_state = dict(G.named_parameters(recurse=True))
    with torch.no_grad():
        for name, param in G_ema.named_parameters(recurse=True):
            curr_value = curr_state[name].to(param.device)
            param.lerp_(curr_value, weight)


def sample_latent(batch_size, dim, device=None):
    # type: (int, int, Device) -> Tensor
    return torch.randn(batch_size, dim, device=device)


def sample_rand_label(batch_size, num_classes, device=None):
    # type: (int, int, Device) -> Tensor
    return torch.randint(num_classes, (batch_size,), device=device)


def create_train_closures(G, D, G_loss_func, D_loss_func, G_opt, D_opt, G_ema=None, device=None,
                          options=None):
    # type: (Generator, Discriminator, GLossFunc, DLossFunc, Optimizer, Optimizer, Optional[Generator], Device, Options) -> Tuple[TrainFunc, SnapshotFunc]

    def _make_snapshot() -> Tensor:
        G_ema.eval()
        snap_fakes, _ = G_ema(fixed_z, fixed_label)
        return snap_fakes

    def _loop(iteration, batch):
        # type: (int, Batch) -> FloatDict
        G.train()
        D.train()

        if isinstance(batch, Tensor):
            image, label = batch, None
        else:
            image, label = batch
        N = image.size(0)

        if not iteration % rounds:
            # Training generator
            G.requires_grad_(True)
            D.requires_grad_(False)
            G_opt.zero_grad()
            for _ in range(rounds):
                z = _sample_z(N)
                fake_label = _sample_label(N)
                g_loss, g_stats = G_loss_func(G, D, z, fake_label, stats)
                g_loss.backward()
            G_opt.step()
            del z, fake_label

            # Average G weights
            if G_ema is not None:
                ema_step(G_, G_ema, ema_decay)

        # Training discriminator
        if not iteration % rounds:
            G.requires_grad_(False)
            D.requires_grad_(True)
            D_opt.zero_grad()

        image = image.to(device)
        if label is not None:
            label = label.to(device)

        z = _sample_z(N)
        d_loss, d_stats = D_loss_func(G, D, image, z, label, stats)
        d_loss.backward()

        if not (iteration + 1) % rounds:
            D_opt.step()

        return stats

    assert options is not None
    train_opts = options['train']
    ema_decay = train_opts.get("G_ema_decay", 0.999)
    ema_decay = 1 - ema_decay
    rounds = train_opts['update_interval']

    G_ = G.module if hasattr(G, 'module') else G
    D_ = D.module if hasattr(D, 'module') else D
    latent_dim = G_.latent_dim
    num_classes = G_.num_classes
    _sample_z = partial(sample_latent, dim=latent_dim, device=device)

    if G_.num_classes != D_.num_classes:
        raise AttributeError("num_classes for G and D doesn't match, G: {}, D: {}"
                             .format(G_.num_classes, D_.num_classes))

    def _sample_label(*args): return None
    if num_classes > 1:
        _sample_label = partial(sample_rand_label, num_classes=num_classes, device=device)

    snap_opts = options['snapshot']
    fixed_z, fixed_label = None, None
    if snap_opts['enabled'] and G_ema is not None:
        N_snap = snap_opts.get("num_images", 16)
        G_ema_device = next(G_ema.parameters()).device
        fixed_z = sample_latent(N_snap, latent_dim, G_ema_device)
        if num_classes > 1:
            fixed_label = sample_rand_label(N_snap, num_classes, G_ema_device)

    # Update optimizer settings if lazy_regularization is used (reg_interval > 1)
    for optimizer, loss in zip([G_opt, D_opt], [G_loss_func, D_loss_func]):
        if hasattr(loss, 'reg_interval') and loss.reg_interval > 1:
            mb_ratio = loss.reg_interval / (loss.reg_interval + 1)
            for group in optimizer.param_groups:
                group['lr'] *= mb_ratio
                if 'betas' in group:
                    beta1, beta2 = group['betas']
                    group['betas'] = (beta1 ** mb_ratio, beta2 ** mb_ratio)

    stats = dict()
    return _loop, _make_snapshot
