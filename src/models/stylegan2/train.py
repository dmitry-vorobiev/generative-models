import logging
import torch

from functools import partial
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Any, Mapping, Optional, Tuple

from .net import Discriminator, Generator
from my_types import Batch, Device, DLossFunc, FloatDict, GLossFunc, SampleImages, TrainFunc

Options = Optional[Mapping[str, Any]]

log = logging.getLogger(__name__)


def update_G_ema(G, G_ema, weight=0.001):
    # type: (Generator, Generator, float) -> None
    with torch.no_grad():
        for param, curr_value in zip(G_ema.parameters(), G.parameters()):
            curr_value = curr_value.to(param.device)
            param.lerp_(curr_value, weight)

        # No point to iterate over all buffers, most of them doesn't change their values
        w_avg = G_ema.w_avg
        w_avg.copy_(G.w_avg.to(w_avg.device))


def sample_latent(batch_size, dim, device=None):
    # type: (int, int, Device) -> Tensor
    return torch.randn(batch_size, dim, device=device)


def sample_rand_label(batch_size, num_classes, device=None):
    # type: (int, int, Device) -> Tensor
    return torch.randint(num_classes, (batch_size,), device=device)


def create_train_closures(G, D, G_loss_func, D_loss_func, G_opt, D_opt, G_ema=None, device=None,
                          options=None):
    # type: (Generator, Discriminator, GLossFunc, DLossFunc, Optimizer, Optimizer, Optional[Generator], Device, Options) -> Tuple[TrainFunc, SampleImages]

    def _sample_fake_images() -> Tensor:
        G_ema.eval()
        # TODO: add separate batch_size for G_ema
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
                if not iteration % ema_rounds:
                    update_G_ema(G_, G_ema, ema_weight)

        # Training discriminator
        if not iteration % rounds:
            G.requires_grad_(False)
            D.requires_grad_(True)
            D_opt.zero_grad()

        z = _sample_z(N)
        image = image.to(device)
        if label is not None:
            label = label.to(device)
        d_loss, d_stats = D_loss_func(G, D, image, z, label, stats)
        d_loss.backward()

        if not (iteration + 1) % rounds:
            D_opt.step()

        return stats

    assert options is not None
    train_opts = options['train']
    batch_size = train_opts['batch_size']
    rounds = train_opts['update_interval']

    smooth_opts = options['smoothing']
    if G_ema is not None:
        ema_rounds = smooth_opts.get('upd_interval', 1)
        smooth_num_images = smooth_opts.get('num_kimg', 10.0) * 1000
        beta = 0.5 ** (batch_size / smooth_num_images)
        beta **= ema_rounds
        log.info("Using exponential moving average of G weights with beta %.06f "
                 "and update interval %d steps" % (beta, ema_rounds))
        ema_weight = 1 - beta

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

    stats = dict()
    return _loop, _sample_fake_images


# There are no extra optimizer steps currently
def update_optimizer_params(G_opt, D_opt, G_loss_func, D_loss_func):
    # type: (Optimizer, Optimizer, GLossFunc, DLossFunc) -> None
    """
    To compensate for the fact that we now perform
    k+1 training iterations instead of k, we adjust the optimizer
    hyperparameters
    """
    # Update optimizer settings if lazy_regularization is used (reg_interval > 1)
    for optimizer, loss in zip([G_opt, D_opt], [G_loss_func, D_loss_func]):
        if hasattr(loss, 'reg_interval') and loss.reg_interval > 1:
            mb_ratio = loss.reg_interval / (loss.reg_interval + 1)
            for group in optimizer.param_groups:
                group['lr'] *= mb_ratio
                if 'betas' in group:
                    beta1, beta2 = group['betas']
                    group['betas'] = (beta1 ** mb_ratio, beta2 ** mb_ratio)
