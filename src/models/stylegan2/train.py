import torch
import torch.nn.functional as F

from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Any, Mapping, Optional, Tuple

from .net import Discriminator, Generator
from my_types import Device, DLossFunc, FloatDict, GLossFunc, SnapshotFunc, TrainFunc

Options = Optional[Mapping[str, Any]]


# TODO: how to properly handle buffers?
def ema_step(G, G_ema, decay=0.001):
    # type: (Generator, Generator, float) -> None
    curr_state = dict(G.named_parameters(recurse=True))
    with torch.no_grad():
        for name, param in G_ema.named_parameters(recurse=True):
            curr_value = curr_state[name].to(param.device)
            param.lerp_(curr_value, decay)


def create_train_closures(G, D, G_loss_func, D_loss_func, G_opt, D_opt, G_ema=None,
                          device=None, options=None):
    # type: (Generator, Discriminator, GLossFunc, DLossFunc, Optimizer, Optimizer, Optional[Generator], Device, Options) -> Tuple[TrainFunc, SnapshotFunc]

    def _sample_latent(batch_size: int) -> Tensor:
        return torch.randn(batch_size, G.latent_dim, device=device)

    def _sample_rand_label(batch_size: int) -> Optional[Tensor]:
        if num_classes < 2:
            return None
        y = torch.randint(num_classes, (batch_size,), device=device)
        return F.one_hot(y, num_classes=num_classes)

    def _ohe(y: Optional[Tensor]) -> Optional[Tensor]:
        if y is None:
            return None
        return F.one_hot(y, num_classes=num_classes)

    def _make_snapshot() -> Tensor:
        G.eval()
        snap_fakes, _ = G(fixed_z, fixed_label)
        return snap_fakes

    def _loop(image, label=None):
        # type: (Tensor, Optional[Tensor]) -> FloatDict
        G.train()
        D.train()
        N = image.size(0)

        # Training generator
        G.requires_grad_(True)
        D.requires_grad_(False)
        G_opt.zero_grad()

        z = _sample_latent(N)
        fake_label = _sample_rand_label(N)
        g_loss, g_stats = G_loss_func(G, D, z, fake_label, stats)
        g_loss.backward()
        G_opt.step()
        del z, fake_label

        # Average G weights
        if G_ema is not None:
            ema_step(G, G_ema, ema_decay)

        # Training discriminator
        G.requires_grad_(False)
        D.requires_grad_(True)
        D_opt.zero_grad()

        image = image.to(device)
        if label is not None:
            label = label.to(device)

        z = _sample_latent(N)
        label = _ohe(label)
        d_loss, d_stats = D_loss_func(G, D, image, z, label, stats)
        d_loss.backward()
        D_opt.step()

        return stats

    assert options is not None
    train_opts = options['train']
    num_classes = train_opts.get("num_classes", -1)
    ema_decay = train_opts.get("G_ema_decay", 0.999)
    ema_decay = 1 - ema_decay

    snap_opts = options['snapshot']
    N_snap = snap_opts.get("num_images", 16)
    fixed_z, fixed_label = None, None
    if snap_opts['enabled']:
        fixed_z = _sample_latent(N_snap)
        fixed_label = _sample_rand_label(N_snap)

    stats = dict()
    return _loop, _make_snapshot
