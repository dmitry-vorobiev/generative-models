import torch
import torch.nn.functional as F

from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Optional

from .net import Discriminator, Generator
from my_types import Device, DLossFunc, FloatDict, GLossFunc, TrainFunc


def create_train_loop(G, D, G_loss_func, D_loss_func, G_opt, D_opt, num_classes=-1, device=None):
    # type: (Generator, Discriminator, GLossFunc, DLossFunc, Optimizer, Optimizer, int, Device) -> TrainFunc

    def _sample_latent(batch_size: int) -> Tensor:
        return torch.randn(batch_size, G.latent_dim, device=device)

    def _sample_rnd_label(batch_size: int) -> Optional[Tensor]:
        if num_classes < 2:
            return None
        y = torch.randint(num_classes, (batch_size,), device=device)
        return F.one_hot(y, num_classes=num_classes)

    def _ohe(y: Optional[Tensor]) -> Optional[Tensor]:
        if y is None:
            return None
        return F.one_hot(y, num_classes=num_classes)

    def _loop(image: Tensor, label=None) -> FloatDict:
        G.train()
        D.train()

        N = image.size(0)

        # Training generator
        G.requires_grad_(True)
        D.requires_grad_(False)
        G_opt.zero_grad()

        z = _sample_latent(N)
        fake_label = _sample_rnd_label(N)
        g_loss, g_stats = G_loss_func(G, D, z, fake_label)
        g_loss.backward()
        G_opt.step()
        del z, fake_label

        # Training discriminator
        G.requires_grad_(False)
        D.requires_grad_(True)
        D_opt.zero_grad()

        image = image.to(device)
        if label is not None:
            label = label.to(device)

        z = _sample_latent(N)
        label = _ohe(label)
        d_loss, d_stats = D_loss_func(G, D, image, z, label)
        d_loss.backward()
        D_opt.step()

        g_stats.update(d_stats)
        return g_stats

    return _loop
