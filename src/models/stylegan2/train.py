import torch
import torch.nn.functional as F

from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Optional

from .net import Discriminator, Generator
from ..types import Batch, Device, DLossFunc, GLossFunc, TrainFunc


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

    def _loop(batch: Batch):
        G.train()
        D.train()

        real_image, real_label = batch
        N = real_image.size(0)

        # Training generator
        G.requires_grad_(True)
        D.requires_grad_(False)
        G_opt.zero_grad()

        z = _sample_latent(N)
        fake_label = _sample_rnd_label(N)
        g_loss = G_loss_func(G, D, z, fake_label).mean()
        g_loss.backward()
        G_opt.step()

        # Training discriminator
        G.requires_grad_(False)
        D.requires_grad_(True)
        D_opt.zero_grad()

        z = _sample_latent(N)
        real_label = _ohe(real_label)
        d_loss = D_loss_func(G, D, real_image, z, real_label).mean()
        d_loss.backward()
        D_opt.step()

        return dict(G_loss=g_loss, D_loss=d_loss)

    return _loop
