import math
import torch
import torch.nn.functional as F

from torch import autograd, nn, Tensor
from typing import Union, Tuple

from .net import Discriminator, Generator, Latent, Label


def r1_penalty(real_pred: Tensor, real_img: Tensor):
    grad, = autograd.grad(real_pred.sum(), real_img, create_graph=True)
    return grad.pow(2).sum(dim=(1, 2, 3))[:, None]


# noinspection PyPep8Naming
class D_LogisticLoss_R1(nn.Module):
    """ R1 and R2 regularizers from the paper
        "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018
    """

    def __init__(self, generator, discriminator, r1_freq=16, r1_gamma=10.0):
        # type: (Generator, Discriminator, int, float) -> D_LogisticLoss_R1
        super(D_LogisticLoss_R1, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.freq = r1_freq
        self.gamma = r1_gamma
        self.count = 0

    @property
    def should_reg(self) -> bool:
        return self.count % self.freq == 0

    def update_count(self) -> None:
        self.count = (self.count + 1) % self.freq

    def forward(self, reals, z, label=None):
        # type: (Tensor, Latent, Label) -> Tensor
        if self.should_reg:
            reals.requires_grad_(True)

        fakes, fake_w = self.generator(z, label)
        real_score = self.discriminator(reals, label)
        fake_score = self.discriminator(fakes, label)
        loss = F.softplus(fake_score) + F.softplus(-real_score)

        if self.should_reg:
            penalty = r1_penalty(real_score, reals)
            reg = penalty * (self.gamma * 0.5)
            loss = loss + (reg * self.freq)

        self.update_count()
        return loss


def path_length(fake_img: Tensor, fake_w: Tensor) -> Tensor:
    N, C, H, W = fake_img.shape
    noise = torch.randn_like(fake_img) / math.sqrt(H * W)
    grad, = autograd.grad((fake_img * noise).sum(), fake_w, create_graph=True)
    # fake_w: (L, N, S)
    return torch.sqrt(grad.pow(2).sum(dim=2, keepdim=True).mean(dim=0))


def path_len_penalty(fake_img, fake_w, path_len_avg=0.0, decay=0.01):
    # type: (Tensor, Tensor, Union[float, Tensor], float) -> Tuple[Tensor, Tensor]
    path_len = path_length(fake_img, fake_w)
    with torch.no_grad():
        path_len_avg = path_len_avg + decay * (path_len.mean() - path_len_avg)
    penalty = torch.pow(path_len - path_len_avg, 2)
    return penalty, path_len_avg


# noinspection PyPep8Naming
class G_LogisticNSLoss_PathLenReg(nn.Module):
    """Non-saturating logistic loss with path length regularizer from the paper
       "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019
    """

    def __init__(self, generator, discriminator,
                 pl_ema_decay=0.01, pl_reg_freq=4, pl_reg_weight=2.0):
        # type: (Generator, Discriminator, float, int, float) -> G_LogisticNSLoss_PathLenReg
        super(G_LogisticNSLoss_PathLenReg, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.decay = pl_ema_decay
        self.freq = pl_reg_freq
        self.weight = pl_reg_weight

        self.register_buffer('pl_avg', torch.zeros(1))
        self.count = 0

    @property
    def should_reg(self) -> bool:
        return self.count % self.freq == 0

    def update_count(self) -> None:
        self.count = (self.count + 1) % self.freq

    def forward(self, z, label=None):
        # type: (Latent, Label) -> Tensor
        fakes, w = self.generator(z, label)
        fake_score = self.discriminator(fakes, label)
        loss = F.softplus(-fake_score)

        if self.should_reg:
            penalty, self.pl_avg = path_len_penalty(fakes, w, self.pl_avg, self.decay)

            # Note: The division in pl_noise decreases the weight by num_pixels,
            # and the reduce_mean in pl_lengths decreases it by num_affine_layers.
            # The effective weight then becomes:
            #
            # gamma_pl = pl_weight / num_pixels / num_affine_layers
            # = 2 / (r^2) / (log2(r) * 2 - 2)
            # = 1 / (r^2 * (log2(r) - 1))
            # = ln(2) / (r^2 * (ln(r) - ln(2))
            #
            reg = penalty * self.weight
            loss = loss + (reg * self.freq)

        self.update_count()
        return loss
