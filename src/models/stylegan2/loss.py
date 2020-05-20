import logging
import math
import torch
import torch.nn.functional as F

from torch import autograd, nn, Tensor
from typing import Optional, Union, Tuple

from .net import Discriminator, Generator
from my_types import FloatDict, LossWithStats


log = logging.getLogger(__name__)


def r1_penalty(real_pred: Tensor, real_img: Tensor, reduce_mean=True) -> Tensor:
    grad, = autograd.grad(real_pred.sum(), real_img, create_graph=True)
    penalty = grad.pow(2).sum(dim=(1, 2, 3))
    if reduce_mean:
        return penalty.mean()
    return penalty[:, None]


# noinspection PyPep8Naming
class D_LogisticLoss_R1(nn.Module):
    """ R1 and R2 regularizers from the paper
        "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

        Args:
            r1_gamma: Scale for regularization term, None - disabled.
            r1_interval: Number of training steps between consecutive regularized updates,
                None - disable lazy regularization
    """
    def __init__(self, r1_gamma=10.0, r1_interval=16):
        super(D_LogisticLoss_R1, self).__init__()

        if r1_gamma is None or r1_gamma <= 0.0:
            log.warning("Given r1_gamma={} R1 regularization in D_loss will be turned off"
                        .format(r1_gamma))
            r1_gamma = 0.0

        if r1_interval is None or r1_interval < 1:
            r1_interval = 1

        self.r1_gamma = r1_gamma
        self.reg_interval = r1_interval
        self.count = 0

    @property
    def should_reg(self) -> bool:
        if self.r1_gamma <= 0.0:
            return False
        return self.count % self.reg_interval == 0

    def update_count(self) -> None:
        self.count = (self.count + 1) % self.reg_interval

    @staticmethod
    def zero_stats():
        return dict(D_loss=0.0, D_r1=0.0, D_real=0.0, D_fake=0.0)

    def forward(self, G, D, reals, z, label=None, stats=None):
        # type: (Generator, Discriminator, Tensor, Tensor, Tensor, Optional[FloatDict]) -> LossWithStats
        if stats is None:
            stats = self.zero_stats()

        if self.should_reg:
            reals.requires_grad_(True)

        fakes, fake_w = G(z, label)
        real_score = D(reals, label)
        fake_score = D(fakes, label)
        # -log(1 - sigmoid(fake_score)) + -log(sigmoid(real_score))
        loss = (F.softplus(fake_score) + F.softplus(-real_score)).mean()

        with torch.no_grad():
            stats['D_real'] = real_score.mean().item()
            stats['D_fake'] = fake_score.mean().item()
        del fake_score, fakes, fake_w

        if self.should_reg:
            penalty = r1_penalty(real_score, reals, reduce_mean=True)
            stats['D_r1'] = penalty.item()
            reg = penalty * ((self.r1_gamma * 0.5) * self.reg_interval)
            loss = loss + reg

        stats['D_loss'] = loss.item()
        self.update_count()
        return loss, stats


def path_length(fake_images: Tensor, fake_w: Tensor) -> Tensor:
    N, C, H, W = fake_images.shape
    # Compute |J*y|.
    pl_noise = torch.randn_like(fake_images).div_(math.sqrt(H * W))
    pl_grad, = autograd.grad(torch.sum(fake_images * pl_noise), fake_w, create_graph=True)
    # fake_w comes in shape (L, N, S).
    # Summing style_dim (dlatent) and averaging across all layers.
    return torch.sqrt(pl_grad.pow(2).sum(dim=2, keepdim=True).mean(dim=0))


def path_len_penalty(fake_images, fake_w, pl_avg=0.0, pl_decay=0.01, reduce_mean=True):
    # type: (Tensor, Tensor, Union[float, Tensor], float, bool) -> Tensor
    pl_lengths = path_length(fake_images, fake_w)
    # Track exponential moving average of |J*y|. It's updated inplace.
    with torch.no_grad():
        pl_avg += pl_decay * (pl_lengths.mean() - pl_avg)
    # Calculate (|J*y|-a)^2.
    penalty = torch.pow(pl_lengths - pl_avg, 2)
    if reduce_mean:
        penalty = penalty.mean()
    return penalty


# noinspection PyPep8Naming
class G_LogisticNSLoss_PathLenReg(nn.Module):
    r"""Non-saturating logistic loss with path length regularizer from the paper
       "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

       Args:
            pl_decay: Decay for tracking the moving average of path lengths.
            pl_reg_weight: Scale for regularization term, None - disabled.
            pl_reg_interval: Number of training steps between consecutive regularized updates,
                None - disable lazy regularization
            pl_minibatch_shrink: Use lower batch_size to evaluate regularization term,
                None - use original batch_size
    """
    def __init__(self, pl_decay=0.01, pl_reg_weight=2.0, pl_reg_interval=4,
                 pl_minibatch_shrink=2):
        super(G_LogisticNSLoss_PathLenReg, self).__init__()

        if pl_decay <= 0.0:
            raise AttributeError("pl_decay should be greater than zero")

        if pl_reg_weight is None or pl_reg_weight <= 0.0:
            log.warning("Given pl_reg_weight={} path length regularization in G_loss "
                        "will be turned off".format(pl_reg_weight))
            pl_reg_weight = 0.0

        if pl_reg_interval is None or pl_reg_interval < 1:
            pl_reg_interval = 1

        if pl_minibatch_shrink is None or pl_minibatch_shrink < 1:
            pl_minibatch_shrink = 1

        self.pl_decay = pl_decay
        self.pl_weight = pl_reg_weight
        self.reg_interval = pl_reg_interval
        self.pl_minibatch_shrink = pl_minibatch_shrink

        self.register_buffer('pl_avg', torch.zeros(1))
        self.count = 0

    @property
    def should_reg(self) -> bool:
        if self.pl_weight <= 0.0:
            return False
        return self.count % self.reg_interval == 0

    def update_count(self) -> None:
        self.count = (self.count + 1) % self.reg_interval

    @staticmethod
    def zero_stats():
        return dict(G_loss=0.0, G_pl=0.0, G_fake=0.0)

    def forward(self, G, D, z, label=None, stats=None):
        # type: (Generator, Discriminator, Tensor, Tensor, Optional[FloatDict]) -> LossWithStats
        if stats is None:
            stats = self.zero_stats()

        fakes, w = G(z, label)
        fake_score = D(fakes, label)
        loss = F.softplus(-fake_score).mean()  # -log(sigmoid(fake_score))

        with torch.no_grad():
            stats['G_fake'] = fake_score.mean().item()
        del fake_score

        if self.should_reg:

            # Evaluate the regularization term using a smaller minibatch to conserve memory.
            if self.pl_minibatch_shrink > 1:
                N = fakes.size(0)
                N_pl = max(N // self.pl_minibatch_shrink, 1)
                indices = torch.randperm(N, device=z.device)[:N_pl]
                z = z[indices]
                if label is not None:
                    label = label[indices]
                fakes, w = G(z, label)

            penalty = path_len_penalty(fakes, w, self.pl_avg, self.pl_decay, reduce_mean=True)
            stats['G_pl'] = penalty.item()

            # Apply weight.
            #
            # Note: The division in pl_noise decreases the weight by num_pixels,
            # and the reduce_mean in pl_lengths decreases it by num_affine_layers.
            # The effective weight then becomes:
            #
            # gamma_pl = pl_weight / num_pixels / num_affine_layers
            # = 2 / (r^2) / (log2(r) * 2 - 2)
            # = 1 / (r^2 * (log2(r) - 1))
            # = ln(2) / (r^2 * (ln(r) - ln(2))
            #
            reg = penalty * (self.pl_weight * self.reg_interval)
            loss = loss + reg

        stats['G_loss'] = loss.item()
        self.update_count()
        return loss, stats
