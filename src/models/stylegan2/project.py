import logging
import torch

from torch import Tensor

from .net import Generator

log = logging.getLogger(__name__)


class Projector(object):
    def __init__(self, G: Generator, batch_size=1, num_steps=1000, dlatent_avg_samples=10_000,
                 init_lr=0.1, init_noise_mult=0.05, lr_warm_len=0.05, lr_anneal_len=0.25,
                 noise_ramp_len=0.75, noise_reg_weight=1e+5, device=None):
        self.batch_size = batch_size
        self.G = G
        self.num_steps = num_steps
        self.dlatent_avg_samples = dlatent_avg_samples
        self.init_lr = init_lr
        self.init_noise_mult = init_noise_mult
        self.lr_warm_len = lr_warm_len
        self.lr_anneal_len = lr_anneal_len
        self.noise_ramp_len = noise_ramp_len
        self.noise_reg_weight = noise_reg_weight

        if G is None:
            raise AttributeError("Unable initialize projector with G=None")

        if G.synthesis.randomize_noise:
            raise AttributeError("randomize_noise in G is unsupported")

        # Find dlatent stats.
        log.info("Searching W mean and std using %d samples" % dlatent_avg_samples)
        torch.manual_seed(123)
        z = torch.randn(dlatent_avg_samples, G.latent_dim, device=device)
        w: Tensor = G.mapping(z, label=None)
        w_mean = w.mean(dim=0, keepdim=True)
        w_std = torch.sqrt((w - w_mean).pow_(2).sum() / dlatent_avg_samples)
        log.info("W std: %g" % w_std)

        # Find noise inputs.
        log.info("Setting up noise inputs...")
        noise_vars = []
        res_log2 = G.synthesis.res_log2
        params = dict(G.synthesis.named_parameters())

        for i in range(0, res_log2 * 2 - 2):
            name = "main.{}.add_noise.noise".format(i)
            v = params[name].normal_()
            noise_vars.append(v)
            log.info("{}: {}".format(i, tuple(v.shape)))
