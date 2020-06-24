import logging
import math
import torch
import torch.nn.functional as F

from torch import nn, Tensor

from .net import Generator
from metrics.lpips import PNetLin

log = logging.getLogger(__name__)


def preproc_image(image: Tensor) -> Tensor:
    # convert image to pixel range
    # (-1, 1) -> (0, 2) -> (0, 255)
    # maybe clip to (-1, 1) ?
    image = (image + 1) * (255 / 2)

    H, W = image.shape[2:]
    if max(H, W) > 256:
        scale = 256 / max(H, W)
        image = F.interpolate(image, scale_factor=scale, mode="nearest")

    return image


class Projector(object):
    def __init__(self, G: Generator, batch_size=1, num_steps=1000, dlatent_avg_samples=10_000,
                 init_lr=0.1, lr_warm_len=0.05, lr_anneal_len=0.25, init_noise_mult=0.05,
                 noise_ramp_len=0.75, reg_noise_weight=1e5, device=None):
        self.batch_size = batch_size
        self.G = G.to(device)
        self.num_steps = num_steps
        self.dlatent_avg_samples = dlatent_avg_samples
        self.init_lr = init_lr
        self.init_noise_mult = init_noise_mult
        self.lr_warm_len = lr_warm_len
        self.lr_anneal_len = lr_anneal_len
        self.noise_ramp_len = noise_ramp_len
        self.reg_noise_weight = reg_noise_weight
        self.device = device

        # maybe use nn.Module for this?
        w = torch.zeros(batch_size, *G.w_avg.shape).to(device)
        self.w = nn.Parameter(w, requires_grad=True)
        self.distance = PNetLin().to(device)

        # call self.init() first
        self.noise_params = None
        self.target_image = None
        self.optim = None
        self._cur_step = 0

        if G is None:
            raise AttributeError("Unable initialize projector with G=None")

        if G.synthesis.randomize_noise:
            raise AttributeError("randomize_noise in G is unsupported")

        # Find dlatent stats.
        print("Searching W mean and std using %d samples" % dlatent_avg_samples)
        torch.manual_seed(123)
        z = torch.randn(dlatent_avg_samples, G.latent_dim, device=device)
        w: Tensor = G.mapping(z, label=None)
        w_mean = w.mean(dim=0, keepdim=True)
        self.w_std = torch.sqrt((w - w_mean).pow_(2).sum() / dlatent_avg_samples)
        print("W std: %g" % self.w_std)

    def init(self, image: Tensor):
        print("Preparing target images...")
        self.target_image = preproc_image(image.to(self.device))
        G = self.G

        G.truncation_psi = None
        G.eval()
        G.requires_grad_(False)

        # Find noise inputs.
        print("Setting up noise inputs...")
        noise_params = []
        buffers = dict(G.synthesis.main.named_buffers())

        for i in range(0, G.synthesis.res_log2 * 2 - 3):
            name = "{}.add_noise.noise".format(i)
            v = buffers[name].normal_().requires_grad_(True)
            noise_params.append(v)

        self.optim = torch.optim.Adam([self.w, *noise_params])
        self.noise_params = noise_params
        self._cur_step = 0

    def _update_lr(self):
        t = self._cur_step / self.num_steps
        lr_ramp = min(1.0, (1.0 - t) / self.lr_anneal_len)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_warm_len)
        lr = self.init_lr * lr_ramp

        for group in self.optim.param_groups:
            group['lr'] = lr

    @property
    def _noise_mult(self):
        t = self._cur_step / self.num_steps
        return self.w_std * self.init_noise_mult * max(0.0, 1 - t / self.noise_ramp_len) ** 2

    def step(self):
        G = self.G
        self._update_lr()

        # Train
        w_noise = torch.randn_like(self.w) * self._noise_mult
        w = (self.w + w_noise).expand(G.num_layers, -1, -1)
        image = G.synthesis(w)

        image = preproc_image(image)
        dist = self.distance(image, self.target_image)
        reg = self._noise_reg()
        loss = torch.sum(dist) + reg * self.reg_noise_weight

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        self._normalize_noise()

        if not self._cur_step % 10:
            print("it: {}, loss: {:.3f}, reg: {:.3f}, dist: {:.7f}".format(
                self._cur_step, loss.item(), reg, dist.mean().item()))
        self._cur_step += 1

    def _noise_reg(self):
        reg = 0.0

        for p in self.noise_params:
            size = p.size(2)
            while True:
                reg += sum(torch.mean(p * torch.roll(p, 1, dims=d)) ** 2 for d in [2, 3])
                if size <= 8:
                    break
                # Downscale
                p = F.interpolate(p, scale_factor=0.5, mode="nearest")
                size = size // 2

        return reg

    def _normalize_noise(self):
        with torch.no_grad():
            for p in self.noise_params:
                p.add_(-1, p.mean()).div_(p.std())
