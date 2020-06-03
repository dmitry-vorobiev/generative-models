import torch

from .net import Generator


def sample_random_z(G: Generator, batch_size: int, device=None):
    return torch.randn(batch_size, G.latent_dim, device=device)


def sample_random_images(G: Generator, batch_size: int, device=None):
    z = sample_random_z(G, batch_size, device=device)
    images, w = G(z)
    return images
