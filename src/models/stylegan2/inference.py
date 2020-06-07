from .net import Generator
from .train import sample_latent, sample_rand_label


def sample_random_images(G: Generator, batch_size: int, device=None):
    z = sample_latent(batch_size, G.latent_dim, device=device)

    label = None
    if G.num_classes > 1:
        label = sample_rand_label(batch_size, G.num_classes, device=device)

    images, w = G(z, label)
    return images
