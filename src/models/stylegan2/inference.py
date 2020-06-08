import torch
from torch import Tensor
from typing import List

from .net import Generator
from .train import sample_latent, sample_rand_label


def sample_random_images(G: Generator, batch_size: int, device=None):
    z = sample_latent(batch_size, G.latent_dim, device=device)

    label = None
    if G.num_classes > 1:
        label = sample_rand_label(batch_size, G.num_classes, device=device)

    images, w = G(z, label)
    return images


def sample_latent_determinist(seeds: List[int], latent_dim: int, device=None) -> Tensor:
    all_z = []
    for seed in seeds:
        torch.manual_seed(seed)
        z = sample_latent(1, latent_dim, device=device)
        all_z.append(z)
    return torch.cat(all_z, dim=0)


def mix_styles(G: Generator, batch_size: int, device=None, row_seeds=None, col_seeds=None,
               style_layers=None) -> Tensor:
    from tqdm import tqdm

    if style_layers is None:
        style_layers = list(range(6))

    cpu = torch.device('cpu')

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = sample_latent_determinist(all_seeds, G.latent_dim, device=device)
    # TODO: how it will work for G with num_classes > 1 ???
    all_w = G.mapping(all_z, None)  # [N, S]
    all_w = G.w_avg + (all_w - G.w_avg) * G.truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    print('Generating images...')
    all_images = []

    for i_start in range(0, len(all_w), batch_size):
        batch_w = all_w[i_start: i_start + batch_size]
        batch_w = batch_w.expand(G.num_layers, -1, -1)
        images = G.synthesis(batch_w).to(cpu)
        all_images += list(images)

    image_dict = {(seed, seed): image[None, :] for seed, image in zip(all_seeds, all_images)}

    pbar = tqdm(desc="Generating style-mixed images",
                total=len(row_seeds) * len(col_seeds),
                unit=' img')

    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].repeat(G.num_layers, 1)  # (S,) -> (L, S)
            w[style_layers, :] = w_dict[col_seed][None, :]
            image = G.synthesis(w[:, None, :]).to(cpu)
            image_dict[(row_seed, col_seed)] = image
            pbar.update(1)

    pbar.close()

    key = (col_seeds[0], col_seeds[0])
    empty = torch.zeros_like(image_dict[key])

    # placing original row images
    images_out = [empty] + [image_dict[(s, s)] for s in col_seeds]
    for row_seed in row_seeds:
        # adding original column image at the beginning of each row
        image = image_dict[(row_seed, row_seed)]
        images_out.append(image)

        for col_seed in col_seeds:
            image = image_dict[(row_seed, col_seed)]
            images_out.append(image)

    images_out = torch.cat(images_out, dim=0)
    images_out.cols = len(col_seeds) + 1
    return images_out
