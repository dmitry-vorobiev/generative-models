import hydra
import logging
import math
import os
import torch
import torchvision

from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.utils import save_image
from typing import Dict

import models
from my_types import SampleRandomImages, TensorGrid
from utils.config import read_int_list


rnd_sample_funcs: Dict[str, SampleRandomImages] = {
    'models.stylegan2.net.Generator': models.stylegan2.inference.sample_random_images
}

style_mixing_funcs = {
    'models.stylegan2.net.Generator': models.stylegan2.inference.mix_styles
}

sample_funcs = {
    'random': rnd_sample_funcs,
    'style-mixing': style_mixing_funcs
}


def make_dir(dir_path: str) -> None:
    if os.path.isfile(dir_path) or os.path.splitext(dir_path)[-1]:
        raise AttributeError('{} is not a directory.'.format(dir_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_model(conf: DictConfig, device: torch.device) -> torch.nn.Module:
    G: torch.nn.Module = instantiate(conf).to(device)
    logging.info("Loading generator weights from {}".format(conf.weights))
    state_dict = torch.load(conf.weights)
    G.load_state_dict(state_dict)
    G.requires_grad_(False)
    G.eval()
    return G


def gen_random_images(G: torch.nn.Module, out_dir: str, conf: DictConfig, device=None):
    num_images = conf.sample.num_images
    bs = min(conf.sample.batch_size, num_images)
    num_images = math.ceil(num_images / bs) * bs

    dynamic_range = tuple(conf.out.range)
    prefix = conf.out.prefix
    pbar = tqdm(desc="Generating images (random)", total=num_images, unit=' img')
    cols, rows = conf.out.cols, conf.out.rows
    sheet_size = cols * rows
    cpu = torch.device('cpu')

    sample_funcs: Dict[str, SampleRandomImages] = {
        'models.stylegan2.net.Generator': models.stylegan2.inference.sample_random_images
    }
    model = conf.model.G['class']
    if model not in sample_funcs:
        raise NotImplementedError("Random image sampling for {}".format(model))
    sample_func = sample_funcs[model]

    def save(images, start_idx, end_idx):
        if len(images) > 1:
            file = '{}{:06d}_{:06d}.png'.format(prefix, start_idx, end_idx)
        else:
            file = '{}{:06d}.png'.format(prefix, start_idx)
        path = os.path.join(out_dir, file)
        save_image(images, path, nrow=cols, normalize=True, range=dynamic_range)
        pbar.update(len(images))

    i_start = 0
    prev_images = None
    for i_batch in range(0, num_images, bs):
        images = sample_func(G, bs, device)

        if prev_images is not None:
            images = images.to(cpu)
            images = torch.cat([prev_images, images], dim=0)
            prev_images = None

        n = len(images)
        if n % sheet_size:
            n = n // sheet_size * sheet_size
            prev_images = images[n:].to(cpu)
            images = images[:n].to(cpu)

        for i in range(0, n, sheet_size):
            images_sheet = images[i: i + sheet_size]
            i_end = i_start + len(images_sheet)
            save(images_sheet, i_start, i_end)
            i_start = i_end
            del images_sheet
        del images

    if prev_images is not None:
        i_end = i_start + len(prev_images)
        save(prev_images, i_start, i_end)

    pbar.close()


def gen_style_mixed_images(G: torch.nn.Module, out_dir: str, conf: DictConfig, device=None):
    bs = conf.sample.batch_size
    dynamic_range = tuple(conf.out.range)
    prefix = conf.out.prefix
    cols, rows = conf.out.cols, conf.out.rows

    if rows * cols < 2:
        raise AttributeError("Image grid ({}, {}) should have multiple entries".format(rows, cols))

    mix_funcs = {
        'models.stylegan2.net.Generator': models.stylegan2.inference.mix_styles
    }
    model = conf.model.G['class']
    if model not in mix_funcs:
        raise NotImplementedError("Style mixing for {}".format(model))
    mix_func = mix_funcs[model]

    def gen_random_seeds(n):
        return list(torch.randint(int(2e+10), [n]))

    options = conf.style_mixing
    row_seeds = options.get('row_seeds', gen_random_seeds(rows))
    col_seeds = options.get('col_seeds', gen_random_seeds(cols))
    style_layers = read_int_list(options, 'style_layers')
    logging.info("Using {} layers for style mixing".format(", ".join(map(str, style_layers))))

    image_dict: TensorGrid = mix_func(G, bs, device=device,
                                      row_seeds=row_seeds,
                                      col_seeds=col_seeds,
                                      style_layers=style_layers)

    # placing original row images
    empty = torch.ones_like(image_dict[(row_seeds[0], col_seeds[0])])
    images = [empty] + [image_dict[(s, s)] for s in col_seeds]

    for row_seed in row_seeds:
        # adding original column image at the beginning of each row
        image = image_dict[(row_seed, row_seed)]
        images.append(image)

        for col_seed in col_seeds:
            image = image_dict[(row_seed, col_seed)]
            images.append(image)

    path = os.path.join(out_dir, '{}{}x{}.png'.format(prefix, rows, cols))
    images = torch.cat(images, dim=0)
    save_image(images, path, nrow=cols + 1, normalize=True, range=dynamic_range)


@hydra.main(config_path="../config/generate_images.yaml")
def main(conf: DictConfig):
    out_dir = conf.out.get('dir', os.path.join(os.getcwd(), 'generated_images'))
    make_dir(out_dir)
    logging.info("Saving images to {}".format(out_dir))

    mode = conf.sample.mode
    device = torch.device(conf.sample.get('device', 'cpu'))
    G = load_model(conf.model.G, device)

    if mode == 'random':
        gen_random_images(G, out_dir, conf, device=device)
    elif mode == 'style-mixing':
        gen_style_mixed_images(G, out_dir, conf, device=device)
    else:
        raise AttributeError("Unknown sample mode: {}".format(mode))

    print("DONE")


if __name__ == '__main__':
    main()
