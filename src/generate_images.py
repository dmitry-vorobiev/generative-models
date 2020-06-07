import hydra
import logging
import math
import os
import torch
import torchvision

from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Dict

import models
from my_types import SampleRandomImages


rnd_sample_funcs: Dict[str, SampleRandomImages] = {
    'models.stylegan2.net.Generator': models.stylegan2.inference.sample_random_images
}

sample_funcs = {
    'random': rnd_sample_funcs
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


@hydra.main(config_path="../config/generate_images.yaml")
def main(conf: DictConfig):
    out_dir = conf.out.get('dir', os.path.join(os.getcwd(), 'generated_images'))
    make_dir(out_dir)
    logging.info("Saving images to {}".format(out_dir))

    device = torch.device(conf.sample.get('device', 'cpu'))
    G = load_model(conf.model.G, device)

    mode = conf.sample.mode
    sample_func = sample_funcs[mode][conf.model.G['class']]

    num_images = conf.sample.num_images
    bs = min(conf.sample.batch_size, num_images)
    num_images = math.ceil(num_images / bs) * bs

    dyn_range = tuple(conf.out.dynamic_range)
    prefix = conf.out.prefix
    pbar = tqdm(desc="Generating images ({})".format(mode), total=num_images, unit=' img')

    for i_batch in range(0, num_images, bs):
        images = sample_func(G, bs, device)
        for i_img, image in enumerate(images):
            image_idx = i_batch + i_img
            path = os.path.join(out_dir, '%s%06d.png' % (prefix, image_idx))
            torchvision.utils.save_image(image, path, normalize=True, range=dyn_range)
            pbar.update(1)
        del image, images

    pbar.close()
    print("DONE")


if __name__ == '__main__':
    main()
