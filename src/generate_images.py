import hydra
import os
import torch
import torchvision

from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Dict

import models.stylegan2.inference
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
    state_dict = torch.load(conf.weights)
    G.load_state_dict(state_dict)
    G.requires_grad_(False)
    G.eval()
    return G


@hydra.main(config_path="../config/generate_images.yaml")
def main(conf: DictConfig):
    out_dir = conf.get('out_dir', os.path.join(os.getcwd(), 'generated_images'))
    make_dir(out_dir)

    device = torch.device(conf.get('device', 'cpu'))
    G = load_model(conf.model.G, device)

    sample_func = sample_funcs[conf.mode][conf.model.G['class']]
    bs = conf.batch_size
    dyn_range = tuple(conf.dynamic_range)
    prefix = conf.file_prefix
    pbar = tqdm(desc="Generating images ({})".format(conf.mode), total=conf.num_images,
                unit=' img')

    for i_batch in range(0, conf.num_images, bs):
        images = sample_func(G, bs, device)

        for i_img, image in enumerate(images):
            image_idx = i_batch * bs + i_img
            path = os.path.join(out_dir, '%s%06d.png' % (prefix, image_idx))
            torchvision.utils.save_image(image, path, normalize=True, range=dyn_range)
            pbar.update(1)
        del images, image

    pbar.close()
    print("DONE")


if __name__ == '__main__':
    main()
