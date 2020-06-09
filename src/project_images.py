import hydra
import torch

from omegaconf import DictConfig


@hydra.main(config_path="../config/project_images.yaml")
def main(conf: DictConfig):
    pass


if __name__ == '__main__':
    main()
