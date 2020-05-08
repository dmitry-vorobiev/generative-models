import os

from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from typing import Tuple


def make_dataset(image_dir, extensions=None, is_valid_file=None):
    images = []
    image_dir = os.path.expanduser(image_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    if not os.path.isdir(image_dir):
        raise RuntimeError("Unable to read folder {}".format(dir))
    for root, _, fnames in sorted(os.walk(image_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                images.append(path)
    return images


class JustImages(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/xxx.ext
        root/123.ext
        root/nsdf3.ext
        root/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        samples (list): List of paths to images
    """

    def __init__(self, root, loader=default_loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(JustImages, self).__init__(root, transform=transform,
                                         target_transform=target_transform)
        samples = make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    def __getitem__(self, index: int) -> Tensor:
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)
