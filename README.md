# generative-models

my playground for generative modeling

## Requirements

The code was developed and tested under the environment:

* OS: Ubuntu 18.04.4 LTS (5.0.0-37-generic)
* CUDA 10.1.243
* PyTorch 1.4.0

## Installation

```shell script
pip install -r requirements.txt
```

## Dataset

To download the FFHQ dataset, please, refer to the [original repo](https://github.com/NVlabs/ffhq-dataset). 

To point the script to your local data folder you will have to update `data.root` setting in the `config/train_gan.yaml` 
or pass it as a command line argument:

```shell script
python src/train_gan.py data.root=/your/local/dir
```


## Usage

Launch distributed training on GPUs:

```shell script
python -m torch.distributed.launch --nproc_per_node=2 --use_env src/train_gan.py distributed.backend=nccl
```

It's important to run `torch.distributed.launch` with `--use_env`, otherwise [hydra](https://github.com/facebookresearch/hydra) will yell at you for passing unrecognized arguments.

## License

Some parts of this work are derivatives and as such they are licensed under the same license terms as the original work. 

**Stylegan2**: 
[original code](https://github.com/NVlabs/stylegan2), 
[NVIDIA's license (.txt)](https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt), 
[NVIDIA's license (.html)](https://nvlabs.github.io/stylegan2/license.html)

## Citation

```
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```
