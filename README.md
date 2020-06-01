# Generative models

My playground for generative modeling.

## Models:
* [StyleGAN2](src/models/stylegan2/README.md)

## Requirements

The code was developed and tested under the environment:

* OS: Ubuntu 18.04.4 LTS (5.0.0-37-generic)
* CUDA 10.1.243
* NVIDIA's drivers 435.21
* Conda 4.8.3
* Python 3.7.7
* PyTorch 1.4.0

## Installation

```shell script
pip install -r requirements.txt
```

## Configuration

Almost all configuration is made using [Hydra framework](https://github.com/facebookresearch/hydra) 
 from Facebook Research.

To change various settings you can either edit *.yaml* files 
in the `config` folder or pass corresponding params to the command line.
The second option is useful for quick testing.

For more information about configuration options available please visit [Hydra docs](https://hydra.cc/).

### Dataset

To point the train script to your local data folder update `data.root` setting in the `config/train_gan.yaml` 
or pass it as a command line argument:

```shell script
python src/train_gan.py data.root=/path/to/local/dir
```

### Model

The are a few predefined model presets available. You can find them in the `config/model` folder. 
Update `defaults.model` entry in the `config/train_gan.yaml` 
or pass `model` arg with a name of preset you want to use:

```shell script
python src/train_gan.py model=stylegan2/config_f
```

Don't forget to change model resolution `model.*.params.img_res` and 
number of classes `model.*.params.num_classes` according to your dataset.


## Usage

### Multi-GPU training
Launch distributed training on GPUs:

```shell script
python -m torch.distributed.launch --nproc_per_node=2 --use_env src/train_gan.py
```

It's important to run `torch.distributed.launch` with `--use_env`, 
otherwise [hydra](https://github.com/facebookresearch/hydra) will yell 
at you for passing unrecognized arguments.

## License

Some parts of this work are derivatives and as such they are licensed 
under the same license terms as the original work. See corresponding README.md
for details:

* [StyleGAN2](src/models/stylegan2/README.md#License)

## Citation

```bibtex
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```
