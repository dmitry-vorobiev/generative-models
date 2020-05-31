# StyleGAN 2

## Description

This is an unofficial PyTorch port of Nvidia's [StyleGAN2](https://github.com/NVlabs/stylegan2) model, 
originally built on top of TensorFlow v.1.14.

It supports new StyleGAN2 architecture with skip connections in G 
and residual blocks in D, which corresponds to `config-e` and `config-f` 
from the original work.

Not all of the original functionality is present here (but may be added in the future).

## Details

There are four implementations available, which are specified by `impl` arg. 
Most of them should produce very similar (if not identical) results. 
The main differences are execution speed and memory consumption 
(though I haven't tested memory footprint thoroughly):

* **torch** version is an attempt to implement the StyleGAN 2 architecture using 
`torch.nn.functional.interpolate` as a way to resize feature maps.
* **ref** version implements `upfirdn_2d_ref` op from the original TF repo using
standard torch primitives. It's a bit slower than **torch** version.
* **cuda** version uses custom `upfirdn_2d` CUDA kernel from the original TF repo 
and considerably faster than **torch** version.
* **cuda_full** is similar to **cuda**, but adds custom `fused_bias_act` CUDA kernel.

## Usage

### Dataset
To download the FFHQ dataset, please, refer to the [original repo](https://github.com/NVlabs/ffhq-dataset). 

### Use pretrained weights from TF
To convert pretrained StyleGAN 2 weights from the official TF distribution run this in the shell:

```shell script
python src/convert_stylegan2_tf_weights.py -d ffhq-config-e -o /path/to/local/dir -I cuda
```

## License

Original Nvidia's license is applied for most of the stuff:

[LICENSE - txt version](https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt), 

[LICENSE - html](https://nvlabs.github.io/stylegan2/license.html).

Original script for porting TensorFlow weights to PyTorch was borrowed from 
[Tetratrio/stylegan2_pytorch](https://github.com/Tetratrio/stylegan2_pytorch) repo 
and adapted to work with current implementation. 
[Original license](https://github.com/Tetratrio/stylegan2_pytorch/blob/master/LICENSE.txt).

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
