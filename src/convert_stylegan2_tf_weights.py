"""
This script originates from another StyleGAN2 PyTorch implementation:
https://github.com/Tetratrio/stylegan2_pytorch/

Some parts were modified or adapted to work with the current implementation, others are
direct copies of the original code.

Original files:
 - run_convert_from_tf.py,
 - stylegan2/utils.py


Copyright 2020 Adrian Sahlman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import io
import math
import os
import pickle
import re
import requests
import torch

from torch import Tensor
from typing import Any, Dict, Mapping, Tuple

from models.stylegan2.net import Discriminator, Generator

SUPPORTED_MODELS = ['G_main', 'G_mapping', 'G_synthesis_stylegan2', 'D_stylegan2']

URLS = {
    'car-config-e': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-e.pkl',
    'car-config-f': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl',
    'cat-config-f': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-f.pkl',
    'church-config-f': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl',
    'ffhq-config-e': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-e.pkl',
    'ffhq-config-f': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
    'horse-config-f': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-horse-config-f.pkl',
    'car-config-e-Gskip-Dresnet': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl',
    'ffhq-config-e-Gskip-Dresnet': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl',
}


class AttributeDict(dict):
    """
    Dict where values can be accessed using attribute syntax.
    Same as "EasyDict" in the NVIDIA stylegan git repository.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __getstate__(self):
        return dict(**self)

    def __setstate__(self, state):
        self.update(**state)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(key, value) for key, value in self.items())
        )

    @classmethod
    def convert_dict_recursive(cls, obj):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                obj[key] = cls.convert_dict_recursive(obj[key])
            if not isinstance(obj, cls):
                return cls(**obj)
        return obj


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return AttributeDict
        return super(Unpickler, self).find_class(module, name)


def load_tf_models_file(path):
    with open(path, 'rb') as f:
        return Unpickler(f).load()


def load_tf_models_url(url):
    print('Downloading file {}...'.format(url))
    with requests.Session() as session:
        with session.get(url) as ret:
            fp = io.BytesIO(ret.content)
            return Unpickler(fp).load()


def stringify(d: dict) -> str:
    key_values = ["{}: {}".format(k, v) for k, v in d.items()]
    return ", ".join(key_values)


def error_if_unsupported(tf_class_name: str):
    if tf_class_name not in SUPPORTED_MODELS:
        raise AttributeError('Found model type {}. Allowed model types are: {}'
                             .format(tf_class_name, SUPPORTED_MODELS))


def handle_act_func_kwargs(kwargs: dict, key="nonlinearity"):
    if key in kwargs:
        if kwargs[key] != 'lrelu':
            raise ValueError("Found unsupported activation fn: {}".format(kwargs[key]))
        del kwargs[key]


def convert_kwargs(static_kwargs, mappings):
    # type: (Dict[str, Any], Mapping[str, str]) -> Dict[str, Any]
    kwargs = dict()
    for key, value in static_kwargs.items():
        if key in mappings:
            new_key = mappings[key]
            kwargs[new_key] = value
    return kwargs


def convert_from_tf(tf_state, impl: str, randomize_noise: bool) -> torch.nn.Module:
    tf_state = AttributeDict.convert_dict_recursive(tf_state)
    model_type = tf_state['build_func_name']
    error_if_unsupported(model_type)

    if model_type == 'G_main':
        kwargs = convert_kwargs(tf_state.static_kwargs, {
            'truncation_psi': 'truncation_psi',
            'truncation_cutoff': 'truncation_cutoff',
            'truncation_psi_val': 'truncation_psi',
            'truncation_cutoff_val': 'truncation_cutoff',
            'dlatent_avg_beta': 'w_avg_beta',
            'style_mixing_prob': 'p_style_mix',
        })

        kwargs_mapping, params_mapping = parse_G_mapping(tf_state.components.mapping, impl)
        kwargs.update(kwargs_mapping)

        kwargs_synthesis, params_synthesis = parse_G_synthesis(tf_state.components.synthesis, impl)
        kwargs.update(kwargs_synthesis)

        G = Generator(impl=impl, randomize_noise=randomize_noise, **kwargs)
        G.requires_grad_(False)

        for var_name, var in tf_state.variables:
            if 'dlatent_avg' in var_name:
                G.w_avg.copy_(torch.from_numpy(var))
                break

        for name, param in G.mapping.named_parameters():
            value = params_mapping[name]
            param.copy_(value)

        for name, param in G.synthesis.named_parameters():
            value = params_synthesis[name]
            param.copy_(value)

        print('{} attributes: {}'.format(tf_state.name, stringify(kwargs)))
        return G

    if model_type == 'D_stylegan2':
        output_vars = {}
        conv_vars = {}
        for var_name, var in tf_state.variables:
            if var_name.startswith('Output'):
                output_vars[var_name[7:]] = var  # len('Output/'): 7
            else:
                group_vars_by_res_log2(conv_vars, var_name, var)

        kwargs = convert_kwargs(tf_state.static_kwargs, {
            'num_channels': 'img_channels',
            'resolution': 'img_res',
            'label_size': 'num_classes',
            'fmap_base': 'fmap_base',
            'fmap_decay': 'fmap_decay',
            'fmap_min': 'fmap_min',
            'fmap_max': 'fmap_max',
            'mbstd_group_size': 'mbstd_group_size',
            'mbstd_num_features': 'mbstd_num_features',
            'nonlinearity': 'nonlinearity',
            'resample_kernel': 'blur_kernel',
        })
        handle_act_func_kwargs(kwargs)
        params = dict()

        def convert_layer(values, torch_pref, tf_pref, has_bias=True, tfm_weight=conv_weight):
            params[f'{torch_pref}.weight'] = tfm_weight(values[tf_pref + 'weight'])
            if has_bias:
                params[f'{torch_pref}.bias'] = bias(values[tf_pref + 'bias'])

        res_log2 = max(conv_vars.keys())
        convert_layer(conv_vars[res_log2], 'layers.0.conv', 'FromRGB/')

        for i in range(3, res_log2 + 1):
            idx = res_log2 - i + 1
            vars_i = conv_vars[i]
            convert_layer(vars_i, f'layers.{idx}.conv.0', 'Conv0/')
            convert_layer(vars_i, f'layers.{idx}.conv.2.conv', 'Conv1_down/')
            convert_layer(vars_i, f'layers.{idx}.skip.0.conv', 'Skip/', has_bias=False)

        convert_layer(conv_vars[2], f'layers.{res_log2}', 'Conv/')
        convert_layer(conv_vars[2], f'layers.{res_log2 + 3}', 'Dense0/', tfm_weight=linear_weight)
        convert_layer(output_vars, f'layers.{res_log2 + 5}', '', tfm_weight=linear_weight)

        D = Discriminator(impl='ref', **kwargs)
        D.requires_grad_(False)

        for name, param in D.named_parameters():
            value = params[name]
            assert value.shape == param.shape
            param.copy_(value)

        print('{} attributes: {}'.format(tf_state.name, stringify(kwargs)))
        return D

    raise NotImplementedError(model_type)


def group_vars_by_res_log2(conv_vars, var_name, var):
    # type: (Dict[int, Dict[str, Tensor]], str, 'np.ndarray') -> None
    match = re.search('(\d+)x[0-9]+/*', var_name)
    res = int(match.groups()[0])
    res_log2 = int(math.log2(res))

    if res_log2 not in conv_vars:
        conv_vars[res_log2] = dict()

    var_name = var_name.replace('{}x{}/'.format(res, res), '')
    conv_vars[res_log2][var_name] = var


def bias(b) -> Tensor:
    return torch.from_numpy(b)


def linear_weight(weight) -> Tensor:
    return torch.from_numpy(weight.T).contiguous()


def conv_weight(weight, transposed=False) -> Tensor:
    dims = [2, 3, 0, 1] if transposed else [3, 2, 0, 1]
    w = torch.from_numpy(weight).permute(*dims)
    if transposed:
        w = w.flip(dims=(2, 3))
    return w.contiguous()


def parse_G_mapping(tf_state: AttributeDict, impl: str) -> Tuple[Dict[str, Any], Dict[str, Tensor]]:
    model_type = tf_state['build_func_name']
    error_if_unsupported(model_type)

    kwargs = convert_kwargs(tf_state.static_kwargs, {
        'latent_size': 'latent_dim',
        'label_size': 'num_classes',
        'dlatent_size': 'style_dim',
        'mapping_layers': 'num_mapping_layers',
        'mapping_fmaps': 'mapping_hidden_dim',
        'normalize_latents': 'normalize_latent',
        'mapping_nonlinearity': 'nonlinearity',
    })
    handle_act_func_kwargs(kwargs)

    norm_latents = True
    if 'normalize_latent' in kwargs:
        norm_latents = kwargs['normalize_latent']

    params = dict()
    fused_bias_act = impl == "cuda_full"

    for var_name, var in tf_state.variables:
        if re.match('Dense[0-9]+/[a-zA-Z]*', var_name):
            match = re.search('Dense(\d+)/[a-zA-Z]*', var_name)
            layer_idx = int(match.groups()[0])
            idx = int(norm_latents) + layer_idx * 2

            if var_name.endswith('weight'):
                params[f'layers.{idx}.weight'] = linear_weight(var)
            elif var_name.endswith('bias'):
                # TODO: how to make a unified state structure?
                params[f'layers.{idx + int(fused_bias_act)}.bias'] = bias(var)

        elif var_name == 'LabelConcat/weight':
            params[f'cat_label.weight'] = linear_weight(var)

    return kwargs, params


def parse_G_synthesis(tf_state, impl):
    # type: (AttributeDict, str) -> Tuple[Dict[str, Any], Dict[str, Tensor]]
    model_type = tf_state['build_func_name']
    error_if_unsupported(model_type)

    kwargs = convert_kwargs(tf_state.static_kwargs, {
        'dlatent_size': 'style_dim',
        'num_channels': 'img_channels',
        'resolution': 'img_res',
        'fmap_base': 'fmap_base',
        'fmap_decay': 'fmap_decay',
        'fmap_min': 'fmap_min',
        'fmap_max': 'fmap_max',
        'randomize_noise': 'randomize_noise',
        'nonlinearity': 'nonlinearity',
        'resample_kernel': 'blur_kernel',
    })
    handle_act_func_kwargs(kwargs)

    params = dict()
    conv_vars = dict()
    noise_vars = list()

    for var_name, var in tf_state.variables:
        if var_name.startswith('noise'):
            noise_vars.append(var)
        else:
            group_vars_by_res_log2(conv_vars, var_name, var)
    noise_vars = sorted(noise_vars, key=lambda x: x.shape[-1])

    def convert_layer(values, torch_pref, tf_pref, noise=None, up=False, fused_bias_act=False):
        mod_name = f'{torch_pref}.style.'
        params[mod_name + 'weight'] = linear_weight(values[tf_pref + 'mod_weight'])
        params[mod_name + 'bias'] = bias(values[tf_pref + 'mod_bias']) + 1

        params[torch_pref + '.conv.weight'] = conv_weight(values[tf_pref + 'weight'], transposed=up)
        if fused_bias_act:
            bias_pref = torch_pref + '.act_fn.'
        else:
            bias_pref = torch_pref + '.conv.'
        params[bias_pref + 'bias'] = bias(values[tf_pref + 'bias'])

        if noise is not None:
            noise_name = f'{torch_pref}.add_noise.'
            params[noise_name + 'noise'] = torch.from_numpy(noise)
            params[noise_name + 'gain'] = torch.tensor(values[tf_pref + 'noise_strength'])

    fused_bias_act = impl == "cuda_full"
    early_layers = conv_vars[2]
    params['input.weight'] = torch.from_numpy(early_layers['Const/const'])
    convert_layer(early_layers, 'main.0', 'Conv/', noise=noise_vars[0],
                  fused_bias_act=fused_bias_act)
    convert_layer(early_layers, 'outs.0', 'ToRGB/')

    for i in range(3, max(conv_vars.keys()) + 1):
        idx = (i - 3) * 2 + 1
        for j, (tf_pref, idx) in enumerate(zip(['Conv0_up/', 'Conv1/'], [idx, idx + 1])):
            convert_layer(conv_vars[i], f'main.{idx}', tf_pref,
                          noise=noise_vars[idx],
                          up=(j == 0),
                          fused_bias_act=fused_bias_act)
        convert_layer(conv_vars[i], f'outs.{i - 2}', 'ToRGB/')

    return kwargs, params


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Convert tensorflow StyleGAN2 model to PyTorch.',
        epilog='Pretrained models that can be downloaded:\n{}'.format('\n'.join(URLS.keys())))

    parser.add_argument('-i', '--input', type=str, default=None,
                        help='File path to pickled tensorflow models.')
    parser.add_argument('-d', '--download', type=str, default=None,
                        help='Download the specified pretrained model. '
                             'Use --help to see a list of available models.')
    parser.add_argument('-o', '--output', type=str, nargs='*', default=['.'],
                        help='One or more output file paths. Alternatively a directory path, '
                             'where all models will be saved. Default: current directory')
    parser.add_argument('-I', '--impl', type=str, default='ref',
                        choices=['ref', 'cuda', 'cuda_full'],
                        help='Implementation (there are some differences in param names')
    parser.add_argument('-n', '--noise', type=str, default='random',
                        choices=['const', 'random'],
                        help='Which noise layers to use')
    return parser


def main():
    args = get_arg_parser().parse_args()
    input_path = args.input
    model_name = args.download
    save_paths = args.output
    random_noise = args.noise == 'random'

    if not any([input_path, model_name]):
        raise AttributeError(
            'Incorrect input format. One of the [-i, --input] or [-d, --download] '
            'args must be specified.')

    if bool(input_path) == bool(model_name):
        raise AttributeError(
            'Incorrect input format. Can only take either one input filepath to a pickled '
            'tensorflow model or a model name to download.')

    if input_path:
        unpickled = load_tf_models_file(input_path)
    else:
        if model_name not in URLS.keys():
            raise AttributeError('Unknown model {}. Use --help for list of models.'
                                 .format(model_name))
        unpickled = load_tf_models_url(URLS[model_name])
    if not isinstance(unpickled, (tuple, list)):
        unpickled = [unpickled]

    print('Converting tensorflow models and saving them...')
    converted = [convert_from_tf(tf_state, args.impl, random_noise) for tf_state in unpickled]

    if len(save_paths) == 1:
        dir_path = save_paths[0]
        if os.path.isfile(dir_path) or os.path.splitext(dir_path)[-1]:
            raise AttributeError(
                'Please specify correct directory to save output files '
                'or provide a list of paths for each file')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_paths = [os.path.join(dir_path, tf_state['name'] + '.pth')
                      for tf_state in unpickled]

    if len(save_paths) != len(converted):
        raise AttributeError(
            'Found {} models in pickled file but only {} output paths '
            'were given.'.format(len(converted), len(save_paths)))

    for out_path, torch_model in zip(save_paths, converted):
        torch.save(torch_model.state_dict(), out_path)

    print('Done!')


if __name__ == '__main__':
    main()
