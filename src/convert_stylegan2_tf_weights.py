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

from models.stylegan2.net import Generator

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


def error_if_unsupported(tf_class_name):
    if tf_class_name not in SUPPORTED_MODELS:
        raise AttributeError('Found model type {}. Allowed model types are: {}'
                             .format(tf_class_name, SUPPORTED_MODELS))


def convert_kwargs(static_kwargs, mappings):
    kwargs = dict()
    for key, value in static_kwargs.items():
        if key in mappings:
            new_key = mappings[key]
            kwargs[new_key] = value
    return kwargs


def convert_from_tf(tf_state):
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

        kwargs_mapping, params_mapping = parse_G_mapping(tf_state.components.mapping)
        kwargs.update(kwargs_mapping)

        kwargs_synthesis, params_synthesis = parse_G_synthesis(tf_state.components.synthesis)
        kwargs.update(kwargs_synthesis)

        G = Generator(impl='ref', **kwargs)
        G.requires_grad_(False)

        for name, param in G.mapping.named_parameters(recurse=True):
            value = params_mapping[name]
            param.copy_(value)

        for name, param in G.synthesis.named_parameters(recurse=True):
            value = params_synthesis[name]
            param.copy_(value)

        print('Generator attributes: {}'.format(kwargs))
        return G

    return model_type


def bias(b):
    return torch.from_numpy(b)


def linear_weight(weight):
    return torch.from_numpy(weight.T).contiguous()


def conv_weight(weight, transposed=False):
    dims = [2, 3, 0, 1] if transposed else [3, 2, 0, 1]
    w = torch.from_numpy(weight).permute(*dims)
    if transposed:
        w = w.flip(dims=(2, 3))
    return w.contiguous()


def parse_G_mapping(tf_state):
    tf_state = AttributeDict.convert_dict_recursive(tf_state)
    model_type = tf_state['build_func_name']
    error_if_unsupported(model_type)

    kwargs = convert_kwargs(tf_state.static_kwargs, {
        'latent_size': 'latent_dim',
        'label_size': 'num_classes',
        'dlatent_size': 'style_dim',
        'mapping_layers': 'num_mapping_layers',
        'mapping_fmaps': 'mapping_hidden_dim',
        'normalize_latents': 'normalize_latent',
        'mapping_nonlinearity': 'mapping_nonlinearity',
    })

    if 'mapping_nonlinearity' in kwargs:
        act_fn = kwargs['mapping_nonlinearity']
        if act_fn != 'lrelu':
            raise ValueError("Found unsupported activation fn: {}".format(act_fn))
        del kwargs['mapping_nonlinearity']

    norm_latents = True
    if 'normalize_latent' in kwargs:
        norm_latents = kwargs['normalize_latent']

    params = dict()

    for var_name, var in tf_state.variables:
        if re.match('Dense[0-9]+/[a-zA-Z]*', var_name):
            match = re.search('Dense(\d+)/[a-zA-Z]*', var_name)
            layer_idx = int(match.groups()[0])
            idx = int(norm_latents) + layer_idx * 2
            torch_pref = f'layers.{idx}'

            if var_name.endswith('weight'):
                params[f'{torch_pref}.weight'] = linear_weight(var)
            elif var_name.endswith('bias'):
                params[f'{torch_pref}.bias'] = bias(var)

        elif var_name == 'LabelConcat/weight':
            params[f'cat_label.weight'] = linear_weight(var)

    return kwargs, params


def parse_G_synthesis(tf_state):
    tf_state = AttributeDict.convert_dict_recursive(tf_state)
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
        'nonlinearity': 'nonlinearity',
        'resample_kernel': 'blur_kernel',
    })

    if 'nonlinearity' in kwargs:
        act_fn = kwargs['nonlinearity']
        if act_fn != 'lrelu':
            raise ValueError("Found unsupported activation fn: {}".format(act_fn))
        del kwargs['nonlinearity']

    params = dict()
    conv_vars = dict()

    for var_name, var in tf_state.variables:
        if var_name.startswith('noise'):
            continue
        else:
            match = re.search('(\d+)x[0-9]+/*', var_name)
            res = int(match.groups()[0])
            res_log2 = int(math.log2(res))

            if res_log2 not in conv_vars:
                conv_vars[res_log2] = dict()

            var_name = var_name.replace('{}x{}/'.format(res, res), '')
            conv_vars[res_log2][var_name] = var

    def convert_layer(values, torch_pref, tf_pref, noise=False, up=False):
        mod_name = f'{torch_pref}.style'
        params[f'{mod_name}.weight'] = linear_weight(values[f'{tf_pref}/mod_weight'])
        params[f'{mod_name}.bias'] = bias(values[f'{tf_pref}/mod_bias']) + 1

        conv_name = f'{torch_pref}.conv'
        params[f'{conv_name}.weight'] = conv_weight(values[f'{tf_pref}/weight'], transposed=up)
        params[f'{conv_name}.bias'] = bias(values[f'{tf_pref}/bias'])

        if noise:
            params[f'{torch_pref}.add_noise.gain'] = torch.tensor(
                values[f'{tf_pref}/noise_strength'])

    params['input.weight'] = torch.from_numpy(conv_vars[2]['Const/const'])
    convert_layer(conv_vars[2], f'main.0', 'Conv', noise=True)
    convert_layer(conv_vars[2], f'outs.0', 'ToRGB')

    for i in range(3, max(conv_vars.keys()) + 1):
        idx = (i - 3) * 2 + 1
        for j, (tf_pref, idx) in enumerate(zip(['Conv0_up', 'Conv1'], [idx, idx + 1])):
            convert_layer(conv_vars[i], f'main.{idx}', tf_pref, noise=True, up=(j == 0))
        convert_layer(conv_vars[i], f'outs.{i - 2}', 'ToRGB')

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
    return parser


def main():
    args = get_arg_parser().parse_args()
    input_path = args.input
    model_name = args.download
    save_path = args.output

    if not (bool(input_path) or bool(model_name)):
        raise AttributeError('Incorrect input format. One of the [-i, --input] or [-d, --download] '
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
    converted = [convert_from_tf(tf_state) for tf_state in unpickled]

    if len(save_path) == 1:
        save_path = save_path[0]
        if os.path.isdir(save_path) or not os.path.splitext(save_path)[-1]:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for tf_state, torch_model in zip(unpickled, converted):
                path = os.path.join(save_path, tf_state['name'] + '.pth')
                if isinstance(torch_model, torch.nn.Module):
                    torch.save(torch_model.state_dict(), path)
    else:
        if len(save_path) != len(converted):
            raise AttributeError('Found {} models in pickled file but only {} output paths '
                                 'were given.'.format(len(converted), len(save_path)))
        for out_path, torch_model in zip(save_path, converted):
            if isinstance(torch_model, torch.nn.Module):
                torch.save(torch_model.state_dict(), out_path)
    print('Done!')


if __name__ == '__main__':
    main()
