from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='stylegan2_custom_ops',
    ext_modules=[
        CUDAExtension('upfirdn_2d_op', [
            'upfirdn_2d.cpp',
            'upfirdn_2d_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
