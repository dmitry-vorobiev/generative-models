from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='upfirdn_2d',
    ext_modules=[
        CUDAExtension('upfirdn_2d_cuda', [
            'upfirdn_2d_torch.cpp',
            'upfirdn_2d.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
