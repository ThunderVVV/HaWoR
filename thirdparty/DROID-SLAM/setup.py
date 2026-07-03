from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

# On AMD ROCm/HIP the sources are hipified and compiled by clang++ (hipcc), which
# rejects NVIDIA-only `-gencode=arch=compute_*` flags. The target GPU arch is taken
# from the PYTORCH_ROCM_ARCH env var (e.g. gfx942) instead. Keep the CUDA gencode
# list only when building against a CUDA toolkit.
IS_ROCM = torch.version.hip is not None
CUDA_ARCH_FLAGS = [] if IS_ROCM else [
    '-gencode=arch=compute_60,code=sm_60',
    '-gencode=arch=compute_61,code=sm_61',
    '-gencode=arch=compute_70,code=sm_70',
    '-gencode=arch=compute_75,code=sm_75',
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_86,code=sm_86',
]

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/droid.cpp', 
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'] + CUDA_ARCH_FLAGS,
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'), 
                osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp', 
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O2'], 
                'nvcc': ['-O2'] + CUDA_ARCH_FLAGS,
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
