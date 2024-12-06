import contextlib
import datetime
import itertools
import os
import pathlib
import platform
import re
import subprocess

import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent


def glob(pattern):
    return [str(p) for p in root.glob(pattern)]


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        with contextlib.suppress(ValueError):
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)



if __name__ == "__main__":
    remove_unwanted_pytorch_nvcc_flags()
    # generate_build_meta()

    ext_modules = []
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="ops._kernels",
            sources=[
                "csrc/lora_ops.cc",
                "csrc/sgmv/sgmv_cutlass.cu",
            ],
            # + generate_flashinfer_cu(),
            include_dirs=[
                str(root.resolve() / "third_party/cutlass/include"),
                # str(root.resolve() / "third_party/flashinfer/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    )

    setuptools.setup(
        # version=get_version(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
    )