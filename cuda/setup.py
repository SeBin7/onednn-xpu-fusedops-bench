from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, shutil

arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.1")
arch_digits = arch.replace(".", "")

# Prefer g++-11 if present (CUDA 11.8 officially supports up to GCC 11)
ccbin = os.environ.get("CUDAHOSTCXX") or shutil.which("g++-11")

nvcc_flags = [
    "-O3",
    f"-gencode=arch=compute_{arch_digits},code=sm_{arch_digits}",
    "--use_fast_math",
]
cxx_flags = ["-O3", "-std=c++17"]

if ccbin:
    nvcc_flags += [f"-ccbin={ccbin}"]
else:
    # Fallback: allow newer host compiler (use at your own risk)
    nvcc_flags += ["-allow-unsupported-compiler"]

ext = CUDAExtension(
    name="linear_gelu_cuda",
    sources=["linear_gelu.cpp", "linear_gelu_kernel.cu"],
    extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
)

setup(
    name="linear_gelu_cuda",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
