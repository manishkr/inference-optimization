import os
from pathlib import Path

from build import wheel_cuda_home

if "CUDA_HOME" not in os.environ and (home := wheel_cuda_home()):
    os.environ["CUDA_HOME"] = str(home)

from setuptools import setup  # noqa: E402
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension  # noqa: E402

ARCHS = [a.strip() for a in os.environ.get("NESTED_KV_CUDA_ARCHS", "75;90").split(";") if a.strip()]
LOWEST = min(ARCHS, key=int)
GENCODE = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in ARCHS]
GENCODE.append(f"-gencode=arch=compute_{LOWEST},code=compute_{LOWEST}")

extension = CUDAExtension(
    "_nested_kv",
    ["csrc/api.cpp", "csrc/scores_fast.cu", "csrc/scores_portable.cu"],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--threads", "2",
            "-lineinfo",
            *GENCODE,
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
    },
)

cuda_lib = Path(CUDA_HOME) / "lib"
if not (cuda_lib / "libcudart.so").exists():
    versioned = sorted(cuda_lib.glob("libcudart.so.*"))
    if versioned:
        extension.libraries = [lib for lib in extension.libraries if lib != "cudart"]
        extension.extra_link_args = [f"-l:{versioned[0].name}"]

setup(
    name="nested-kv",
    version="0.1.0",
    packages=[],
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
