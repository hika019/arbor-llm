from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from shutil import which

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _load_extension():
    root = Path(__file__).resolve().parents[2]
    build_dir = Path(os.environ.get("ARBOR_TORCH_EXTENSIONS_DIR", root / ".torch_extensions"))
    build_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parent / "csrc"
    include_paths = []
    if Path("/usr/include/cuda_runtime.h").exists():
        include_paths.append("/usr/include")
    old_cc, old_cxx = os.environ.get("CC"), os.environ.get("CXX")
    custom_cc = os.environ.get("ARBOR_EXT_CC")
    custom_cxx = os.environ.get("ARBOR_EXT_CXX")
    gcc12, gxx12 = which("gcc-12"), which("g++-12")
    if custom_cc and custom_cxx:
        os.environ["CC"] = custom_cc
        os.environ["CXX"] = custom_cxx
    elif gcc12 and gxx12:
        os.environ["CC"] = gcc12
        os.environ["CXX"] = gxx12
    try:
        return load(
            name="arbor_patch_starts",
            sources=[str(src_dir / "patch_starts.cpp"), str(src_dir / "patch_starts.cu")],
            build_directory=str(build_dir),
            extra_include_paths=include_paths,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=bool(int(os.environ.get("ARBOR_EXT_VERBOSE", "0"))),
        )
    finally:
        if old_cc is None:
            os.environ.pop("CC", None)
        else:
            os.environ["CC"] = old_cc
        if old_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = old_cxx


def patch_starts_cuda(raw: torch.Tensor, min_len: int, max_len: int) -> torch.Tensor:
    return _load_extension().patch_starts_cuda(raw.contiguous(), int(min_len), int(max_len))
