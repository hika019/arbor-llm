"""cuBLASLt FP8 dW GEMM (beta=1 accumulate) の拡張ローダ.

`torch._scaled_mm` は beta=0 固定なので dW を別 tensor に書いてから param.grad へ
eager add するしかなく、その add が 1 update あたり (param bytes × 3 × micro-step 数)
の DRAM 往復になる (950M param / accum 8 で実測 60 ms = GPU 時間の 11%)。
cuBLASLt を直接呼び、dW GEMM の epilogue で gradient accumulation buffer へ
足し込む。

nvcc は不要 (host API のみ)。CUDA header/lib は torch が依存する pip の
`nvidia/cu13` (または cu12) から取るので、system の CUDA toolkit の version と
torch の CUDA version が食い違っていても build できる。
"""
from __future__ import annotations

import os
import sysconfig
from functools import lru_cache
from pathlib import Path

import torch


def _pip_cuda_dir() -> Path | None:
    site = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not site.is_dir():
        return None
    major = str(torch.version.cuda or "").split(".")[0]
    candidates = [site / f"cu{major}"] if major else []
    candidates += sorted(site.glob("cu*"), reverse=True)
    for c in candidates:
        if (c / "include" / "cublasLt.h").is_file():
            return c
    # cu12 系は package ごとに分かれている (nvidia/cublas, nvidia/cuda_runtime)
    if (site / "cublas" / "include" / "cublasLt.h").is_file():
        return site / "cublas"
    return None


@lru_cache(maxsize=None)
def _load_extension():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    build_dir = Path(os.environ.get("ARBOR_TORCH_EXTENSIONS_DIR", root / ".torch_extensions"))
    build_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent / "csrc" / "fp8_wgrad_lt.cpp"
    torch_lib = Path(torch.__file__).resolve().parent / "lib"

    include_paths: list[str] = []
    ldflags = [f"-L{torch_lib}", "-lc10_cuda", "-ltorch_cuda", f"-Wl,-rpath,{torch_lib}"]
    cuda_dir = _pip_cuda_dir()
    if cuda_dir is not None:
        include_paths.append(str(cuda_dir / "include"))
        lib_dir = cuda_dir / "lib"
        lt = sorted(lib_dir.glob("libcublasLt.so*"))
        if not lt:
            raise RuntimeError(f"libcublasLt.so not found under {lib_dir}")
        ldflags += [f"-L{lib_dir}", f"-l:{lt[0].name}", f"-Wl,-rpath,{lib_dir}"]
        # cu12 系: cuda_runtime は別 package
        rt = cuda_dir.parent / "cuda_runtime" / "include"
        if rt.is_dir():
            include_paths.append(str(rt))
    else:
        cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        include_paths.append(str(cuda_home / "include"))
        ldflags += [f"-L{cuda_home / 'lib64'}", "-lcublasLt"]

    # patch_starts_cuda と同じ理由で system gcc を使う (micromamba gcc の include 設定を避ける)
    env_backup = {k: os.environ.get(k) for k in ("CC", "CXX", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH")}
    custom_cc, custom_cxx = os.environ.get("ARBOR_EXT_CC"), os.environ.get("ARBOR_EXT_CXX")
    if custom_cc and custom_cxx:
        os.environ["CC"], os.environ["CXX"] = custom_cc, custom_cxx
    elif Path("/usr/bin/gcc").exists() and Path("/usr/bin/g++").exists():
        os.environ["CC"], os.environ["CXX"] = "/usr/bin/gcc", "/usr/bin/g++"
        os.environ.pop("C_INCLUDE_PATH", None)
        os.environ.pop("CPLUS_INCLUDE_PATH", None)
    try:
        return load(
            name="arbor_fp8_wgrad_lt",
            sources=[str(src)],
            build_directory=str(build_dir),
            extra_include_paths=include_paths,
            extra_cflags=["-O2", "-std=c++20"],
            extra_ldflags=ldflags,
            with_cuda=False,
            verbose=bool(int(os.environ.get("ARBOR_EXT_VERBOSE", "0"))),
        )
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fp8_wgrad_lt_available() -> bool:
    try:
        _load_extension()
        return True
    except Exception:  # noqa: BLE001 - 呼び出し側で fallback を選ぶ
        return False


@torch.library.custom_op("arbor::fp8_wgrad_lt", mutates_args=("out",))
def fp8_wgrad_lt(
    a_km: torch.Tensor,
    scale_a: torch.Tensor,
    b_nm: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    accumulate: bool,
) -> None:
    """out[N,K] (+)= b_nm[N,M] @ a_km[K,M]^T (tensorwise FP8, cuBLASLt beta=accumulate)."""
    _load_extension().fp8_wgrad_lt(a_km, scale_a, b_nm, scale_b, out, accumulate)


@fp8_wgrad_lt.register_fake
def _fp8_wgrad_lt_fake(a_km, scale_a, b_nm, scale_b, out, accumulate) -> None:
    return None
