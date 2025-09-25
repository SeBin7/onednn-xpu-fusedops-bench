from __future__ import annotations
import os
import torch

_PRECISION2DTYPE = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,  # (CUDA/XPU 일부 op 제한 주의)
}

def get_dtype(precision: str) -> torch.dtype:
    precision = (precision or "fp32").lower()
    return _PRECISION2DTYPE.get(precision, torch.float32)

def pick_device(device_str: str) -> torch.device:
    d = (device_str or "cpu").lower()
    if d == "xpu":
        if not torch.xpu.is_available():
            raise RuntimeError("XPU is not available. Check driver/runtime and SYCL_DEVICE_FILTER.")
        return torch.device("xpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")

def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def empty_cache(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

def describe_device(device: torch.device) -> str:
    if device.type == "xpu":
        return f"xpu (name={torch.xpu.get_device_name(0)})"
    if device.type == "cuda":
        return f"cuda (name={torch.cuda.get_device_name(0)})"
    if device.type == "cpu":
        # 안전 가드: 0 또는 음수 피함
        try:
            nt = max(1, int(os.environ.get("OMP_NUM_THREADS", "0") or torch.get_num_threads()))
        except Exception:
            nt = torch.get_num_threads()
        return f"cpu (threads={nt})"
    return device.type
