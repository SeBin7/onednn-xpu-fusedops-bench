from __future__ import annotations
import os
import torch
from contextlib import contextmanager

# Map precision flag to torch dtype
_PRECISION2DTYPE = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,  # partial support across devices
}

def get_dtype(precision: str) -> torch.dtype:
    precision = (precision or "fp32").lower()
    return _PRECISION2DTYPE.get(precision, torch.float32)

def bf16_supported(device: torch.device) -> bool:
    """Quick check if bf16 math works on the given device."""
    try:
        a = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
        b = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
        _ = (a @ b).sum().item()
        return True
    except Exception:
        return False

@contextmanager
def autocast_ctx(device: torch.device, dtype: torch.dtype):
    """
    Optional autocast context for bf16.
    NOTE: XPU support for torch.autocast may vary by PyTorch build. Use only if verified.
    """
    if dtype is torch.bfloat16 and device.type in ("xpu", "cpu", "cuda"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    else:
        yield

def pick_device(device_str: str) -> torch.device:
    d = (device_str or "cpu").lower()
    if d == "xpu":
        if not torch.xpu.is_available():
            raise RuntimeError("XPU is not available. Check drivers/runtime and SYCL_DEVICE_FILTER.")
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
        # Ensure a positive thread count for logging
        try:
            nt_env = os.environ.get("OMP_NUM_THREADS", "")
            nt = int(nt_env) if nt_env else torch.get_num_threads()
            nt = max(1, nt)
        except Exception:
            nt = max(1, torch.get_num_threads())
        return f"cpu (threads={nt})"
    return device.type
