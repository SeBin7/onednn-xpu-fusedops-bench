from __future__ import annotations
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import myops_xpu  # C++/SYCL fused op (Linear+Bias+GELU)
except Exception:
    myops_xpu = None

def has_myops() -> bool:
    return myops_xpu is not None

def _get_param_version(p: torch.Tensor) -> int:
    return int(getattr(p, "_version", 0))

def _cast_params_cached(mod: nn.Module, *, target_dtype: torch.dtype, device: torch.device):
    """
    weight/bias를 (가능하면) 디바이스에서 target_dtype으로 캐스팅해 캐시한다.
    UR 에러가 날 때만 CPU 경유 폴백을 사용한다.
    캐시 키: (weight._version, target_dtype, device)
    """
    w, b = mod.weight, mod.bias
    ver = _get_param_version(w)
    key = (ver, target_dtype, device.type, device.index)
    cache = getattr(mod, "_cast_cache", None)
    if isinstance(cache, dict) and cache.get("key") == key:
        return cache["w"], cache["b"]

    # 우선 디바이스에서 캐스팅 시도
    try:
        w_cast = w.to(device=device, dtype=target_dtype, non_blocking=True)
        b_cast = b.to(device=device, dtype=target_dtype, non_blocking=True) if b is not None else None
    except RuntimeError as e:
        if "UR error" not in str(e):
            raise
        # 최후의 수단: CPU에서 dtype 맞춘 뒤 디바이스로 복사
        w_cast = w.to("cpu", dtype=target_dtype).to(device, non_blocking=True)
        b_cast = b.to("cpu", dtype=target_dtype).to(device, non_blocking=True) if b is not None else None

    cache = {"key": key, "w": w_cast, "b": b_cast}
    setattr(mod, "_cast_cache", cache)
    return w_cast, b_cast

class _LinearView(nn.Module):
    """safe_linear에 넘길 weight/bias 뷰"""
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.weight = weight
        self.bias = bias

def safe_linear(x: torch.Tensor, linear: nn.Module) -> torch.Tensor:
    """
    XPU에서 dtype 미스매치 시 UR 에러 피하려고:
      - 가능하면 **디바이스에서 캐스팅**
      - 안 되면 그때만 CPU 경유
    """
    w, b = linear.weight, linear.bias
    if x.dtype == w.dtype:
        return F.linear(x, w, b)

    if x.device.type == "xpu":
        w_cast, b_cast = _cast_params_cached(linear, target_dtype=x.dtype, device=x.device)
        return F.linear(x, w_cast, b_cast)
    else:
        w_cast = w.to(dtype=x.dtype)
        b_cast = b.to(dtype=x.dtype) if b is not None else None
        return F.linear(x, w_cast, b_cast)

class LinearGELU_Torch(nn.Module):
    """PyTorch 경로: y = GELU(x @ W^T + b)"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, +bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = safe_linear(x, _LinearView(self.weight, self.bias))
        return F.gelu(y)

class LinearGELU_Fused(nn.Module):
    """
    - XPU + myops_xpu 있을 때 fused 경로(가능 시) 사용
    - BF16 등 dtype 미스매치면 캐시된 캐스팅 파라미터로 시도
    - 실패하면 PyTorch 경로 폴백
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()
        self._use_myops = myops_xpu is not None
        self._warned = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, +bound)

    def _warn_once(self, msg: str):
        if not self._warned:
            print(f"[WARN] {msg}")
            self._warned = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_myops and x.device.type == "xpu":
            x_in = x.contiguous()

            # FP32면 가장 안정적인 경로: 파라미터/입력 동일 dtype → fused 직행
            if x_in.dtype == torch.float32 and self.weight.dtype == torch.float32:
                try:
                    return myops_xpu.linear_gelu(x_in, self.weight, self.bias)
                except Exception as e:
                    self._warn_once(f"myops_xpu.linear_gelu fallback due to: {e}")
                    self._use_myops = False
            else:
                # dtype 다를 때: 디바이스 캐스트된 파라미터로 fused 시도
                try:
                    w_cast, b_cast = _cast_params_cached(self, target_dtype=x_in.dtype, device=x_in.device)
                    return myops_xpu.linear_gelu(x_in, w_cast, b_cast)
                except Exception as e:
                    self._warn_once(f"myops_xpu.linear_gelu fallback due to: {e}")
                    self._use_myops = False

        # 폴백: 안전 선형 + GELU
        y = safe_linear(x, _LinearView(self.weight, self.bias))
        return F.gelu(y)

    @property
    def is_active(self) -> bool:
        return self._use_myops and (myops_xpu is not None)

    def disable_myops(self) -> None:
        self._use_myops = False

class ConvSiLU(nn.Module):
    """
    Conv2d + bias + SiLU. When myops_xpu is available and device=XPU, use fused kernel.
    Otherwise fall back to PyTorch path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        use_fused: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype)) if bias else None

        self.reset_parameters()

        self._use_myops = use_fused and has_myops()
        self._warned = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._use_myops
            and myops_xpu is not None
            and x.device.type == "xpu"
            and x.dtype in (torch.float32, torch.bfloat16)
        ):
            try:
                weight = self.weight
                bias = self.bias
                if bias is None:
                    bias = torch.zeros(weight.size(0), device=x.device, dtype=x.dtype)
                return myops_xpu.conv2d_silu(
                    x,
                    weight,
                    bias,
                    list(self.stride),
                    list(self.padding),
                    list(self.dilation),
                    self.groups,
                )
            except Exception as exc:
                if not self._warned:
                    print(f"[WARN] myops_xpu.conv2d_silu fallback due to {exc}")
                    self._warned = True
                self._use_myops = False
        return F.silu(
            F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        )

    @property
    def is_active(self) -> bool:
        return self._use_myops and (myops_xpu is not None)
