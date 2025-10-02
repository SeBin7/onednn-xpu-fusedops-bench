# xpu_bench/core/ops.py
from __future__ import annotations
import os, math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import myops_xpu  # fused kernels (Linear+GELU, Conv+SiLU)
    _HAS_MYOPS = True
except Exception:
    myops_xpu = None
    _HAS_MYOPS = False


def has_myops() -> bool:
    """Return True if custom fused kernels are importable."""
    return _HAS_MYOPS

def _same_device_dtype(*tensors: torch.Tensor) -> None:
    dev = tensors[0].device
    dt  = tensors[0].dtype
    for t in tensors[1:]:
        if t.device != dev:
            raise RuntimeError("device mismatch")
        if t.dtype  != dt:
            raise RuntimeError("dtype mismatch")


# ============ truthy/env gates ============
_TRUTHY = {"1", "true", "yes", "on"}
def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1","true","yes","on"}


# ============ BF16-Bridge toggle ============
# If enabled: keep params FP32; when input is BF16 on XPU, upcast input -> fused/torch op -> downcast output
_ENABLE_BF16_BRIDGE = os.getenv("BF16_FUSED_BRIDGE", "0").strip().lower() in _TRUTHY


def _get_param_version(p: torch.Tensor) -> int:
    return int(getattr(p, "_version", 0))


def _cast_params_cached(mod: nn.Module, *, target_dtype: torch.dtype, device: torch.device):
    """
    Cast mod.weight/bias to target_dtype on device and cache by (param_version, dtype, device).
    Falls back through CPU relay on XPU UR error.
    """
    w, b = mod.weight, mod.bias
    ver = _get_param_version(w)
    key = (ver, target_dtype, device.type, device.index)
    cache = getattr(mod, "_cast_cache", None)
    if isinstance(cache, dict) and cache.get("key") == key:
        return cache["w"], cache["b"]

    try:
        w_cast = w.to(device=device, dtype=target_dtype, non_blocking=True)
        b_cast = b.to(device=device, dtype=target_dtype, non_blocking=True) if b is not None else None
    except RuntimeError as e:
        if "UR error" not in str(e):
            raise
        # CPU relay
        w_cast = w.to("cpu", dtype=target_dtype).to(device, non_blocking=True)
        b_cast = b.to("cpu", dtype=target_dtype).to(device, non_blocking=True) if b is not None else None

    setattr(mod, "_cast_cache", {"key": key, "w": w_cast, "b": b_cast})
    return w_cast, b_cast


# -------- safe casts (avoid XPU UR error) --------
def _safe_cast(x: torch.Tensor, target_dtype: torch.dtype, *, require_contiguous: bool = True) -> torch.Tensor:
    if x.device.type != "xpu":
        return x.to(dtype=target_dtype)
    try:
        x_in = x.contiguous() if require_contiguous else x
        return x_in.to(dtype=target_dtype, non_blocking=True)
    except RuntimeError as e:
        if "UR error" not in str(e):
            raise
    # CPU relay
    x_cpu = x.to("cpu", dtype=target_dtype)
    return x_cpu.to(x.device, non_blocking=True)


def _safe_to_bf16(x: torch.Tensor) -> torch.Tensor:
    return _safe_cast(x, torch.bfloat16, require_contiguous=True)


def _safe_to_fp32(x: torch.Tensor) -> torch.Tensor:
    return _safe_cast(x, torch.float32, require_contiguous=True)


# -------- linear helpers --------
class _LinearView(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.weight = weight
        self.bias = bias


def safe_linear(x: torch.Tensor, linear: nn.Module) -> torch.Tensor:
    """
    Linear with BF16-Bridge + cached param casting on XPU.
    `linear` must expose .weight/.bias.
    """
    w, b = linear.weight, linear.bias

    # BF16-Bridge: inputs BF16@XPU, params FP32 -> upcast input -> linear -> downcast output
    if (
        _ENABLE_BF16_BRIDGE
        and x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and w.dtype == torch.float32
    ):
        x32 = _safe_to_fp32(x)
        y32 = F.linear(x32, w, b)
        return _safe_to_bf16(y32)

    # matching dtype fast path
    if x.dtype == w.dtype:
        return F.linear(x, w, b)

    # XPU: cast params to input dtype (with CPU relay fallback)
    if x.device.type == "xpu":
        w_cast, b_cast = _cast_params_cached(linear, target_dtype=x.dtype, device=x.device)
        return F.linear(x, w_cast, b_cast)

    # CPU/CUDA generic
    w_cast = w.to(dtype=x.dtype)
    b_cast = b.to(dtype=x.dtype) if b is not None else None
    return F.linear(x, w_cast, b_cast)


# -------- reference Linear+GELU (PyTorch) --------
class LinearGELU_Torch(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, use_residual: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, +bound)

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = safe_linear(x, _LinearView(self.weight, self.bias))
        y = F.gelu(y)
        if self.use_residual and residual is not None:
            # dtype 정렬 후 더하기 (UR error 회피는 safe_linear 내부가 담당)
            if residual.dtype != y.dtype:
                residual = residual.to(dtype=y.dtype)
            y = y + residual
        return y


# -------- fused Linear+GELU (myops) --------
class LinearGELU_Fused(nn.Module):
    """
    Fused Linear+Bias+GELU (myops_xpu) with BF16-Bridge support on XPU.
    - Fast path: FP32 end-to-end (x, W, b in FP32)
    - BF16-Bridge: if x is BF16 and W is FP32 on XPU, upcast x -> fused -> downcast output
    - Fallback: safe_linear + GELU
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, use_residual: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()
        self._use_myops = has_myops() and _env_on("USE_FUSED_LINEAR_GELU", "1")
        self._warned = False
        self.use_residual = use_residual

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

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._use_myops and x.device.type == "xpu":
            # BF16-Bridge for fused path
            if (
                _ENABLE_BF16_BRIDGE
                and x.dtype == torch.bfloat16
                and self.weight.dtype == torch.float32
            ):
                try:
                    x32 = _safe_to_fp32(x)
                    w32 = self.weight.contiguous()
                    b32 = self.bias.contiguous() if self.bias is not None else None
                    y32 = myops_xpu.linear_gelu(x32, w32, b32)
                    y = _safe_to_bf16(y32)
                    if self.use_residual and residual is not None:
                        if residual.dtype != y.dtype:
                            residual = residual.to(dtype=y.dtype)
                        y = y + residual
                    return y
                except Exception as e:
                    self._warn_once(f"linear_gelu bridge fallback due to: {e}")
                    self._use_myops = False

            # Native fused fast path (matching dtype)
            if x.dtype == torch.float32 and self.weight.dtype == torch.float32:
                try:
                    y = myops_xpu.linear_gelu(x, self.weight, self.bias)
                    if self.use_residual and residual is not None:
                        if residual.dtype != y.dtype:
                            residual = residual.to(dtype=y.dtype)
                        y = y + residual
                    return y
                except Exception as e:
                    self._warn_once(f"linear_gelu fused fallback due to: {e}")
                    self._use_myops = False

            # If input is BF16, avoid trying fused with casted params on XPU
            # (this path is the most common source of UR errors). Fall back gracefully.
            if x.dtype == torch.bfloat16:
                self._use_myops = False
            else:
                try:
                    w_cast, b_cast = _cast_params_cached(self, target_dtype=x.dtype, device=x.device)
                    y = myops_xpu.linear_gelu(x, w_cast, b_cast)
                    if self.use_residual and residual is not None:
                        if residual.dtype != y.dtype:
                            residual = residual.to(dtype=y.dtype)
                        y = y + residual
                    return y
                except Exception as e:
                    self._warn_once(f"linear_gelu casted-params fallback due to: {e}")
                    self._use_myops = False

        # Fallback: safe linear + GELU
        y = safe_linear(x, _LinearView(self.weight, self.bias))
        y = F.gelu(y)
        if self.use_residual and residual is not None:
            if residual.dtype != y.dtype:
                residual = residual.to(dtype=y.dtype)
            y = y + residual
        return y

    @property
    def is_active(self) -> bool:
        return self._use_myops and (myops_xpu is not None)

    def disable_myops(self) -> None:
        self._use_myops = False

class LinearGELUResidual_Fused(torch.nn.Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias   = nn.Parameter(torch.empty(out_f)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # 새 게이트 (필요 시 끌 수 있게)
        self._use_myops = (myops_xpu is not None) and _env_on("USE_FUSED_LINEAR_GELU_RESIDUAL", "1")
        self._warned = False

    def _warn_once(self, msg: str):
        if not self._warned:
            print(f"[WARN] linear_gelu_residual: {msg}")
            self._warned = True

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 레지듀얼 dtype/contig 정리
        x = x.contiguous()
        r = residual.contiguous()
        if r.dtype != x.dtype:
            r = r.to(dtype=x.dtype)

        # myops 경로
        if self._use_myops and (myops_xpu is not None) and (x.device.type == "xpu"):
            w = self.weight.contiguous()
            b = (self.bias.contiguous() if self.bias is not None
                 else torch.zeros(w.size(0), device=x.device, dtype=w.dtype))

            try:
                # BF16-Bridge (입력 BF16, 가중치 FP32 흔함)
                if x.dtype == torch.bfloat16 and w.dtype == torch.float32:
                    from .ops import _safe_to_fp32, _safe_to_bf16
                    y32 = myops_xpu.linear_gelu_residual(_safe_to_fp32(x), w, b, _safe_to_fp32(r))
                    return _safe_to_bf16(y32)

                # dtype 맞추기
                if w.dtype != x.dtype:
                    w = w.to(dtype=x.dtype)
                    b = b.to(dtype=x.dtype)

                return myops_xpu.linear_gelu_residual(x, w, b, r)

            except Exception as e:
                self._warn_once(f"primitive failed -> fallback ({e})")
                self._use_myops = False  # 그 이후엔 Torch 경로로

        # Torch 폴백 (안전)
        y = torch.nn.functional.gelu(torch.nn.functional.linear(x, self.weight, self.bias))
        return y + r


# -------- fused Conv+SiLU (myops) --------
class ConvSiLU(nn.Module):
    """
    Conv2d + bias + SiLU with BF16-Bridge on XPU.
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
        self._use_myops = use_fused and has_myops() and _env_on("USE_FUSED_CONV_SILU", "1")
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
        ):
            # BF16-Bridge for conv: inputs BF16@XPU, params FP32
            if _ENABLE_BF16_BRIDGE and x.dtype == torch.bfloat16 and self.weight.dtype == torch.float32:
                try:
                    x32 = _safe_to_fp32(x)
                    w32 = self.weight.contiguous()
                    b32 = (
                        self.bias.contiguous()
                        if self.bias is not None
                        else torch.zeros(self.weight.size(0), device=x.device, dtype=torch.float32)
                    )
                    y32 = myops_xpu.conv2d_silu(
                        x32, w32, b32,
                        list(self.stride),
                        list(self.padding),
                        list(self.dilation),
                        self.groups,
                    )
                    return _safe_to_bf16(y32)
                except Exception as exc:
                    if not self._warned:
                        print(f"[WARN] conv2d_silu bridge fallback due to {exc}")
                        self._warned = True
                    self._use_myops = False

            # Native fused (matching dtype)
            if x.dtype in (torch.float32, torch.bfloat16) and self.weight.dtype == x.dtype:
                try:
                    b = (
                        self.bias
                        if self.bias is not None
                        else torch.zeros(self.weight.size(0), device=x.device, dtype=x.dtype)
                    )
                    return myops_xpu.conv2d_silu(
                        x,
                        self.weight,
                        b,
                        list(self.stride),
                        list(self.padding),
                        list(self.dilation),
                        self.groups,
                    )
                except Exception as exc:
                    if not self._warned:
                        print(f"[WARN] conv2d_silu fused fallback due to {exc}")
                        self._warned = True
                    self._use_myops = False

        # Fallback: torch.conv2d + SiLU
        return F.silu(
            F.conv2d(
                x, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups,
            )
        )

    @property
    def is_active(self) -> bool:
        return self._use_myops and (myops_xpu is not None)