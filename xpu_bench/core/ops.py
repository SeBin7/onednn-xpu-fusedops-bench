from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import myops_xpu  # C++/SYCL fused op (Linear+Bias+GELU)
except Exception:
    myops_xpu = None

def has_myops() -> bool:
    return myops_xpu is not None

class LinearGELU_Torch(nn.Module):
    """Reference PyTorch path: y = GELU(x @ W^T + b)"""
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
        return F.gelu(F.linear(x, self.weight, self.bias))


class LinearGELU_Fused(nn.Module):
    """
    Uses myops_xpu.linear_gelu when:
      - myops_xpu is importable AND
      - input lives on XPU device.
    Falls back to PyTorch path otherwise.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()
        self._nan_warning = False
        self._use_myops = myops_xpu is not None

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, +bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_myops and (x.device.type == "xpu"):
            x_in = x.contiguous()
            out = myops_xpu.linear_gelu(x_in, self.weight, self.bias)
            if not torch.isfinite(out).all().item():
                if not self._nan_warning:
                    print("[WARN] myops_xpu.linear_gelu produced non-finite values; falling back to PyTorch path")
                    self._nan_warning = True
                self._use_myops = False
                # fallback if kernel produced NaNs/Infs (e.g., unsupported config)
                return F.gelu(F.linear(x, self.weight, self.bias))
            return out
        # safe fallback
        return F.gelu(F.linear(x, self.weight, self.bias))

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

# =========================
# [ADD] CUDA fused backend
# =========================
import os as _os
try:
    import linear_gelu_cuda as _cuda_ext
except Exception:
    _cuda_ext = None

def _cuda_cc_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return (major > 6) or (major == 6 and minor >= 1)  # Pascal+ (>= 6.1)
    except Exception:
        return False

class LinearGELU_CUDA_Fused(nn.Module):
    """
    GEMM(addmm/cuBLAS) + (bias+gelu) 단일 CUDA 커널.
    - 조건: CUDA 디바이스 + 확장 모듈 import 성공 + CC >= 6.1
    - 실패 시 자동으로 PyTorch 경로로 폴백
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()
        self._ext_ok = _cuda_ext is not None
        self._warned = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, +bound)

    @property
    def is_active(self) -> bool:
        return bool(self._ext_ok and _cuda_cc_ok())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ext_ok and x.is_cuda and _cuda_cc_ok():
            preact = torch.addmm(
                torch.zeros(x.size(0), self.weight.size(0), device=x.device, dtype=x.dtype),
                x, self.weight.t()
            )
            try:
                return _cuda_ext.bias_gelu_forward(preact, self.bias)
            except Exception as exc:
                if not self._warned:
                    print(f"[WARN] CUDA fused bias+gelu fell back due to: {exc}")
                    self._warned = True
        return F.gelu(F.linear(x, self.weight, self.bias))

# =========================
# [ADD] Conv2d + Bias + SiLU (CUDA fused epilogue)
# =========================
import torch.autograd as _ag
try:
    import linear_gelu_cuda as _cuda_ext2
except Exception:
    _cuda_ext2 = None

def _cuda_ok():
    if not torch.cuda.is_available() or _cuda_ext2 is None:
        return False
    try:
        maj, minr = torch.cuda.get_device_capability(0)
        return (maj > 6) or (maj == 6 and minr >= 1)
    except Exception:
        return False

class _BiasSiLUFn(_ag.Function):
    @staticmethod
    def forward(ctx, preact: torch.Tensor, bias: torch.Tensor):
        y = _cuda_ext2.conv_bias_silu_forward(preact, bias)
        ctx.save_for_backward(preact, bias)
        return y
    @staticmethod
    def backward(ctx, grad_out):
        preact, bias = ctx.saved_tensors
        grad_xb = _cuda_ext2.conv_bias_silu_backward(grad_out.contiguous(), preact, bias)
        grad_bias = grad_xb.sum(dim=(0,2,3))
        return grad_xb, grad_bias

class Conv2dSiLU_CUDA_Fused(nn.Module):
    """
    Conv2d(bias=False) -> fused (bias+SiLU) CUDA epilogue.
    Falls back to F.silu if CUDA fused is unavailable.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, device=None, dtype=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation,
                              groups, bias=False, device=device, dtype=dtype)
        self.bias = nn.Parameter(torch.zeros(out_ch, device=device, dtype=dtype)) if bias else None

    @property
    def is_active(self):
        return bool(_cuda_ok() and self.bias is not None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.conv(x)
        if self.is_active and x.is_cuda:
            return _BiasSiLUFn.apply(pre, self.bias)
        # fallback: (pre + b).silu()
        if self.bias is not None:
            pre = pre + self.bias.view(1, -1, 1, 1)
        return torch.nn.functional.silu(pre)
