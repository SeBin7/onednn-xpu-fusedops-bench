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
