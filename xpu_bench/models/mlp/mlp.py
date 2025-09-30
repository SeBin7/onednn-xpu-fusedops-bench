from __future__ import annotations
import torch
import torch.nn as nn

# 핵심: core.ops 를 통해 (fused|torch) 버전 선택
from ...core.ops import LinearGELU_Fused, LinearGELU_Torch


class _LinearGELUBlock(nn.Module):
    """
    Linear + GELU 를 한 블록으로 묶은 모듈.
    use_fused=True 이고 XPU + myops 가 있을 때는 C++ fused (oneDNN) 경로를 사용.
    아니면 PyTorch 순정 경로를 사용.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, use_fused: bool = False):
        super().__init__()
        if use_fused:
            self.block = LinearGELU_Fused(in_dim, out_dim, bias=bias)
        else:
            self.block = LinearGELU_Torch(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLP(nn.Module):
    """
    [in_dim -> hidden]*layers + head(out_dim)
    중간 레이어는 Linear+GELU 반복, 마지막은 Linear만 (logits).
    """
    def __init__(
        self,
        *,
        in_dim: int,
        hidden: int,
        layers: int,
        out_dim: int,
        use_fused: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        assert layers >= 1, "layers must be >= 1"

        blocks = []
        # 첫 블록: in -> hidden
        blocks.append(_LinearGELUBlock(in_dim, hidden, use_fused=use_fused))
        # 중간 블록: (hidden -> hidden) * (layers-1)
        for _ in range(layers - 1):
            blocks.append(_LinearGELUBlock(hidden, hidden, use_fused=use_fused))

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, out_dim, bias=True)

        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

# =========================
# [ADD] Auto fused blocks
# =========================
import os as _os
from ...core.ops import LinearGELU_Fused, LinearGELU_Torch, LinearGELU_CUDA_Fused  # 기존 + CUDA fused

class AutoLinearGELUBlock(nn.Module):
    """
    Try XPU fused first (myops_xpu) then CUDA fused; else fall back to torch.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, use_fused: bool = False):
        super().__init__()
        # [ADD] 강제 스위치: XPU_FORCE_FUSED=1 이면 use_fused 무시하고 켠다
        force = _os.environ.get("XPU_FORCE_FUSED", "0") == "1"
        self.use_fused = bool(use_fused or force)
        self.block_xpu  = LinearGELU_Fused(in_dim, out_dim, bias=bias) if self.use_fused else None
        self.block_cuda = LinearGELU_CUDA_Fused(in_dim, out_dim, bias=bias) if self.use_fused else None
        self.block_ref  = LinearGELU_Torch(in_dim, out_dim, bias=bias)

    @property
    def is_fused_active(self) -> bool:
        bx = getattr(self.block_xpu, "is_active", False) if self.block_xpu else False
        bc = getattr(self.block_cuda, "is_active", False) if self.block_cuda else False
        return bool(bx or bc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused:
            if (self.block_xpu is not None) and getattr(self.block_xpu, "is_active", False) and x.device.type == "xpu":
                return self.block_xpu(x)
            if (self.block_cuda is not None) and getattr(self.block_cuda, "is_active", False) and x.is_cuda:
                return self.block_cuda(x)
        return self.block_ref(x)


class MLP_AutoFused(nn.Module):
    """
    MLP variant that uses AutoLinearGELUBlock (XPU fused -> CUDA fused -> torch).
    원래 MLP는 건드리지 않고 새 클래스로만 추가합니다.
    """
    def __init__(
        self,
        *,
        in_dim: int,
        hidden: int,
        layers: int,
        out_dim: int,
        use_fused: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        assert layers >= 1, "layers must be >= 1"

        blocks = []
        blocks.append(AutoLinearGELUBlock(in_dim, hidden, use_fused=use_fused))
        for _ in range(layers - 1):
            blocks.append(AutoLinearGELUBlock(hidden, hidden, use_fused=use_fused))

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, out_dim, bias=True)
        self.to(device=device, dtype=dtype)

        if _os.environ.get("XPU_BENCH_VERBOSE", "0") == "1":
            active = getattr(self.backbone[0], "is_fused_active", False)
            print(f"[INFO] Fused Linear+GELU (auto): {'ENABLED' if active else 'DISABLED'}")

    # ★ forward 빠져서 에러났던 부분 — 추가
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

# (옵션) 기존 CLI를 건드리지 않고 심볼만 바꿔치기하고 싶을 때:
if _os.environ.get("XPU_BENCH_AUTO_BACKEND", "0") == "1":
    MLP = MLP_AutoFused  # 원본 MLP는 삭제하지 않고 alias만 제공
