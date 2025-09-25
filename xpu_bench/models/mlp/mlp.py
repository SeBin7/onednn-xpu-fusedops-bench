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
