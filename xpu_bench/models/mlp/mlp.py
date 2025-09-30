from __future__ import annotations
import torch
import torch.nn as nn
import os

# 핵심: core.ops 를 통해 (fused|torch) 버전 선택
from ...core.ops import LinearGELU_Fused, LinearGELU_Torch
import torch.nn.functional as F
from ...core.ops import safe_linear

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

# 추가: IPEX 그래프가 잘 잡는 FFN 블록 (Linear→GELU→Linear, bias 포함 권장)
class _FFNBlock(nn.Module):
    """
    입력/출력 차원 동일: x -> Linear(dim->hidden) -> GELU -> Linear(hidden->dim)
    (첫 Linear+GELU는 fused/torch 선택 가능, 두번째 Linear는 일반)
    """
    def __init__(self, dim: int, hidden: int, bias: bool = True, use_fused: bool = False):
        super().__init__()
        # 첫 단계는 Linear+GELU (fused/tiny torch)
        if use_fused:
            self.fc1 = LinearGELU_Fused(dim, hidden, bias=bias)
        else:
            self.fc1 = LinearGELU_Torch(dim, hidden, bias=bias)
        # 두번째 Linear (그래프에서 bias+eltwise(post-op)와 잘 결합됨)
        self.fc2 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)                # 그대로
        x = safe_linear(x, self.fc2)   # ✅ UR-safe + bf16-safe 선형
        return x


class MLP(nn.Module):
    """
    [기본] [in_dim -> hidden]*layers + head(out_dim)
      - 중간 레이어는 Linear+GELU 반복, 마지막은 Linear만 (logits).
    [IPEX 우대 모드] FFN(dim->hidden->dim) * layers + head(out_dim)
      - 입력/출력 차원 동일 블록을 쌓아 IPEX(oneDNN Graph) 합성이 잘 걸리게.
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
        #dtype: torch.dtype,
    ):
        super().__init__()
        assert layers >= 1, "layers must be >= 1"

        # 토글: IPEX 그래프 우대 모드
        prefer_ipex_graph = (os.getenv("PREFER_IPEX_GRAPH", "0") == "1") and (not use_fused)

        if prefer_ipex_graph:
            # FFN(dim->hidden->dim) 블록을 layers 개 쌓고, head는 in_dim->out_dim
            blocks = [ _FFNBlock(in_dim, hidden, bias=True, use_fused=False) for _ in range(layers) ]
            self.backbone = nn.Sequential(*blocks)
            self.head = nn.Linear(in_dim, out_dim, bias=True)
        else:
            # 기존: 첫 블록 in->hidden, 이후 (hidden->hidden) * (layers-1), head는 hidden->out_dim
            blocks = []
            blocks.append(_LinearGELUBlock(in_dim, hidden, use_fused=use_fused))
            for _ in range(layers - 1):
                blocks.append(_LinearGELUBlock(hidden, hidden, use_fused=use_fused))
            self.backbone = nn.Sequential(*blocks)
            self.head = nn.Linear(hidden, out_dim, bias=True)

        self.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = safe_linear(x, self.head)   # ✅
        return x