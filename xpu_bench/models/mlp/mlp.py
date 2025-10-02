from __future__ import annotations
import torch
import torch.nn as nn
import os
from typing import Optional
from math import sqrt

# 핵심: core.ops 를 통해 (fused|torch) 버전 선택
from ...core.ops import LinearGELU_Fused, LinearGELU_Torch, LinearGELUResidual_Fused
import torch.nn.functional as F
from ...core.ops import safe_linear
from xpu_bench.core.ops import _safe_to_fp32, _safe_to_bf16

_TRUTHY = {"1","true","yes","on"}

_RESIDUAL_ENV = os.getenv("RESIDUAL_MLP", "0").strip().lower() in {"1","true","yes","on"}
_RESIDUAL_SCALE = float(os.getenv("RESIDUAL_SCALE", "0.1"))
_USE_FUSED_RESIDUAL = os.getenv("USE_FUSED_RESIDUAL", "1").strip().lower() in _TRUTHY
_RESIDUAL_PROJ    = os.getenv("RESIDUAL_PROJ",    "0").strip().lower() in _TRUTHY

class _LinearGELUBlock(nn.Module):
    """
    Linear + GELU 를 한 블록으로 묶은 모듈.
    use_fused=True 이고 XPU + myops 가 있을 때는 C++ fused (oneDNN) 경로를 사용.
    아니면 PyTorch 순정 경로를 사용.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True,
                 use_fused: bool = False, use_residual: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 단일 Linear+GELU 블록에는 residual을 적용하지 않는다 (FFN 수준에서만)
        self.use_residual = False
        if use_fused:
            self.block = LinearGELU_Fused(in_dim, out_dim, bias=bias, use_residual=False)
        else:
            self.block = LinearGELU_Fused(in_dim, out_dim, bias=bias, use_residual=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# 추가: IPEX 그래프가 잘 잡는 FFN 블록 (Linear→GELU→Linear, bias 포함 권장)
class _FFNBlock(nn.Module):
    """
    입력/출력 차원 동일: x -> Linear(dim->hidden) -> GELU -> Linear(hidden->dim)
    (첫 Linear+GELU는 fused/torch 선택 가능, 두번째 Linear는 일반)
    """
    def __init__(self, dim: int, hidden: int, bias: bool = True,
                 use_fused: bool = False, use_residual: bool = False,
                 residual_scale: float = _RESIDUAL_SCALE):
        super().__init__()
        self.use_residual = use_residual
        self.res_alpha = residual_scale
        # 출력 게이트: 초기 0.0 → 학습되며 열리도록
        self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(gate) ∈ (0,1)
        # 첫 단계: (옵션) Linear+GELU+Residual까지 한 번에 묶기
        self._fused_residual_first = False
        if use_fused and self.use_residual and _USE_FUSED_RESIDUAL:
            if hidden == dim:
                # residual == 입력 x 그대로 사용 가능
                self.fc1 = LinearGELUResidual_Fused(dim, hidden, bias=bias)
                self._fused_residual_first = True
            elif _RESIDUAL_PROJ:
                # 차원 불일치 시, 얇은 projection 으로 hidden에 맞춰 전달
                self.res_proj = nn.Linear(dim, hidden, bias=False)
                self.fc1 = LinearGELUResidual_Fused(dim, hidden, bias=bias)
                self._fused_residual_first = True
        if not self._fused_residual_first:
            # 기본: Linear+GELU만
            self.fc1 = LinearGELU_Fused(dim, hidden, bias=bias, use_residual=False)
        # 두번째 Linear (그래프에서 bias+eltwise(post-op)와 잘 결합됨)
        self.fc2 = nn.Linear(hidden, dim, bias=bias)
        # 더 보수적 초기화(안정성 우선): 필요 시 ENV로 제어
        if os.getenv("RESIDUAL_ZERO_INIT", "0") in {"1","true","yes","on"}:
            nn.init.zeros_(self.fc2.weight)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.kaiming_uniform_(self.fc2.weight, a=5**0.5)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        if self._fused_residual_first:
            # scale·gate 적용 후, 첫 matmul에 binary_add로 합치기
            scale = self.res_alpha * torch.sigmoid(self.gate)
            if hasattr(self, "res_proj"):
                r = safe_linear(res, self.res_proj)
            else:
                r = res
            if r.dtype != x.dtype:
                r = r.to(dtype=x.dtype)
            if scale.item() != 1.0:
                r = r * scale
            x = self.fc1(x, r)        # 여기서 matmul+gelu+binary_add 로 묶임
        else:
            x = self.fc1(x)           # 기존 경로
        x = safe_linear(x, self.fc2)   # UR-safe + bf16-safe 선형
        # (대체 경로) residual을 마지막에 더하는 기존 방식
        if (not self._fused_residual_first) and self.use_residual and x.shape == res.shape:
            x = res + (self.res_alpha * torch.sigmoid(self.gate) * x)
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
        use_residual: bool = False,
        #dtype: torch.dtype,
    ):
        super().__init__()
        assert layers >= 1, "layers must be >= 1"

        prefer_ipex_graph = (os.getenv("PREFER_IPEX_GRAPH", "0") == "1") and (not use_fused)
        if _RESIDUAL_ENV:
            use_residual = True

        # residual을 켜면 경로 통일: 항상 FFN 블록 쌓기(그래프/비그래프 모두 안정)
        if prefer_ipex_graph or use_residual:
            blocks = [ _FFNBlock(in_dim, hidden, bias=True,
                                 use_fused=use_fused and not prefer_ipex_graph,
                                 use_residual=use_residual)
                      for _ in range(layers) ]
            self.backbone = nn.Sequential(*blocks)
            self.head = nn.Linear(in_dim, out_dim, bias=True)
        else:
            # residual 완전 OFF일 때만 LinearGELU 스택 유지(베이스라인)
            blocks = [ _LinearGELUBlock(in_dim, hidden, use_fused=use_fused, use_residual=False) ]
            for _ in range(layers - 1):
                blocks.append(_LinearGELUBlock(hidden, hidden, use_fused=use_fused, use_residual=False))
            self.backbone = nn.Sequential(*blocks)
            self.head = nn.Linear(hidden, out_dim, bias=True)

        self.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) 이미 넣어둔 "블록-브릿지"가 있다면 그대로 유지
        bridge_on = os.getenv("BF16_FUSED_BRIDGE", "0").strip().lower() in _TRUTHY
        if bridge_on and x.device.type == "xpu" and x.dtype == torch.bfloat16:
            x32 = _safe_to_fp32(x)      # ① 한 번만 업캐스트
            x32 = self.backbone(x32)
            x32 = self.head(x32)
            return _safe_to_bf16(x32)   # ② 한 번만 다운캐스트

        # (2) 기본 경로: backbone 실행
        x = self.backbone(x)

        # (3) 헤드-브릿지: bf16 입력 + FP32 헤드 가중치면 안전 업캐스트→헤드→다운캐스트
        if (
            x.device.type == "xpu"
            and x.dtype == torch.bfloat16
            and hasattr(self.head, "weight")
            and getattr(self.head.weight, "dtype", None) == torch.float32
        ):
            x32 = _safe_to_fp32(x)
            y32 = self.head(x32)
            return _safe_to_bf16(y32)

        # (4) dtype이 이미 맞으면 그대로
        return self.head(x)