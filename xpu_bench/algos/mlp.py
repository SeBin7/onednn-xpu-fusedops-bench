# xpu_bench/algos/mlp.py
from __future__ import annotations
import torch
import torch.nn as nn

# 모델은 models/mlp/mlp.py 에서
from ..models.mlp.mlp import MLP


def build_mlp_model(
    *,
    in_dim: int,
    hidden: int,
    layers: int,
    out_dim: int,
    use_fused: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """
    use_fused=True & device=xpu 인 경우 core.ops.LinearGELU_Fused 사용하도록
    MLP 내부에서 선택적으로 wiring 되어 있어야 한다.
    """
    model = MLP(
        in_dim=in_dim,
        hidden=hidden,
        layers=layers,
        out_dim=out_dim,
        use_fused=use_fused,
        device=device,
        dtype=dtype,
    )
    return model


def mlp_loss():
    # 분류용 더미 로스
    return nn.CrossEntropyLoss()
