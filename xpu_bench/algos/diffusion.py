# xpu_bench/algos/diffusion.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

# UNet은 models/diffusion/unet2d.py 에서
from ..models.diffusion.unet2d import UNetTiny


def build_diffusion_model(
    *,
    in_ch: int,
    base_ch: int,
    ch_mult: Tuple[int, ...],
    use_fused: bool,
    device: torch.device,
    dtype: torch.dtype,
    memory_format: torch.memory_format | None = None,
) -> nn.Module:
    model = UNetTiny(
        in_ch=in_ch,
        base_ch=base_ch,
        ch_mult=ch_mult,
        use_fused=use_fused,
    ).to(device=device, dtype=dtype)
    if memory_format is not None:
        model = model.to(memory_format=memory_format)
    return model


def make_beta_schedule(T: int, device: torch.device, dtype: torch.dtype):
    """
    간단한 선형 베타 스케줄
    """
    # 일부 XPU 드라이버는 linspace를 직접 BF16 등으로 생성할 때 UR 오류가 발생하므로
    # 항상 CPU float32에서 생성 후 원하는 dtype/device로 옮긴다.
    betas = torch.linspace(1e-4, 0.02, T, device="cpu", dtype=torch.float32)
    betas = betas.to(device=device, dtype=dtype)
    alphas = (1.0 - betas)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def diffusion_loss(betas: torch.Tensor, alphas_cumprod: torch.Tensor):
    """
    표준 DDPM 예측-노이즈 MSE 로스
    """
    T = betas.shape[0]

    def _loss_fn(pred_eps: torch.Tensor, target_eps: torch.Tensor | None) -> torch.Tensor:
        # 여기서는 pred_eps와 target_eps가 바로 MSE 대상
        # (학습 루프에서 pred = model(x_t, t), target = eps로 전달)
        if target_eps is None:
            raise RuntimeError("diffusion_loss requires target noise tensor")
        return torch.mean((pred_eps - target_eps) ** 2)

    return _loss_fn
