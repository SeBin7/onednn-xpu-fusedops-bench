# xpu_bench/models/diffusion/unet2d.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.ops import LinearGELU_Fused, LinearGELU_Torch, ConvSiLU, has_myops


def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    *,
    target_dtype: torch.dtype | None = None,
    target_device: torch.device | None = None,
) -> torch.Tensor:
    """
    DDPM 스타일 사인-코사인 임베딩 (싱글 주파수 세트).
    t: [B] (int or float)
    return: [B, dim]
    """
    half = dim // 2
    t_cpu = t.detach().to("cpu", dtype=torch.float32)
    freqs = torch.exp(
        torch.arange(half, device="cpu", dtype=torch.float32)
        * -(math.log(10000.0) / max(1, half - 1))
    )
    args = t_cpu[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        # dim 이 홀수면 1개 채워줌
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    if target_dtype is None:
        target_dtype = torch.float32
    if target_device is None:
        target_device = t.device
    emb = emb.to(dtype=target_dtype)
    return emb.to(device=target_device, non_blocking=True)


class TimeMLP(nn.Module):
    """
    시간(timestep) 임베딩을 채널 임베딩으로 사상.
    use_fused=True 이고 XPU + myops 사용 가능할 때 Linear+GELU fused 경로 사용.
    """
    def __init__(self, time_dim: int, out_dim: int, *, use_fused: bool = False):
        super().__init__()
        if use_fused:
            self.fused = LinearGELU_Fused(time_dim, out_dim, bias=True)
            self.net = nn.Linear(out_dim, out_dim)
        else:
            self.fused = None
            self.net = nn.Sequential(
                nn.Linear(time_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:
            h = self.fused(t_emb)
            return self.net(h)
        return self.net(t_emb)  # [B, out_dim]


class ResBlock(nn.Module):
    """
    Conv-GN-SiLU + time-proj + Conv-GN-SiLU 의 간단한 ResBlock.
    time_dim 으로 들어오는 임베딩을 Linear -> [B, out_ch] 로 만들고
    [B, C, H, W] 텐서에 브로드캐스트하여 더함.
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, *, use_fused: bool = False, down: bool = False, up: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.down = down
        self.up = up

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        self.time = TimeMLP(time_dim, out_ch, use_fused=use_fused)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        # 다운/업 샘플은 별도 처리 (stride conv 안 씀)
        self.downsample = nn.Identity()
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.upsample = nn.Identity()
        if up:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # time projection
        # t_emb: [B, time_dim] -> [B, out_ch, 1, 1]
        t_proj = self.time(t_emb)[:, :, None, None]

        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h + t_proj)  # time conditioning

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h)

        out = h + self.skip(x)

        if self.down:
            out = self.downsample(out)
        if self.up:
            out = self.upsample(out)
        return out


class UNetTiny(nn.Module):
    """
    매우 간단한 2D UNet:
      enc1 -> enc2(down) -> enc3(down) -> bottleneck -> dec3(up with skip) -> dec2(up with skip) -> out
    ch_mult 예: (1, 2, 4)
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 64, ch_mult: tuple[int, ...] = (1, 2, 4), *, use_fused: bool = False):
        super().__init__()
        assert len(ch_mult) >= 3, "ch_mult must have at least 3 values, e.g., 1,2,4"

        self.in_ch = in_ch
        self.base_ch = base_ch
        self.ch_mult = tuple(ch_mult)
        time_dim = base_ch  # timestep embedding dim

        c1 = base_ch * ch_mult[0]
        c2 = base_ch * ch_mult[1]
        c3 = base_ch * ch_mult[2]

        # 입력 stem
        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)

        # Encoder
        self.enc1 = ResBlock(c1, c1, time_dim=time_dim, use_fused=use_fused, down=False)
        self.enc2 = ResBlock(c1, c2, time_dim=time_dim, use_fused=use_fused, down=True)
        self.enc3 = ResBlock(c2, c3, time_dim=time_dim, use_fused=use_fused, down=True)

        # Bottleneck
        self.mid = ResBlock(c3, c3, time_dim=time_dim, use_fused=use_fused, down=False)

        # Decoder (skip concat: 채널 합)
        self.dec3 = ResBlock(c3 + c2, c2, time_dim=time_dim, use_fused=use_fused, up=True)
        self.dec2 = ResBlock(c2 + c1, c1, time_dim=time_dim, use_fused=use_fused, up=True)

        # 출력 head
        if use_fused and has_myops():
            self.head = nn.Sequential(
                ConvSiLU(c1, c1, 3, padding=1, use_fused=True),
                nn.Conv2d(c1, in_ch, 3, padding=1),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(c1, in_ch, 3, padding=1),
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: [B], sinusoidal -> mlp 없이 바로 사용 (간단화)
        t_emb = timestep_embedding(
            t,
            self.base_ch,
            target_dtype=x.dtype,
            target_device=x.device,
        )  # [B, base_ch]

        # enc
        x0 = self.stem(x)                  # [B, c1, H, W]
        e1 = self.enc1(x0, t_emb)          # [B, c1, H, W]
        e2 = self.enc2(e1, t_emb)          # [B, c2, H/2, W/2]
        e3 = self.enc3(e2, t_emb)          # [B, c3, H/4, W/4]

        # mid
        m = self.mid(e3, t_emb)            # [B, c3, H/4, W/4]

        # dec (skip feature spatial sizes를 맞춰서 concat)
        e2_skip = e2 if e2.shape[-1] == m.shape[-1] else F.avg_pool2d(e2, kernel_size=2, stride=2)
        d3_in = torch.cat([m, e2_skip], dim=1)  # [B, c3+c2, H/4, W/4]
        d3 = self.dec3(d3_in, t_emb)            # up -> [B, c2, H/2, W/2]

        e1_skip = e1 if e1.shape[-1] == d3.shape[-1] else F.avg_pool2d(e1, kernel_size=2, stride=2)
        d2_in = torch.cat([d3, e1_skip], dim=1) # [B, c2+c1, H/2, W/2]
        d2 = self.dec2(d2_in, t_emb)            # up -> [B, c1, H, W]

        out = self.head(d2)                # [B, in_ch, H, W]  (예측 잡음)
        return out
