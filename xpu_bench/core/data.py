from __future__ import annotations
from typing import Iterator, Tuple
import torch

def steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, dataset_size // batch_size)

# ----- MLP -----
def mlp_batch_generator(
    *,
    dataset_size: int,
    batch_size: int,
    in_dim: int,
    out_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    reuse_batch: bool = False,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    steps = steps_per_epoch(dataset_size, batch_size)

    if reuse_batch:
        x = torch.randn(batch_size, in_dim, device=device, dtype=dtype)
        y = torch.randint(0, out_dim, (batch_size,), device=device)
        while True:
            yield x, y

    while True:
        for _ in range(steps):
            x = torch.randn(batch_size, in_dim, device=device, dtype=dtype)
            y = torch.randint(0, out_dim, (batch_size,), device=device)
            yield x, y

# ----- Diffusion -----
def diffusion_batch_generator(
    *,
    dataset_size: int,
    batch_size: int,
    img_size: int,
    in_ch: int,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    reuse_batch: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Iterator[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    steps = steps_per_epoch(dataset_size, batch_size)

    def _format_image(x: torch.Tensor) -> torch.Tensor:
        if memory_format is not None:
            x = x.to(memory_format=memory_format)
        return x

    def _make_clean() -> torch.Tensor:
        base = torch.randn(batch_size, in_ch, img_size, img_size, device="cpu", dtype=torch.float32)
        base = base.to(dtype=dtype)
        base = _format_image(base)
        return base.to(device=device, non_blocking=True)

    def _make_noise_like(ref: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(ref.shape, device="cpu", dtype=torch.float32)
        noise = noise.to(dtype=dtype)
        if memory_format is not None:
            noise = noise.to(memory_format=memory_format)
        return noise.to(device=device, non_blocking=True)

    if reuse_batch:
        x0 = _make_clean()
        while True:
            t = torch.randint(0, alphas_cumprod.shape[0], (batch_size,), device=device, dtype=torch.long)
            noise = _make_noise_like(x0)
            ac_t = alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1).to(dtype)
            sqrt_ac = torch.sqrt(ac_t)
            sqrt_om = torch.sqrt(torch.clamp(1.0 - ac_t, min=0.0))
            x_t = sqrt_ac * x0 + sqrt_om * noise
            yield (x_t, t), noise

    while True:
        for _ in range(steps):
            x0 = _make_clean()
            t = torch.randint(0, alphas_cumprod.shape[0], (batch_size,), device=device, dtype=torch.long)
            noise = _make_noise_like(x0)
            ac_t = alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1).to(dtype)
            sqrt_ac = torch.sqrt(ac_t)
            sqrt_om = torch.sqrt(torch.clamp(1.0 - ac_t, min=0.0))
            x_t = sqrt_ac * x0 + sqrt_om * noise
            yield (x_t, t), noise
