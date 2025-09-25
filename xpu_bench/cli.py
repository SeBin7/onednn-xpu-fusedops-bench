# xpu_bench/cli.py
from __future__ import annotations
import argparse
import os
import torch

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except Exception:
    ipex = None

# ★ core 하위로 경로 정리
from .core.device import pick_device, get_dtype
from .core.train import TrainConfig, train_loop
from .core.data import (
    mlp_batch_generator,
    diffusion_batch_generator,
    steps_per_epoch,
)
from .core.ops import has_myops, LinearGELU_Fused, ConvSiLU


def _env_flag(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}

# 알고리즘별 빌더/로스
from .algos.mlp import build_mlp_model, mlp_loss
from .algos.diffusion import build_diffusion_model, diffusion_loss, make_beta_schedule


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cpu", choices=["cpu", "xpu", "cuda"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dataset-size", type=int, default=12800)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--reuse-batch", action="store_true")


def _count_active_fused_layers(model: torch.nn.Module) -> tuple[int, int]:
    active = 0
    total = 0
    for module in model.modules():
        if hasattr(module, "is_active"):
            total += 1
            if module.is_active:
                active += 1
    return active, total


def _preflight_mlp(
    model: torch.nn.Module,
    *,
    in_dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.randn(batch_size, in_dim, device=device, dtype=dtype)
        model(x)
    model.train(training)


def _preflight_diffusion(
    model: torch.nn.Module,
    *,
    in_ch: int,
    img_size: int,
    timesteps: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.randn(batch_size, in_ch, img_size, img_size, device=device, dtype=dtype)
        t = torch.randint(0, timesteps, (batch_size,), device=device, dtype=torch.long)
        model(x, t)
    model.train(training)


def main() -> None:
    ap = argparse.ArgumentParser("xpu_bench")
    sub = ap.add_subparsers(dest="task", required=True)

    # MLP
    p_mlp = sub.add_parser("mlp", help="Linear-heavy MLP benchmark")
    _common_args(p_mlp)
    p_mlp.add_argument("--in-dim", type=int, default=2048)
    p_mlp.add_argument("--hidden", type=int, default=4096)
    p_mlp.add_argument("--layers", type=int, default=20)
    p_mlp.add_argument("--out-dim", type=int, default=1000)
    p_mlp.add_argument("--use-fused", action="store_true",
                       help="Use C++ fused Linear+GELU if available & device=xpu")

    # Diffusion (tiny UNet)
    p_diff = sub.add_parser("diffusion", help="Tiny UNet diffusion benchmark")
    _common_args(p_diff)
    p_diff.add_argument("--img-size", type=int, default=64)
    p_diff.add_argument("--in-ch", type=int, default=3)
    p_diff.add_argument("--base-ch", type=int, default=64)
    p_diff.add_argument("--ch-mult", type=str, default="1,2,4")
    p_diff.add_argument("--timesteps", type=int, default=1000)
    p_diff.add_argument("--use-fused", action="store_true",
                        help="Use C++ fused Linear+GELU for time embeddings when available & device=xpu")

    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = get_dtype(args.precision)

    if args.task == "mlp":
        env_use_fused = _env_flag("USE_MYOPS")
        fused_requested = args.use_fused or env_use_fused
        fused_active = fused_requested and device.type == "xpu" and has_myops()

        model = build_mlp_model(
            in_dim=args.in_dim,
            hidden=args.hidden,
            layers=args.layers,
            out_dim=args.out_dim,
            use_fused=fused_active,
            device=device,
            dtype=dtype,
        )
        if fused_active:
            _preflight_mlp(
                model,
                in_dim=args.in_dim,
                batch_size=args.batch_size,
                device=device,
                dtype=dtype,
            )
            active_layers, total_layers = _count_active_fused_layers(model)
            fused_active = fused_active and active_layers > 0 and active_layers == total_layers
            if not fused_active:
                print("[WARN] MLP fused path disabled after preflight (falling back to PyTorch)")
        loss_fn = mlp_loss()
        data_iter = mlp_batch_generator(
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            device=device,
            dtype=dtype,
            reuse_batch=args.reuse_batch,
        )
        steps = steps_per_epoch(args.dataset_size, args.batch_size)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        cfg = TrainConfig(
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            steps_per_epoch=steps,
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            log_interval=args.log_interval,
        )
        print(f"[INFO] Fused Linear+GELU: "
              f"{'ENABLED' if fused_active else 'DISABLED'}")
        train_loop(cfg, model, data_iter, loss_fn, optimizer=opt)
        return

    if args.task == "diffusion":
        ch_mult = tuple(int(x) for x in args.ch_mult.split(",") if x)
        env_use_fused = _env_flag("USE_MYOPS")
        fused_requested = args.use_fused or env_use_fused
        fused_active = fused_requested and device.type == "xpu" and has_myops()
        if args.use_fused and not fused_active:
            print("[WARN] --use-fused requested but myops_xpu not available or device != xpu; falling back")
        use_channels_last = (device.type == "xpu")
        memory_format = torch.channels_last if use_channels_last else None

        model = build_diffusion_model(
            in_ch=args.in_ch,
            base_ch=args.base_ch,
            ch_mult=ch_mult,
            use_fused=fused_active,
            device=device,
            dtype=dtype,
            memory_format=memory_format,
        )
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        if fused_active:
            _preflight_diffusion(
                model,
                in_ch=args.in_ch,
                img_size=args.img_size,
                timesteps=args.timesteps,
                batch_size=args.batch_size,
                device=device,
                dtype=dtype,
            )
            active_layers, total_layers = _count_active_fused_layers(model)
            fused_active = fused_active and active_layers > 0 and active_layers == total_layers
            if not fused_active:
                print("[WARN] Diffusion fused path disabled after preflight (falling back to PyTorch)")
        betas, alphas_cumprod = make_beta_schedule(args.timesteps, device, dtype)
        loss_fn = diffusion_loss(betas, alphas_cumprod)

        data_iter = diffusion_batch_generator(
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            img_size=args.img_size,
            in_ch=args.in_ch,
            alphas_cumprod=alphas_cumprod,
            device=device,
            dtype=dtype,
            reuse_batch=args.reuse_batch,
            memory_format=memory_format,
        )
        steps = steps_per_epoch(args.dataset_size, args.batch_size)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        if device.type == "xpu" and ipex is not None:
            opt_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32
            model, opt = ipex.optimize(model, optimizer=opt, dtype=opt_dtype, inplace=True)

        cfg = TrainConfig(
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            steps_per_epoch=steps,
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            log_interval=args.log_interval,
        )
        print(f"[INFO] Diffusion Fused Linear+GELU: {'ENABLED' if fused_active else 'DISABLED'}")
        fused_conv_active = sum(1 for m in model.modules() if isinstance(m, ConvSiLU) and m.is_active)
        total_conv = sum(1 for m in model.modules() if isinstance(m, ConvSiLU))
        print(f"[INFO] Diffusion Fused Conv+SiLU: {'ENABLED' if fused_conv_active == total_conv and total_conv > 0 else 'DISABLED'}")
        train_loop(cfg, model, data_iter, loss_fn, optimizer=opt)
        return


if __name__ == "__main__":
    main()
