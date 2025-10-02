from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except Exception:
    ipex = None

# core
from .core.device import pick_device, get_dtype
from .core.train import TrainConfig, train_loop
from .core.data import (
    mlp_batch_generator,
    diffusion_batch_generator,
    steps_per_epoch,
)
from .core.ops import (
    has_myops,
    LinearGELU_Fused,
    LinearGELUResidual_Fused,
    ConvSiLU,
    _safe_to_fp32,
    _safe_to_bf16,
)

# algos
from .algos.mlp import build_mlp_model, mlp_loss
from .algos.diffusion import build_diffusion_model, diffusion_loss, make_beta_schedule


# -------------------- helpers --------------------
def _truthy(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "on"}

def _build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    if args.opt == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
        print(f"[INFO] Optimizer: SGD(lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay})")
        return opt
    # default: AdamW
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"[INFO] Optimizer: AdamW(lr={args.lr}, wd={args.weight_decay})")
    return opt

def _env_flag(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cpu", choices=["cpu", "xpu", "cuda"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dataset-size", type=int, default=12800)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--reuse-batch", action="store_true")
    p.add_argument("--ipex", action="store_true")
    p.add_argument("--channels-last", action="store_true")
    # optimizer options
    p.add_argument("--opt", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)  # SGD 용
    # (MLP 전용) residual add 토글: shapes가 맞는 블록에서만 적용
    p.add_argument("--residual-mlp", action="store_true",
                   help="Enable residual add inside MLP blocks when shapes match")

def _apply_fuse_gates(args) -> None:
    """
    --use-fused: 둘 다 ON (명시 스위치 없을 때 기본)
    --fuse-linear-gelu / --no-fuse-linear-gelu: LinearGELU 개별 제어
    --fuse-conv-silu  / --no-fuse-conv-silu : ConvSiLU 개별 제어 (diffusion)
    """
    if getattr(args, "use_fused", False):
        os.environ.setdefault("USE_FUSED_LINEAR_GELU", "1")
        os.environ.setdefault("USE_FUSED_CONV_SILU", "1")

    if getattr(args, "fuse_linear_gelu", None) is not None:
        os.environ["USE_FUSED_LINEAR_GELU"] = "1" if args.fuse_linear_gelu else "0"
    if hasattr(args, "fuse_conv_silu") and getattr(args, "fuse_conv_silu", None) is not None:
        os.environ["USE_FUSED_CONV_SILU"] = "1" if args.fuse_conv_silu else "0"

    print(f"[INFO] FuseGate Linear+GELU={os.getenv('USE_FUSED_LINEAR_GELU','(unset)')}, "
          f"Conv+SiLU={os.getenv('USE_FUSED_CONV_SILU','(unset)')}")


def _count_active_fused_layers(model: torch.nn.Module) -> tuple[int, int]:
    """
    LinearGELU_Fused/ConvSiLU 등 'is_active' 속성을 가진 모듈만 집계.
    속성이 없으면 비활성(False)로 간주.
    """
    active = 0
    total = 0
    for module in model.modules():
        if hasattr(module, "is_active"):
            total += 1
            if getattr(module, "is_active", False):
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


class _BlockBridgeDiffusion(nn.Module):
    """
    Diffusion 전용 BF16 브릿지 래퍼:
      - 입력(x)이 BF16@XPU이면 FP32로 올려 모델 전체 실행 → 출력만 BF16으로 내림
      - 파라미터는 FP32 유지 (AdamW 등과 안전)
      - 캐스트는 입/출구 2회로 고정
    """
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if (
            x.device.type == "xpu"
            and x.dtype == torch.bfloat16
            and _truthy(os.environ.get("BF16_FUSED_BRIDGE", "0"))
        ):
            x32 = _safe_to_fp32(x)           # ① upcast once
            y32 = self.inner(x32, t)
            return _safe_to_bf16(y32)        # ② downcast once
        return self.inner(x, t)


# -------------------- main --------------------
def main() -> None:
    ap = argparse.ArgumentParser("xpu_bench")
    sub = ap.add_subparsers(dest="task", required=True)

    # -------- Microbench: LinearGELU vs LinearGELUResidual --------
    p_bench = sub.add_parser("bench-op", help="Microbench for LinearGELU vs LinearGELUResidual")
    p_bench.add_argument("--device", default="xpu", choices=["cpu","xpu","cuda"])
    p_bench.add_argument("--precision", default="fp32", choices=["fp32","bf16"])
    p_bench.add_argument("--batch", type=int, default=1024)
    p_bench.add_argument("--dim", type=int, default=4096)
    p_bench.add_argument("--iters", type=int, default=200)
    p_bench.add_argument("--warmup", type=int, default=50)

    # MLP
    p_mlp = sub.add_parser("mlp", help="Linear-heavy MLP benchmark")
    _common_args(p_mlp)
    p_mlp.add_argument("--in-dim", type=int, default=2048)
    p_mlp.add_argument("--hidden", type=int, default=4096)
    p_mlp.add_argument("--layers", type=int, default=20)
    p_mlp.add_argument("--out-dim", type=int, default=1000)
    p_mlp.add_argument(
        "--use-fused", action="store_true",
        help="Use C++ fused Linear+GELU if available & device=xpu"
    )
    # 선택적 퓨즈 게이트(MLP: Linear+GELU만)
    p_mlp.add_argument("--fuse-linear-gelu", dest="fuse_linear_gelu", action="store_true")
    p_mlp.add_argument("--no-fuse-linear-gelu", dest="fuse_linear_gelu", action="store_false")
    p_mlp.set_defaults(fuse_linear_gelu=None)

    # Diffusion (tiny UNet)
    p_diff = sub.add_parser("diffusion", help="Tiny UNet diffusion benchmark")
    _common_args(p_diff)
    p_diff.add_argument("--img-size", type=int, default=64)
    p_diff.add_argument("--in-ch", type=int, default=3)
    p_diff.add_argument("--base-ch", type=int, default=64)
    p_diff.add_argument("--ch-mult", type=str, default="1,2,4")
    p_diff.add_argument("--timesteps", type=int, default=1000)
    p_diff.add_argument(
        "--use-fused", action="store_true",
        help="Use C++ fused ops when available & device=xpu"
    )
    # 선택적 퓨즈 게이트(Diffusion: Linear+GELU, Conv+SiLU 둘 다)
    p_diff.add_argument("--fuse-linear-gelu", dest="fuse_linear_gelu", action="store_true")
    p_diff.add_argument("--no-fuse-linear-gelu", dest="fuse_linear_gelu", action="store_false")
    p_diff.set_defaults(fuse_linear_gelu=None)
    p_diff.add_argument("--fuse-conv-silu", dest="fuse_conv_silu", action="store_true")
    p_diff.add_argument("--no-fuse-conv-silu", dest="fuse_conv_silu", action="store_false")
    p_diff.set_defaults(fuse_conv_silu=None)

    args = ap.parse_args()

    # oneDNN BF16 선호
    if args.precision == "bf16" and os.environ.get("ONEDNN_DEFAULT_FPMATH_MODE") is None:
        os.environ["ONEDNN_DEFAULT_FPMATH_MODE"] = "BF16"

    # bf16 + fused 요청이면 브릿지 기본 ON
    if args.precision == "bf16" and getattr(args, "use_fused", False):
        os.environ.setdefault("BF16_FUSED_BRIDGE", "1")

    device = pick_device(args.device)
    dtype = get_dtype(args.precision)

    # =========================
    # bench-op task
    # =========================
    if args.task == "bench-op":
        import time
        # 입력 텐서 준비
        N, D = args.batch, args.dim
        x = torch.randn(N, D, device=device, dtype=dtype)
        r = x.clone()

        m1 = LinearGELU_Fused(D, D, bias=True).to(device)
        m2 = LinearGELUResidual_Fused(D, D, bias=True, device=device, dtype=dtype).to(device)

        # warmup & measure helper
        def run(op, use_residual=False):
            for _ in range(args.warmup):
                _ = op(x) if not use_residual else op(x, r)
            if device.type == "xpu":
                torch.xpu.synchronize()
            t0 = time.time()
            for _ in range(args.iters):
                _ = op(x) if not use_residual else op(x, r)
            if device.type == "xpu":
                torch.xpu.synchronize()
            dt = time.time() - t0
            return (dt * 1000.0 / args.iters), (N * args.iters) / dt

        # 실행
        print(f"[INFO] bench-op: device={device.type} dtype={dtype} N={N} D={D} iters={args.iters} warmup={args.warmup}")
        t1, ips1 = run(m1, use_residual=False)
        t2, ips2 = run(m2, use_residual=True)

        # 결과
        print("=== Microbench Result (per-step) ===")
        print(f"LinearGELU            : {t1:.3f} ms | {ips1:,.1f} samples/s")
        print(f"LinearGELU+Residual   : {t2:.3f} ms | {ips2:,.1f} samples/s")
        return


    # =========================
    # MLP task
    # =========================
    if args.task == "mlp":
        env_use_fused = _env_flag("USE_MYOPS")
        fused_requested = args.use_fused or env_use_fused
        bridge_on = _truthy(os.environ.get("BF16_FUSED_BRIDGE", "0"))
        fused_active = (
            fused_requested
            and device.type == "xpu"
            and has_myops()
            and (
                dtype == torch.float32
                or (dtype == torch.bfloat16 and bridge_on)  # BF16 + Bridge 허용
            )
        )
        print(f"[INFO] BF16 Fused Bridge: {'ENABLED' if (dtype == torch.bfloat16 and bridge_on) else 'DISABLED'}")

        # (1) Residual-MLP 토글을 ENV로 내려서 하위 스택(ops/models/algos) 어느 경로든 읽게 함
        os.environ["RESIDUAL_MLP"] = "1" if getattr(args, "residual_mlp", False) else "0"
        print(f"[INFO] Residual MLP: {'ENABLED' if args.residual_mlp else 'DISABLED'}")

        # (2) 퓨즈 게이트 환경변수 적용 (모델 빌드 전에!)
        _apply_fuse_gates(args)

        model = build_mlp_model(
            in_dim=args.in_dim,
            hidden=args.hidden,
            layers=args.layers,
            out_dim=args.out_dim,
            use_fused=fused_active,
            device=device,
        )

        if fused_active:
            if dtype == torch.float32:
                _preflight_mlp(
                    model,
                    in_dim=args.in_dim,
                    batch_size=args.batch_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                print("[INFO] Skip fused preflight for bf16 (bridge mode)")
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
        opt = _build_optimizer(model, args)

        # IPEX optimize: --ipex 지정시에만
        if device.type == "xpu" and ipex is not None and args.ipex:
            try:
                opt_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32
                model, opt = ipex.optimize(model, optimizer=opt, dtype=opt_dtype, inplace=True)
                # mark done (module flag + env flag)
                from .core import train as _train_mod
                _train_mod._CLI_IPEX_OPT_DONE = True
                os.environ["CLI_IPEX_OPT_DONE"] = "1"
                print("[INFO] CLI IPEX optimize: ENABLED for mlp")
            except Exception as e:
                print(f"[WARN] CLI IPEX optimize failed for mlp: {e}")
        else:
            print("[INFO] CLI IPEX optimize: SKIPPED for mlp")

        cfg = TrainConfig(
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            steps_per_epoch=steps,
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            log_interval=args.log_interval,
        )
        print(f"[INFO] Fused Linear+GELU: {'ENABLED' if fused_active else 'DISABLED'}")

        try:
            from xpu_bench.core.ops import LinearGELUResidual_Fused, LinearGELU_Fused
            fused_types = (LinearGELU_Fused, LinearGELUResidual_Fused)
            lin_modules = [m for m in model.modules() if isinstance(m, fused_types)]
            n_lin_active = sum(1 for m in lin_modules if getattr(m, "is_active", False))
            num_fused     = sum(isinstance(m, LinearGELU_Fused)            for m in model.modules())
            num_fused_res = sum(isinstance(m, LinearGELUResidual_Fused)    for m in model.modules())
            if num_fused or num_fused_res:
                print(f"[INFO] Fused Linear+GELU modules: {num_fused}")
                print(f"[INFO] Fused Linear+GELU+Residual modules: {num_fused_res}")
        except Exception as e:
            print(f"[INFO] Fused module count skipped: {e}")

        train_loop(cfg, model, data_iter, loss_fn, optimizer=opt)
        return

    # =========================
    # Diffusion task
    # =========================
    if args.task == "diffusion":
        ch_mult = tuple(int(x) for x in args.ch_mult.split(",") if x)
        env_use_fused = _env_flag("USE_MYOPS")
        fused_requested = args.use_fused or env_use_fused
        bridge_on = _truthy(os.environ.get("BF16_FUSED_BRIDGE", "0"))

        # BF16 + bridge일 때도 fused 허용 (MLP와 동일 규칙)
        fused_active = (
            fused_requested
            and device.type == "xpu"
            and has_myops()
            and (
                dtype == torch.float32
                or (dtype == torch.bfloat16 and bridge_on)
            )
        )
        print(f"[INFO] BF16 Fused Bridge: {'ENABLED' if (dtype == torch.bfloat16 and bridge_on) else 'DISABLED'}")
        if args.use_fused and not fused_active:
            print("[WARN] --use-fused requested but myops_xpu not available or not allowed; falling back")

        use_channels_last = (args.channels_last or device.type == "xpu")
        memory_format = torch.channels_last if use_channels_last else None

        # ★ BF16 브릿지 모드에선 파라미터는 FP32로 빌드(입출력만 BF16로 브릿지)
        build_dtype = dtype
        if dtype == torch.bfloat16 and bridge_on:
            build_dtype = torch.float32

        # 퓨즈 게이트 환경변수 적용 (모델 빌드 전에!)
        #_apply_fuse_gates(args)
        # --- Diffusion 기본값: Conv+SiLU=ON, Linear+GELU=OFF ---
        if fused_active:
            os.environ.setdefault("USE_FUSED_CONV_SILU", "1")   # conv 우선
        os.environ.setdefault("USE_FUSED_LINEAR_GELU", "0")      # linear 기본 OFF
        # 명시 스위치를 반영 (있으면 override)
        _apply_fuse_gates(args)
        # 최종 게이트 확인 출력
        print(f"[INFO] FuseGate Linear+GELU={os.getenv('USE_FUSED_LINEAR_GELU')}, "
              f"Conv+SiLU={os.getenv('USE_FUSED_CONV_SILU')}")

        model = build_diffusion_model(
            in_ch=args.in_ch,
            base_ch=args.base_ch,
            ch_mult=ch_mult,
            use_fused=fused_active,
            device=device,
            dtype=build_dtype,
            memory_format=memory_format,
        )
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)

        if fused_active:
            if dtype == torch.float32:
                _preflight_diffusion(
                    model,
                    in_ch=args.in_ch,
                    img_size=args.img_size,
                    timesteps=args.timesteps,
                    batch_size=args.batch_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                print("[INFO] Skip fused preflight for bf16 (bridge mode)")
            # 선형/conv 모두 'is_active'가 있을 수도/없을 수도 → 안전 집계
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
        opt = _build_optimizer(model, args)

        # IPEX optimize: --ipex 지정시에만
        if device.type == "xpu" and ipex is not None and args.ipex:
            try:
                opt_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32
                model, opt = ipex.optimize(model, optimizer=opt, dtype=opt_dtype, inplace=True)
                from .core import train as _train_mod
                _train_mod._CLI_IPEX_OPT_DONE = True
                os.environ["CLI_IPEX_OPT_DONE"] = "1"
                print("[INFO] CLI IPEX optimize: ENABLED for diffusion")
            except Exception as e:
                print(f"[WARN] CLI IPEX optimize failed for diffusion: {e}")
        else:
            print("[INFO] CLI IPEX optimize: SKIPPED for diffusion")

        # BF16 브릿지: 모델 전체 래핑 (캐스트 2회로 고정)
        if dtype == torch.bfloat16 and bridge_on:
            model = _BlockBridgeDiffusion(model)

        cfg = TrainConfig(
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            steps_per_epoch=steps,
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            log_interval=args.log_interval,
        )

        # 상태 출력(안전 집계) + 활성 모듈 수
        print(f"[INFO] Diffusion Fused Linear+GELU: {'ENABLED' if fused_active else 'DISABLED'}")
        lin_modules = [m for m in model.modules() if isinstance(m, LinearGELU_Fused)]
        n_lin_active = sum(1 for m in lin_modules if getattr(m, "is_active", False))
        from xpu_bench.core.ops import LinearGELUResidual_Fused, LinearGELU_Fused
        num_fused = sum(1 for m in model.modules() if isinstance(m, LinearGELU_Fused))
        num_fused_res = sum(1 for m in model.modules() if isinstance(m, LinearGELUResidual_Fused))
        print(f"[INFO] Fused Linear+GELU modules: {num_fused}")
        print(f"[INFO] Fused Linear+GELU+Residual modules: {num_fused_res}")

        conv_modules = [m for m in model.modules() if isinstance(m, ConvSiLU)]
        fused_conv_active = sum(1 for m in conv_modules if getattr(m, "is_active", False))
        total_conv = len(conv_modules)
        print(
            f"[INFO] Diffusion Fused Conv+SiLU: "
            f"{'ENABLED' if (total_conv > 0 and fused_conv_active == total_conv) else 'DISABLED'}"
        )
        print(f"[INFO] Fused Conv+SiLU modules:  {fused_conv_active}/{total_conv}")

        train_loop(cfg, model, data_iter, loss_fn, optimizer=opt)
        return


if __name__ == "__main__":
    main()
