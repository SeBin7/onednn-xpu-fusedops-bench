from __future__ import annotations
import os, time, random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .device import describe_device, synchronize

@dataclass
class TrainConfig:
    device: torch.device
    dtype: torch.dtype
    batch_size: int
    steps_per_epoch: int
    epochs: int
    warmup_steps: int = 10
    log_interval: int = 10

def _set_threads_for_cpu() -> None:
    """Apply OMP_NUM_THREADS if provided; ensure a positive value."""
    try:
        nt = int(os.environ.get("OMP_NUM_THREADS", "0") or "0")
    except Exception:
        nt = 0
    nt = max(1, nt)
    try:
        torch.set_num_threads(nt)
    except Exception:
        pass

def train_loop(
    cfg: TrainConfig,
    model: nn.Module,
    data_iter: Iterable,
    loss_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    optimizer: Optional[optim.Optimizer] = None,
    targets_iter: Optional[Iterable] = None,
) -> None:
    device = cfg.device
    is_train = optimizer is not None
    model.train(is_train)

    if device.type == "cpu":
        _set_threads_for_cpu()

    # Reproducibility (optional)
    _seed = os.getenv("SEED")
    if _seed is not None:
        try:
            s = int(_seed)
            torch.manual_seed(s)
            random.seed(s)
        except Exception:
            pass

    # IPEX optimize: if CLI already did it, skip here (robust guard)
    cli_ipex_done_flag = globals().get("_CLI_IPEX_OPT_DONE", False)
    env_ipex_done_flag = (os.getenv("CLI_IPEX_OPT_DONE", "0") == "1")
    use_ipex = (device.type in ("xpu","cpu")) and (os.getenv("USE_IPEX_OPT", "0") == "1")
    if use_ipex and (cli_ipex_done_flag or env_ipex_done_flag):
        print("[INFO] IPEX optimize: SKIPPED (done in CLI)")
    elif use_ipex:
        import intel_extension_for_pytorch as ipex
        opt_dtype = cfg.dtype if cfg.dtype in (torch.bfloat16, torch.float16) else torch.float32
        opt_kwargs = dict(inplace=True, auto_kernel_selection=True, dtype=opt_dtype)
        if is_train:
            if optimizer is None:
                raise RuntimeError("IPEX training optimize requires an optimizer.")
            model, optimizer = ipex.optimize(model, optimizer=optimizer, **opt_kwargs)
            print(f"[INFO] IPEX optimize: ENABLED (train, dtype={opt_dtype})")
        else:
            model = ipex.optimize(model, **opt_kwargs)
            print(f"[INFO] IPEX optimize: ENABLED (eval, dtype={opt_dtype})")
    else:
        print("[INFO] IPEX optimize: DISABLED")

    print("====== Core Train Loop ======")
    print(f"device          : {describe_device(device)}")
    print(f"precision       : {cfg.dtype}")
    print(f"steps/epoch     : {cfg.steps_per_epoch}")
    print(f"epochs          : {cfg.epochs}")
    print(f"warmup_steps    : {cfg.warmup_steps}")
    print("=============================")

    global_step = 0
    data_stream = iter(data_iter)
    targets_stream = iter(targets_iter) if targets_iter is not None else None

    def _next_batch():
        nonlocal data_stream
        try:
            batch = next(data_stream)
        except StopIteration as exc:
            raise RuntimeError(
                "Data iterator exhausted during training. "
                "Provide an iterator that can yield enough batches."
            ) from exc
        return _unpack_batch(batch, targets_stream)

    use_autocast = _use_autocast(cfg.dtype, device)
    autocast_device = _autocast_device(device)

    # ---- Warmup (not timed for throughput) ----
    if cfg.warmup_steps > 0:
        print(f"\n-- Warmup {cfg.warmup_steps} steps --")
    w_ips_hist, w_step_times = [], []

    for warm_step in range(cfg.warmup_steps):
        t0 = time.time()
        x, y = _next_batch()
        with (torch.autocast(device_type=autocast_device, dtype=cfg.dtype) if use_autocast else _nullcontext()):
            out = _forward_model(model, x)
            loss = loss_fn(out, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        synchronize(device)
        t1 = time.time()

        step_time_ms = (t1 - t0) * 1000.0
        ips = cfg.batch_size / ((t1 - t0) + 1e-9)
        w_step_times.append(step_time_ms)
        w_ips_hist.append(ips)

        if ((warm_step + 1) % cfg.log_interval == 0) or (warm_step + 1 == cfg.warmup_steps):
            loss_val = loss.detach().to(device="cpu", dtype=torch.float32).item()
            print(f"[W{warm_step+1}/{cfg.warmup_steps}] "
                  f"loss={loss_val:.4f} ips={ips:,.1f} step_time={step_time_ms:.2f} ms")

        global_step += 1

    if w_ips_hist:
        mean_ips = sum(w_ips_hist) / len(w_ips_hist)
        mean_t   = sum(w_step_times) / len(w_step_times)
        print(f">>> [Warmup Summary] mean_ips={mean_ips:,.1f} "
              f"mean_step_time={mean_t:.2f} ms")

    # ---- Timed epochs ----
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n-- Epoch {epoch}/{cfg.epochs} --")
        t_epoch_start = time.time()
        loss_accum = torch.zeros((), device="cpu", dtype=torch.float32)
        ips_hist: list[float] = []
        step_times: list[float] = []
        seen_steps = 0

        for step in range(1, cfg.steps_per_epoch + 1):
            t0 = time.time()
            x, y = _next_batch()
            with (torch.autocast(device_type=autocast_device, dtype=cfg.dtype) if use_autocast else _nullcontext()):
                out = _forward_model(model, x)
                loss = loss_fn(out, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            synchronize(device)
            t1 = time.time()

            step_time = (t1 - t0) * 1000.0
            ips = cfg.batch_size / ((t1 - t0) + 1e-9)
            loss_accum = loss_accum + loss.detach().to(device="cpu", dtype=torch.float32)
            seen_steps += 1
            ips_hist.append(ips)
            step_times.append(step_time)
            global_step += 1

            if step % cfg.log_interval == 0 or step == cfg.steps_per_epoch:
                loss_val = loss.detach().to(device="cpu", dtype=torch.float32).item()
                avg_loss = (loss_accum / seen_steps).item()
                print(
                    f"[E{epoch} B{step}/{cfg.steps_per_epoch}] "
                    f"loss={loss_val:.4f} avg_loss={avg_loss:.4f} "
                    f"ips={ips:,.1f} step_time={step_time:.2f} ms"
                )

        t_epoch = time.time() - t_epoch_start
        avg_loss_epoch = (loss_accum / max(seen_steps, 1)).item()
        tp = (cfg.batch_size * seen_steps) / t_epoch if t_epoch > 0 else 0.0
        print(
            f">>> [Epoch {epoch} Summary] avg_loss={avg_loss_epoch:.4f} "
            f"throughput={tp:,.1f} images/sec epoch_time={t_epoch:.2f} s"
        )

def _unpack_batch(batch, targets_iter):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
    else:
        x, y = batch, None
    if y is None and targets_iter is not None:
        try:
            y = next(targets_iter)
        except StopIteration as exc:
            raise RuntimeError(
                "Targets iterator exhausted before data iterator."
            ) from exc
    return x, y

def _autocast_device(device: torch.device) -> str:
    # torch.autocast supports "cuda" and "cpu"
    return "cuda" if device.type == "cuda" else "cpu"

def _use_autocast(dtype: torch.dtype, device: torch.device) -> bool:
    # Enable autocast on CUDA for fp16/bf16, and on CPU for bf16.
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return True
    if device.type == "cpu" and dtype is torch.bfloat16:
        return True
    # XPU path: rely on IPEX optimize (set in CLI) instead of torch.autocast.
    return False

class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False

def _forward_model(model: nn.Module, x):
    if isinstance(x, dict):
        return model(**x)
    if isinstance(x, (tuple, list)):
        return model(*x)
    return model(x)
