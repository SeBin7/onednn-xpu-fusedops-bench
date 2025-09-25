from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .device import pick_device, get_dtype, describe_device, synchronize

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
    # set_num_threads 는 양수만 허용 → 안전 가드
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
    loss_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    optimizer: Optional[optim.Optimizer] = None,
    targets_iter: Optional[Iterable] = None,
) -> None:
    device = cfg.device
    is_train = optimizer is not None
    model.train(is_train)

    if device.type == "cpu":
        _set_threads_for_cpu()

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
                "Provide an iterator that can yield enough batches (e.g., an infinite generator)."
            ) from exc
        return _unpack_batch(batch, targets_stream)

    use_autocast = _use_autocast(cfg.dtype, device)
    autocast_device = _autocast_device(device)

    # warmup (runs once before timed epochs)
    for warm_step in range(cfg.warmup_steps):
        x, y = _next_batch()
        with torch.autocast(device_type=autocast_device, dtype=cfg.dtype) if use_autocast else _nullcontext():
            out = _forward_model(model, x)
            loss = loss_fn(out, y)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        synchronize(device)
        global_step += 1

    # timed epochs
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n-- Epoch {epoch}/{cfg.epochs} --")
        t_epoch_start = time.time()
        loss_accum = torch.zeros((), device="cpu", dtype=torch.float32)
        ips_hist = []
        step_times = []
        seen_steps = 0

        for step in range(1, cfg.steps_per_epoch + 1):
            t0 = time.time()
            x, y = _next_batch()
            with torch.autocast(device_type=autocast_device, dtype=cfg.dtype) if use_autocast else _nullcontext():
                out = _forward_model(model, x)
                loss = loss_fn(out, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            synchronize(device)
            t1 = time.time()

            step_time = (t1 - t0) * 1000.0  # ms
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
                "Targets iterator exhausted before data iterator. "
                "Ensure targets_iter can yield as many items as needed."
            ) from exc
    return x, y

def _autocast_device(device: torch.device) -> str:
    # torch.autocast는 "cuda"/"cpu"만 명시적 지원.
    # XPU는 내부적으로 cpu path와 유사 처리되므로 autocast는 사용하지 않고 dtype만 맞춤.
    return "cuda" if device.type == "cuda" else "cpu"

def _use_autocast(dtype: torch.dtype, device: torch.device) -> bool:
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return True
    # XPU는 PyTorch autocast 경로 미사용(여기서는 명시적 캐스팅 케이스를 모델/데이터가 처리)
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
