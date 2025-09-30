from __future__ import annotations
import os, time, random
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

    # ---- 실행 경로/재현성 명확화 ----
    # (선택) 재현성: 환경변수 SEED 가 있으면 고정
    _seed = os.getenv("SEED")
    if _seed is not None:
        try:
            s = int(_seed)
            torch.manual_seed(s)
            random.seed(s)
        except Exception:
            pass

    # ipex 최적화 경로 토글 (XPU 전용)
    use_ipex = (device.type == "xpu") and (os.getenv("USE_IPEX_OPT", "0") == "1")
    if use_ipex:
        import intel_extension_for_pytorch as ipex
        # dtype 전달 금지: 파라미터를 FP32로 유지 (연산은 bf16로)
        opt_kwargs = dict(inplace=True, auto_kernel_selection=True)

        # bf16 + SGD 조합은 IPEX fused SGD 미지원 → optimizer는 감싸지 말고 model만 optimize
        if isinstance(optimizer, optim.SGD) and cfg.dtype is torch.bfloat16:
            print("[INFO] IPEX: bf16 + SGD -> model-only optimize (no fused SGD)")
            model = ipex.optimize(model, **opt_kwargs)
        else:
            if is_train and optimizer is not None:
                model, optimizer = ipex.optimize(model, optimizer=optimizer, **opt_kwargs)
            else:
                model = ipex.optimize(model, **opt_kwargs)

        print("[INFO] IPEX optimize: ENABLED")
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
                "Provide an iterator that can yield enough batches (e.g., an infinite generator)."
            ) from exc
        return _unpack_batch(batch, targets_stream)

    use_autocast = _use_autocast(cfg.dtype, device)
    autocast_device = _autocast_device(device)

    # warmup (runs once before timed epochs)
    if cfg.warmup_steps > 0:
        print(f"\n-- Warmup {cfg.warmup_steps} steps --")
    w_ips_hist, w_step_times = [], []

    for warm_step in range(cfg.warmup_steps):
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

        step_time_ms = (t1 - t0) * 1000.0
        ips = cfg.batch_size / ((t1 - t0) + 1e-9)
        w_step_times.append(step_time_ms)
        w_ips_hist.append(ips)

        # 웜업 단계 로그 (log_interval마다 혹은 마지막 스텝)
        if ((warm_step + 1) % cfg.log_interval == 0) or (warm_step + 1 == cfg.warmup_steps):
            loss_val = loss.detach().to(device="cpu", dtype=torch.float32).item()
            print(f"[W{warm_step+1}/{cfg.warmup_steps}] "
                  f"loss={loss_val:.4f} ips={ips:,.1f} step_time={step_time_ms:.2f} ms")

        global_step += 1

    if w_ips_hist:
        # 간단 요약
        mean_ips = sum(w_ips_hist) / len(w_ips_hist)
        mean_t   = sum(w_step_times) / len(w_step_times)
        print(f">>> [Warmup Summary] mean_ips={mean_ips:,.1f} "
              f"mean_step_time={mean_t:.2f} ms")


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
