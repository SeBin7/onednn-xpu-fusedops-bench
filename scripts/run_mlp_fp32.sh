#!/usr/bin/env bash

[ -f "$(dirname "$0")/setup_runtime_env.sh" ] && . "$(dirname "$0")/setup_runtime_env.sh"

set -euo pipefail

MODE=${1:-cpu}

usage() {
  cat <<'EOF'
Usage: ./scripts/run_mlp_fp32.sh [cpu|cuda|cuda-fused|xpu|xpu-fused]
  cpu        - Run MLP benchmark on CPU (fp32, 16 threads)
  cuda       - Run MLP benchmark on CUDA GPU (fp32, PyTorch kernels)
  cuda-fused - Run MLP benchmark on CUDA GPU with torch.compile acceleration
  xpu        - Run MLP benchmark on XPU (fp32, PyTorch kernels)
  xpu-fused  - Run MLP benchmark on XPU using myops fused Linear+GELU

Make sure the virtual environment is active and myops_xpu is built.
EOF
}

case "$MODE" in
  cpu)
    OMP_NUM_THREADS=16 \
    .venv/bin/python -m xpu_bench.cli mlp \
      --device cpu --precision fp32 \
      --batch-size 64 --dataset-size 6400 --epochs 3 \
      --layers 10 --hidden 4096 --in-dim 2048 --out-dim 1000 \
      --log-interval 10
    ;;
  cuda)
    PYTHONUNBUFFERED=1 \
    .venv/bin/python -m xpu_bench.cli mlp \
      --device cuda --precision fp32 \
      --batch-size 64 --dataset-size 6400 --epochs 3 \
      --layers 10 --hidden 4096 --in-dim 2048 --out-dim 1000 \
      --log-interval 10
    ;;
  cuda-fused)
    # [ADD] Auto fused backend for CUDA
    : "${XPU_BENCH_AUTO_BACKEND:=1}"   # MLP → MLP_AutoFused alias
    : "${XPU_FORCE_FUSED:=1}"          # --use-fused 없어도 강제로 fused 시도
    : "${XPU_BENCH_VERBOSE:=1}"        # 활성/폴백 로그

    export XPU_BENCH_AUTO_BACKEND XPU_FORCE_FUSED XPU_BENCH_VERBOSE

    PYTHONUNBUFFERED=1 \
    .venv/bin/python -m xpu_bench.cli mlp \
      --device cuda --precision fp32 --use-fused \
      --batch-size 64 --dataset-size 6400 --epochs 3 \
      --layers 10 --hidden 4096 --in-dim 2048 --out-dim 1000 \
      --log-interval 10
    ;;
  xpu)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE=0 \
        USE_MYOPS=0 \
        .venv/bin/python -m xpu_bench.cli mlp \
          --device xpu --precision fp32 \
          --batch-size 64 --dataset-size 6400 --epochs 3 \
          --layers 10 --hidden 4096 --in-dim 2048 --out-dim 1000 \
          --log-interval 10
    ;;
  xpu-fused)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE=0 \
        USE_MYOPS=1 \
        .venv/bin/python -m xpu_bench.cli mlp \
          --device xpu --precision fp32 --use-fused \
          --batch-size 64 --dataset-size 6400 --epochs 3 \
          --layers 10 --hidden 4096 --in-dim 2048 --out-dim 1000 \
          --log-interval 10
    ;;
  *)
    usage
    exit 1
    ;;
esac
