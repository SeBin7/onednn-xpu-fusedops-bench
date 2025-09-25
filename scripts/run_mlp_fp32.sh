#!/usr/bin/env bash

set -euo pipefail

MODE=${1:-cpu}

usage() {
  cat <<'EOF'
Usage: ./scripts/run_mlp_fp32.sh [cpu|xpu|xpu-fused]
  cpu        - Run MLP benchmark on CPU (fp32, 16 threads)
  xpu        - Run MLP benchmark on XPU (fp32, PyTorch kernels)
  xpu-fused  - Run MLP benchmark on XPU using myops fused Linear+GELU

Make sure the virtual environment is active and myops_xpu is built.
EOF
}

case "$MODE" in
  cpu)
    OMP_NUM_THREADS=16 \
    .venv_xpu_clean/bin/python -m xpu_bench.cli mlp \
      --device cpu --precision fp32 \
      --batch-size 128 --dataset-size 12800 --epochs 3 \
      --layers 20 --hidden 4096 --in-dim 2048 --out-dim 1000 \
      --log-interval 10
    ;;
  xpu)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE=0 \
        USE_MYOPS=0 \
        .venv_xpu_clean/bin/python -m xpu_bench.cli mlp \
          --device xpu --precision fp32 \
          --batch-size 128 --dataset-size 12800 --epochs 3 \
          --layers 20 --hidden 4096 --in-dim 2048 --out-dim 1000 \
          --log-interval 10
    ;;
  xpu-fused)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE=0 \
        USE_MYOPS=1 \
        .venv_xpu_clean/bin/python -m xpu_bench.cli mlp \
          --device xpu --precision fp32 --use-fused \
          --batch-size 128 --dataset-size 12800 --epochs 3 \
          --layers 20 --hidden 4096 --in-dim 2048 --out-dim 1000 \
          --log-interval 10
    ;;
  *)
    usage
    exit 1
    ;;
esac
