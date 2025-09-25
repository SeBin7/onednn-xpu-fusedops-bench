#!/usr/bin/env bash

set -euo pipefail

MODE=${1:-cpu}

usage() {
  cat <<'EOF'
Usage: ./scripts/run_diffusion_fp32.sh [cpu|xpu|xpu-fused]
  cpu        - Run diffusion benchmark on CPU (fp32, 16 threads)
  xpu        - Run diffusion benchmark on XPU (fp32, PyTorch kernels)
  xpu-fused  - Run diffusion benchmark on XPU using myops fused ops

Make sure the virtual environment is active and myops_xpu is built when using xpu-fused.
EOF
}

COMMON_ARGS=(
  --batch-size 64
  --dataset-size 16384
  --epochs 3
  --img-size 64
  --in-ch 3
  --base-ch 96
  --ch-mult 1,2,4
  --timesteps 1000
  --log-interval 20
)

case "$MODE" in
  cpu)
    OMP_NUM_THREADS=16 \
    .venv_xpu_clean/bin/python -m xpu_bench.cli diffusion \
      --device cpu --precision fp32 \
      "${COMMON_ARGS[@]}"
    ;;
  xpu)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE=0 \
        USE_MYOPS=0 \
        .venv_xpu_clean/bin/python -m xpu_bench.cli diffusion \
          --device xpu --precision fp32 \
          "${COMMON_ARGS[@]}"
    ;;
  xpu-fused)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE=0 \
        USE_MYOPS=1 \
        .venv_xpu_clean/bin/python -m xpu_bench.cli diffusion \
          --device xpu --precision fp32 --use-fused \
          "${COMMON_ARGS[@]}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
