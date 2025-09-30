#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
MODE=${1:-cpu}
shift || true                      # MODE 제거
EXTRA_ARGS=("$@")                  # 여기로 추가 인자(예: --precision bf16) 전달

usage() {
  cat <<'EOF'
Usage: ./scripts/run_mlp_fp32.sh <cpu|xpu|xpu-fused|xpu-ipex> [-- <extra cli args>]

  cpu        - Run MLP benchmark on CPU (default: fp32, OMP_NUM_THREADS auto)
  xpu        - Run on XPU with PyTorch kernels (no fused, no IPEX)
  xpu-fused  - Run on XPU using myops fused Linear+GELU (USE_MYOPS=1)
  xpu-ipex   - Run on XPU with Intel Extension for PyTorch optimize (USE_IPEX_OPT=1, PREFER_IPEX_GRAPH=1)

Common extra args (forwarded to CLI after --):
  --precision {fp32,bf16}
  --batch-size <int>       --dataset-size <int>
  --epochs <int>           --log-interval <int>
  --layers <int>           --hidden <int>
  --in-dim <int>           --out-dim <int>
  --warmup-steps <int>

Examples:
  ./scripts/run_mlp_fp32.sh xpu-fused -- --precision bf16 --warmup-steps 50
  ./scripts/run_mlp_fp32.sh xpu-ipex  -- --precision bf16 --batch-size 256

Make sure the virtual environment is active and myops_xpu is built (for xpu-fused).
EOF
}

COMMON_ARGS=(
  --batch-size 64
  --dataset-size 6400
  --epochs 3
  --layers 10
  --hidden 4096
  --in-dim 2048
  --out-dim 1000
  --log-interval 10
)

case "$MODE" in
  cpu)
    OMP_NUM_THREADS=16 \
    "${PYTHON_BIN}" -m xpu_bench.cli mlp \
      --device cpu \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
    ;;
  xpu)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE=0 \
        USE_MYOPS=0 \
        "${PYTHON_BIN}" -m xpu_bench.cli mlp \
        --device xpu \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
    ;;
  xpu-fused)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE=0 \
        USE_MYOPS=1 \
        "${PYTHON_BIN}" -m xpu_bench.cli mlp \
        --device xpu --use-fused \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
    ;;
  xpu-ipex)
    env SYCL_DEVICE_FILTER="level_zero:gpu" PYTHONUNBUFFERED=1 ONEDNN_VERBOSE="${ONEDNN_VERBOSE:-0}" \
        USE_MYOPS=0 USE_IPEX_OPT=1 PREFER_IPEX_GRAPH=1 \
        "${PYTHON_BIN}" -m xpu_bench.cli mlp \
          --device xpu \
          "${COMMON_ARGS[@]}" \
          "${EXTRA_ARGS[@]}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
