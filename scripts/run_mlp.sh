#!/usr/bin/env bash
# Lightweight runner for MLP microbench on CPU/XPU with (myops/ipex) fused paths.
# Usage:
#   ./scripts/run_mlp.sh <cpu|xpu|xpu-fused|xpu-ipex|xpu-fused-residual|xpu-ipex-residual> [options...]
# Examples:
#   ./scripts/run_mlp.sh xpu-fused --precision fp32 --layers 20
#   ./scripts/run_mlp.sh xpu-fused-residual --precision fp32
#   ./scripts/run_mlp.sh xpu-ipex-residual  --precision bf16
#
# Notes:
# - xpu-fused : USE_MYOPS=1 (custom fused kernels)
# - xpu-ipex  : USE_IPEX_OPT=1 (IPEX optimize path). --use-fused 로 Linear+GELU epilogue on
# - 조용한 실행이 기본(원하면 --verbose, --onednn-verbose N)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cpu|cpu-ipex|xpu|xpu-fused|xpu-ipex> [options...]"; exit 1;
fi

MODE="$1"; shift || true

# -------------------------
# Defaults (safe & quiet)
# -------------------------
device="cpu"
precision="fp32"
batch_size=64
dataset_size=6400
epochs=3
warmup_steps=50
log_interval=10
layers=12
in_dim=4096
hidden=4096
out_dim=4096

# Feature toggles
use_myops="auto"            # 1 for xpu-fused
use_ipex_opt="auto"         # 1 for xpu-ipex
use_fused_linear_gelu="auto"
prefer_graph="auto"         # IPEX graph preference
residual_mlp=0
residual_scale="0.05"
residual_zero_init=0

# Logging/verbosity
quiet=1
onednn_verbose=0            # 0(off)/1(summary)/2(kernels)
onednn_graph_verbose=0
threads="16"                  # CPU threads (optional)

# -------------------------
# logging dir 
# -------------------------
LOG_DIR="${LOG_DIR:-./logs}"; mkdir -p "$LOG_DIR"
TS="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/${MODE:-cpu}_${TS}.log"
export DNNL_VERBOSE="${DNNL_VERBOSE:-2}"
export ONEDNN_VERBOSE="${ONEDNN_VERBOSE:-2}"
export DNNL_VERBOSE_TIMESTAMP=1
echo "[INFO] logs -> $LOG_FILE"

# -------------------------
# Mode → env defaults
# -------------------------
case "$MODE" in
  cpu)
    device="cpu"
    use_myops=0
    use_ipex_opt=0
    ;;
  cpu-ipex)
    device="cpu"
    use_myops=0
    use_ipex_opt=1
    ;;
  xpu)
    device="xpu"
    use_myops=0
    use_ipex_opt=0
    ;;
  xpu-fused)
    device="xpu"
    use_myops=1
    use_ipex_opt=0
    use_fused_linear_gelu=1
    ;;
  xpu-ipex)
    device="xpu"
    use_myops=0
    use_ipex_opt=1
    prefer_graph=1
    ;;
  xpu-fused-residual)
    device="xpu"
    use_myops=1
    use_ipex_opt=0
    use_fused_linear_gelu=1
    residual_mlp=1
    ;;

  xpu-ipex-residual)
    device="xpu"
    use_myops=0
    use_ipex_opt=1
    prefer_graph=1
    residual_mlp=1
    ;;  
  *)
    echo "Unknown mode: $MODE (expected: cpu|xpu|xpu-fused|xpu-ipex)"; exit 2;;
esac

# -------------------------
# Parse options
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --precision)          precision="${2}"; shift 2;;
    --batch-size)         batch_size="${2}"; shift 2;;
    --dataset-size)       dataset_size="${2}"; shift 2;;
    --epochs)             epochs="${2}"; shift 2;;
    --warmup-steps)       warmup_steps="${2}"; shift 2;;
    --log-interval)       log_interval="${2}"; shift 2;;
    --layers)             layers="${2}"; shift 2;;
    --in-dim)             in_dim="${2}"; shift 2;;
    --hidden)             hidden="${2}"; shift 2;;
    --out-dim)            out_dim="${2}"; shift 2;;

    --residual-mlp)       residual_mlp=1; shift;;
    --residual-scale)     residual_scale="${2}"; shift 2;;
    --residual-zero-init) residual_zero_init=1; shift;;

    --use-fused)          use_fused_linear_gelu=1; shift;;
    --no-fused)           use_fused_linear_gelu=0; shift;;

    --prefer-graph)       prefer_graph=1; shift;;
    --no-graph)           prefer_graph=0; shift;;

    --threads)            threads="${2}"; shift 2;;

    --verbose)            quiet=0; shift;;
    --quiet)              quiet=1; shift;;

    --onednn-verbose)         onednn_verbose="${2}"; shift 2;;
    --onednn-graph-verbose)   onednn_graph_verbose="${2}"; shift 2;;

    --) shift; break;;
    *)  # passthrough later
        break;;
  esac
done

# -------------------------
# Environment (quiet by default)
# -------------------------
if [[ "$quiet" -eq 1 ]]; then
  unset ONEDNN_VERBOSE DNNL_VERBOSE ONEDNN_GRAPH_VERBOSE DNNL_GRAPH_VERBOSE
  export TORCH_CPP_LOG_LEVEL=ERROR IPEX_LOG_LEVEL=ERROR
else
  # map to both old/new names for safety
  export ONEDNN_VERBOSE="${onednn_verbose}"
  export DNNL_VERBOSE="${onednn_verbose}"
  export ONEDNN_GRAPH_VERBOSE="${onednn_graph_verbose}"
  export DNNL_GRAPH_VERBOSE="${onednn_graph_verbose}"
fi

# Threads if requested (CPU only)
if [[ -n "${threads}" ]]; then
  export OMP_NUM_THREADS="${threads}"
  export MKL_NUM_THREADS="${threads}"
fi

# Feature envs (set only if not 'auto')
[[ "$use_myops" != "auto" ]]            && export USE_MYOPS="$use_myops" || true
[[ "$use_ipex_opt" != "auto" ]]         && export USE_IPEX_OPT="$use_ipex_opt" || true
[[ "$use_fused_linear_gelu" != "auto" ]]&& export USE_FUSED_LINEAR_GELU="$use_fused_linear_gelu" || true
[[ "$prefer_graph" != "auto" ]]         && export PREFER_IPEX_GRAPH="$prefer_graph" || true

# Residual MLP knobs
export RESIDUAL_MLP="$residual_mlp"
export RESIDUAL_SCALE="$residual_scale"
export RESIDUAL_ZERO_INIT="$residual_zero_init"

# -------------------------
# Build python args
# -------------------------
args=(
  --device "$device"
  --precision "$precision"
  --batch-size "$batch_size"
  --dataset-size "$dataset_size"
  --epochs "$epochs"
  --warmup-steps "$warmup_steps"
  --log-interval "$log_interval"
  --layers "$layers"
  --in-dim "$in_dim"
  --hidden "$hidden"
  --out-dim "$out_dim"
)
(( residual_mlp == 1 )) && args+=( --residual-mlp )

# Passthrough remainder
if [[ $# -gt 0 ]]; then args+=( "$@" ); fi

# -------------------------
# Run
# -------------------------
echo "[INFO] device=$device precision=$precision batch=$batch_size layers=$layers hidden=$hidden"
echo "[INFO] env USE_MYOPS=${USE_MYOPS:-unset} USE_IPEX_OPT=${USE_IPEX_OPT:-unset} USE_FUSED_LINEAR_GELU=${USE_FUSED_LINEAR_GELU:-unset} PREFER_IPEX_GRAPH=${PREFER_IPEX_GRAPH:-unset}"
python -m xpu_bench.cli mlp "${args[@]}" 2>&1 | tee "$LOG_FILE"
