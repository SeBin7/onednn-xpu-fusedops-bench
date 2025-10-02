#!/usr/bin/env bash
# Diffusion/UNet runner on CPU/XPU with ipex/myops toggles (quiet by default).
# Usage:
#   ./scripts/run_diffusion.sh <cpu|xpu|xpu-ipex|xpu-fused> [options...]
# Common options:
#   --precision {fp32,bf16}
#   --batch-size N --steps N --height H --width W
#   --verbose / --quiet
#   --use-fused / --no-fused (for ipex epilogue)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cpu|xpu|xpu-fused|xpu-ipex> [options...]"; exit 1;
fi

MODE="$1"; shift || true

# -------------------------
# Defaults (safe & quiet)
# -------------------------
device="cpu"
precision="fp32"
batch_size=1
steps=50
height=512
width=512
channels=4

use_myops="auto"
use_ipex_opt="auto"
use_fused_linear_gelu="auto"
prefer_graph="auto"

quiet=1
onednn_verbose=0
onednn_graph_verbose=0
threads=""

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
  cpu)       device="cpu"; use_myops=0; use_ipex_opt=0;;
  xpu)       device="xpu"; use_myops=0; use_ipex_opt=0;;
  xpu-fused) device="xpu"; use_myops=1; use_ipex_opt=0; use_fused_linear_gelu=1;;
  xpu-ipex)  device="xpu"; use_myops=0; use_ipex_opt=1; prefer_graph=1;;
  *) echo "Unknown mode: $MODE"; exit 2;;
esac

# -------------------------
# Parse options
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --precision) precision="${2}"; shift 2;;
    --batch-size) batch_size="${2}"; shift 2;;
    --steps) steps="${2}"; shift 2;;
    --height) height="${2}"; shift 2;;
    --width) width="${2}"; shift 2;;
    --channels) channels="${2}"; shift 2;;
    --use-fused) use_fused_linear_gelu=1; shift;;
    --no-fused)  use_fused_linear_gelu=0; shift;;
    --prefer-graph) prefer_graph=1; shift;;
    --no-graph)     prefer_graph=0; shift;;
    --threads)  threads="${2}"; shift 2;;
    --verbose)  quiet=0; shift;;
    --quiet)    quiet=1; shift;;
    --onednn-verbose)       onednn_verbose="${2}"; shift 2;;
    --onednn-graph-verbose) onednn_graph_verbose="${2}"; shift 2;;
    --) shift; break;;
    *) break;;
  esac
done

# -------------------------
# Environment (quiet by default)
# -------------------------
if [[ "$quiet" -eq 1 ]]; then
  unset ONEDNN_VERBOSE DNNL_VERBOSE ONEDNN_GRAPH_VERBOSE DNNL_GRAPH_VERBOSE
  export TORCH_CPP_LOG_LEVEL=ERROR IPEX_LOG_LEVEL=ERROR
else
  export ONEDNN_VERBOSE="${onednn_verbose}"
  export DNNL_VERBOSE="${onednn_verbose}"
  export ONEDNN_GRAPH_VERBOSE="${onednn_graph_verbose}"
  export DNNL_GRAPH_VERBOSE="${onednn_graph_verbose}"
fi

[[ -n "${threads}" ]] && export OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}"

[[ "$use_myops" != "auto" ]]             && export USE_MYOPS="$use_myops" || true
[[ "$use_ipex_opt" != "auto" ]]          && export USE_IPEX_OPT="$use_ipex_opt" || true
[[ "$use_fused_linear_gelu" != "auto" ]] && export USE_FUSED_LINEAR_GELU="$use_fused_linear_gelu" || true
[[ "$prefer_graph" != "auto" ]]          && export PREFER_IPEX_GRAPH="$prefer_graph" || true

# -------------------------
# Build python args
# -------------------------
args=(
  --device "$device"
  --precision "$precision"
  --batch-size "$batch_size"
  --steps "$steps"
  --height "$height" --width "$width"
  --channels "$channels"
)

# Passthrough remainder
if [[ $# -gt 0 ]]; then args+=( "$@" ); fi

# -------------------------
# Run
# -------------------------
echo "[INFO] device=$device precision=$precision bs=$batch_size size=${height}x${width} steps=$steps"
echo "[INFO] env USE_MYOPS=${USE_MYOPS:-unset} USE_IPEX_OPT=${USE_IPEX_OPT:-unset} USE_FUSED_LINEAR_GELU=${USE_FUSED_LINEAR_GELU:-unset} PREFER_IPEX_GRAPH=${PREFER_IPEX_GRAPH:-unset}"
# Adjust the subcommand below if your repo uses a different entry (e.g., 'unet', 'sd', etc.)
python -m xpu_bench.cli diffusion "${args[@]}" >"$LOG_FILE" 2>&1 | tee "$LOG_FILE"
