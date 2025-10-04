#!/usr/bin/env bash
set -euo pipefail

# ---- Pascal (GTX 1060) ----
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.1}"

# CUDA 툴킷 자동 감지
if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN="$(command -v nvcc)"
  export CUDA_HOME="$(realpath "$(dirname "$CUDA_BIN")/..")"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# 빌드 속도 향상(선택)
python -m pip install -U pip setuptools wheel >/dev/null
python -m pip install -q ninja || true

echo "[INFO] Python   : $(command -v python)"
echo "[INFO] CUDA_HOME: ${CUDA_HOME:-<unset>}"
echo "[INFO] ARCHLIST : ${TORCH_CUDA_ARCH_LIST}"

pushd "$(dirname "$0")/../cuda" >/dev/null
# ★ editable(-e) 제거 + build isolation 비활성화
python -m pip install --no-build-isolation .
popd >/dev/null

python - <<'PY'
import os, importlib, torch, sys
print("[INFO] sys.executable:", sys.executable)
print("[INFO] torch         :", torch.__version__, "cuda:", torch.version.cuda)
m = importlib.import_module("linear_gelu_cuda")
print("[OK]   linear_gelu_cuda imported; ARCH_LIST =", os.environ.get("TORCH_CUDA_ARCH_LIST"))
PY
