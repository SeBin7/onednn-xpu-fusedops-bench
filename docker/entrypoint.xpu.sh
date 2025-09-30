#!/usr/bin/env bash
set -euo pipefail

# conda env PATH 보장
export PATH=/opt/conda/envs/idp/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# XPU 필터
export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

# Torch/oneDNN 런타임 경로 보강
DNNL_LIB_DEFAULT=/opt/intel/oneapi/dnnl/latest/lib
TORCH_LIB="$(python - <<'PY'
import torch, pathlib
print(pathlib.Path(torch.__file__).parent/'lib')
PY
)" || TORCH_LIB=""
if [ -n "${TORCH_LIB}" ] && [ -d "${TORCH_LIB}" ]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
if [ -d "${DNNL_LIB_DEFAULT}" ]; then
  export LD_LIBRARY_PATH="${DNNL_LIB_DEFAULT}:${LD_LIBRARY_PATH:-}"
fi

# 스크립트 권한 보정(있으면)
chmod +x scripts/*.sh 2>/dev/null || true

# fused .so 자동 빌드(없거나 import 실패 시)
NEED_BUILD=0
if ! ls myops_xpu*.so >/dev/null 2>&1; then
  NEED_BUILD=1
else
  python - <<'PY' || NEED_BUILD=1
import glob, importlib.util, sys
cand = glob.glob('myops_xpu*.so')
assert cand, "no so"
p = cand[0]
spec = importlib.util.spec_from_file_location("myops_xpu", p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("myops_xpu import OK:", p)
PY
fi
if [ "${NEED_BUILD}" = "1" ]; then
  export DNNL_INC=/opt/intel/oneapi/dnnl/latest/include
  export DNNL_LIB=/opt/intel/oneapi/dnnl/latest/lib
  bash scripts/build_myops_xpu.sh
fi

# 기본 셸
if [ $# -eq 0 ]; then
  exec bash
else
  exec "$@"
fi
