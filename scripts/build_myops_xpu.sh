#!/usr/bin/env bash

set -euo pipefail

# Torch include/lib paths (venv 기준)
export TORCH_ROOT=$(python - <<'PY'
import importlib.util, pathlib
print(pathlib.Path(importlib.util.find_spec('torch').origin).parent)
PY
)
export TORCH_LIB="$TORCH_ROOT/lib"

# oneAPI 라이브러리 경로
export DNNL_LIB=/opt/intel/oneapi/dnnl/latest/lib
export TBB_LIB=/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8

# 확장자 추출
export EXT=$(python - <<'PY'
import sysconfig
print(sysconfig.get_config_var('EXT_SUFFIX'))
PY
)

/opt/intel/oneapi/compiler/latest/bin/icpx -fsycl -O3 -fPIC -shared cpp/myops_fused_ops.cpp \
  -o myops_xpu${EXT} \
  $(python -m pybind11 --includes) \
  -I"${TORCH_ROOT}/include" \
  -I"${TORCH_ROOT}/include/torch/csrc/api/include" \
  -L"${TORCH_LIB}" -L"${DNNL_LIB}" -L"${TBB_LIB}" \
  -Wl,-rpath,"${TORCH_LIB}" \
  -Wl,-rpath,"${DNNL_LIB}" \
  -Wl,-rpath,"${TBB_LIB}" \
  -ldnnl -ltorch_xpu -ltorch -ltorch_cpu -ltorch_python -lc10

echo "myops_xpu${EXT} built successfully."
