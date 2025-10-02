#!/usr/bin/env bash
set -euo pipefail

# 0) oneAPI env (icpx 확보)
if ! command -v icpx >/dev/null 2>&1; then
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi
export CXX="$(command -v icpx || command -v dpcpp)"
: "${CXX:?no DPC++ compiler found}"

# 1) Torch/pybind11 경로(임포트 없이 pip show로)
EXT="$(python -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
TORCH_ROOT="$(python -m pip show torch | awk '/Location:/{print $2}')"
TORCH_DIR="$TORCH_ROOT/torch"
TORCH_INC1="$TORCH_DIR/include"
TORCH_INC2="$TORCH_DIR/include/torch/csrc/api/include"
TORCH_LIB="$TORCH_DIR/lib"
DNNL_INC="/opt/intel/oneapi/dnnl/latest/include"
DNNL_LIB="/opt/intel/oneapi/dnnl/latest/lib"
PYBIND11_INC="$(python3 -m pybind11 --includes)"

# 2) 컴파일 (소스는 ./cpp/myops_fused_ops.cpp 기준)
rm -f "myops_xpu$EXT"
"$CXX" -fsycl -O3 -fPIC -shared ./cpp/myops_fused_ops.cpp \
  -o "myops_xpu$EXT" \
  $PYBIND11_INC \
  -I"$TORCH_INC1" -I"$TORCH_INC2" -I"$DNNL_INC" \
  -L"$TORCH_LIB" -Wl,-rpath,"$TORCH_LIB" \
  -L"$DNNL_LIB"  -Wl,-rpath,"$DNNL_LIB" \
  -ldnnl -ltorch -ltorch_cpu -ltorch_python -lc10

# 3) 런타임 충돌 회피(Torch 번들 우선)
export LD_LIBRARY_PATH="$TORCH_LIB:$DNNL_LIB"

# 4) 간단 로드 테스트
python - <<'PY'
import importlib.util, torch, myops_xpu
print("torch:", torch.__version__)
print("myops_xpu found:", importlib.util.find_spec("myops_xpu") is not None)
print("has linear_gelu_residual:", hasattr(myops_xpu, "linear_gelu_residual"))
PY
