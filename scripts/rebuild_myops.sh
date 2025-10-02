#!/usr/bin/env bash
set -euo pipefail

# compilers
CXX_BIN="$(command -v icpx || true)"
[ -z "$CXX_BIN" ] && CXX_BIN="$(command -v dpcpp || true)"
[ -z "$CXX_BIN" ] && CXX_BIN="$(ls -1 /opt/intel/oneapi/compiler/*/bin/icpx 2>/dev/null | tail -n1 || true)"
: "${CXX_BIN:?no DPC++ compiler found (icpx/dpcpp)}"

# torch & pybind11
TORCH_LIB="$(python - <<'PY'
import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
TORCH_INC1="$(python - <<'PY'
import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "include"))
PY
)"
TORCH_INC2="$TORCH_INC1/torch/csrc/api/include"
PYBIND_INC="$(python3 -m pybind11 --includes)"
EXT="$(python - <<'PY'
import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"

# oneDNN (optional)
DNNL_INC="/opt/intel/oneapi/dnnl/latest/include"
DNNL_LIB="/opt/intel/oneapi/dnnl/latest/lib"
[ -d "$DNNL_INC" ] || DNNL_INC=""
[ -d "$DNNL_LIB" ] || DNNL_LIB=""

# clean
rm -f "myops_xpu$EXT" || true
find . -name '*.so' -delete 2>/dev/null || true
rm -rf build/ dist/ .eggs **/*.egg-info ~/.cache/torch_extensions 2>/dev/null || true

# build (전역 경로 건드리지 않고 rpath로만 고정)
set -x
"$CXX_BIN" -fsycl -O3 -fPIC -shared ./cpp/myops_fused_ops.cpp \
  -o "myops_xpu$EXT" \
  $PYBIND_INC \
  -I"$TORCH_INC1" -I"$TORCH_INC2" ${DNNL_INC:+-I"$DNNL_INC"} \
  -L"$TORCH_LIB" -Wl,-rpath,"$TORCH_LIB" \
  ${DNNL_LIB:+-L"$DNNL_LIB" -Wl,-rpath,"$DNNL_LIB" -ldnnl} \
  -Wl,--no-as-needed \
  -ltorch -ltorch_cpu -ltorch_python -lc10
set +x

# 링크 확인
echo "[ldd] deps (sycl/ccl/dnnl)"
ldd "myops_xpu$EXT" | egrep 'libsycl|libccl|libdnnl' || true

# 임포트 스모크 테스트
python - <<'PY'
import myops_xpu
print("import ok; has residual:", hasattr(myops_xpu, "linear_gelu_residual"))
PY
