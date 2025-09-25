#!/usr/bin/env bash

set -euo pipefail

# Torch / oneDNN / TBB
export TORCH_LIB="${TORCH_ROOT:-$(python - <<'PY'
import importlib.util, pathlib
print(pathlib.Path(importlib.util.find_spec('torch').origin).parent / 'lib')
PY
)}"
export DNNL_LIB=/opt/intel/oneapi/dnnl/latest/lib
export TBB_LIB=/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8

# venv 내 추가 so 경로 (선택)
export VENV_LIB="$(dirname "$(dirname "$(which python)")")/lib"

# oneAPI 수학 런타임을 별도 디렉터리에 링크
export ONEAPI_COMP_LIB=/opt/intel/oneapi/compiler/latest/lib
mkdir -p "$HOME/.intelrt"
ln -sf "$ONEAPI_COMP_LIB/libsvml.so"    "$HOME/.intelrt/"
ln -sf "$ONEAPI_COMP_LIB/libimf.so"     "$HOME/.intelrt/"
ln -sf "$ONEAPI_COMP_LIB/libintlc.so.5" "$HOME/.intelrt/"

# LD_LIBRARY_PATH 구성 (ONEAPI_COMP_LIB 는 직접 넣지 않음)
export LD_LIBRARY_PATH="$TORCH_LIB:$DNNL_LIB:$TBB_LIB:$HOME/.intelrt:$VENV_LIB:/usr/lib/x86_64-linux-gnu"

echo "Runtime environment configured." 
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
