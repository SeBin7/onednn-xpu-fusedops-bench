#!/usr/bin/env bash
set -euo pipefail

# oneDNN 경로(없으면 기본)
export DNNL_INC="${DNNL_INC:-/opt/intel/oneapi/dnnl/latest/include}"
export DNNL_LIB="${DNNL_LIB:-/opt/intel/oneapi/dnnl/latest/lib}"

# Torch lib 경로
TORCH_LIB="$(python - <<'PY'
import torch, pathlib
print(pathlib.Path(torch.__file__).parent/'lib')
PY
)"
echo "[INFO] DNNL_INC=${DNNL_INC}"
echo "[INFO] DNNL_LIB=${DNNL_LIB}"
echo "[INFO] TORCH_LIB=${TORCH_LIB}"

# 심볼 체크(interop 있어야 함)
if [ ! -e "${DNNL_LIB}/libdnnl.so" ] && [ -z "$(ls -1 ${DNNL_LIB}/libdnnl.so* 2>/dev/null || true)" ]; then
  echo "[ERROR] ${DNNL_LIB}/libdnnl.so not found"; exit 1
fi
DNNL_SO="$(readlink -f ${DNNL_LIB}/libdnnl.so)"
nm -D "${DNNL_SO}" | grep -q dnnl_sycl_interop_stream_create || {
  echo "[ERROR] ${DNNL_SO} has no SYCL interop symbols"; exit 1; }

# setup 스크립트 생성(절대경로 링크 + rpath + as-needed 해제)
cat >/tmp/setup_myops.py <<PY
import os, pathlib
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, include_paths, library_paths
import torch

name = "myops_xpu"
srcs = ["cpp/myops_fused_ops.cpp"]

torch_inc = include_paths()
torch_libs = library_paths()
torch_lib_dir = pathlib.Path(torch.__file__).parent / "lib"

dnnl_inc = pathlib.Path(os.environ.get("DNNL_INC", "/opt/intel/oneapi/dnnl/latest/include"))
dnnl_lib = pathlib.Path(os.environ.get("DNNL_LIB", "/opt/intel/oneapi/dnnl/latest/lib"))
dnnl_so  = dnnl_lib / "libdnnl.so"
if dnnl_so.is_symlink():
    dnnl_so = dnnl_so.resolve()   # e.g., libdnnl.so.3.8

ext = CppExtension(
    name,
    sources=srcs,
    include_dirs = list(set([*torch_inc, str(dnnl_inc)])),
    library_dirs = list(set([*library_paths(), str(torch_lib_dir), str(dnnl_lib)])),
    extra_compile_args=["-O3","-fPIC","-DDNNL_WITH_SYCL=1"],
    extra_link_args=[
        "-Wl,--no-as-needed",
        str(dnnl_so),                              # 절대경로 링크 → DT_NEEDED 보장
        "-Wl,-u,dnnl_primitive_destroy",
        "-Wl,-u,dnnl_sycl_interop_stream_create",
        f"-Wl,-rpath,{dnnl_lib}",
        f"-Wl,-rpath,{torch_lib_dir}",
        "-Wl,--copy-dt-needed-entries",
    ],
    libraries=[],
)

setup(
    name=name,
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
PY

# 빌드
python /tmp/setup_myops.py build_ext --inplace -v

echo "---- built so ----"; ls -lh myops_xpu*.so
echo "---- NEEDED ----";  readelf -d myops_xpu*.so | grep -i NEEDED || true
echo "---- ldd ----";     ldd myops_xpu*.so | egrep -i 'dnnl|c10|torch' || true
echo "[OK] myops_xpu build finished"
