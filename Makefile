# Makefile (발췌)
ICPX := $(shell command -v icpx 2>/dev/null)
EXT  := $(shell python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

.PHONY: all xpu clean

# 기본 target은 아무 것도 안 함(실수 방지)
all:
	@echo "Nothing to build. Use 'make xpu' on a host/image with icpx to build myops_xpu*.so."

xpu:
ifndef ICPX
	@echo "[SKIP] icpx not found. Build requires Intel oneAPI compiler (icpx)."
	@echo "       Run this on the host or in the XPU image."
else
	icpx -fsycl -O3 -fPIC -shared cpp/myops_fused_ops.cpp \
	  -o myops_xpu$(EXT) \
	  $(shell python -m pybind11 --includes) \
	  -I"$(shell python - <<'PY'\nimport importlib.util, pathlib\np=pathlib.Path(importlib.util.find_spec('torch').origin).parent\nprint(p/'include')\nPY)" \
	  -I"$(shell python - <<'PY'\nimport importlib.util, pathlib\np=pathlib.Path(importlib.util.find_spec('torch').origin).parent\nprint(p/'include/torch/csrc/api/include')\nPY)" \
	  -L"$(shell python - <<'PY'\nimport importlib.util, pathlib\np=pathlib.Path(importlib.util.find_spec('torch').origin).parent\nprint(p/'lib')\nPY)" \
	  -Wl,-rpath,"$(shell python - <<'PY'\nimport importlib.util, pathlib\np=pathlib.Path(importlib.util.find_spec('torch').origin).parent\nprint(p/'lib')\nPY)" \
	  -ldnnl -ltorch -ltorch_cpu -lc10
endif

clean:
	rm -f myops_xpu*.so
