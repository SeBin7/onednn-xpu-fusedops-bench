#!/usr/bin/env bash
set -euo pipefail

# pip 업그레이드 + 의존성
if [ -f "/workspace/requirement.txt" ]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -r /workspace/requirement.txt
fi

# 토치/아이펙스 등 설치가 requirement에 있다면 여기서 잡힘.
# XPU 가용성 체크(실패해도 진행)
python3 - << 'PY' || true
try:
    import torch
    has_xpu = hasattr(torch, "xpu")
    print(f"[check] torch={torch.__version__}, has_xpu={has_xpu}")
    if has_xpu:
        try:
            devs = torch.xpu.device_count()
            print(f"[check] xpu device_count={devs}")
        except Exception as e:
            print("[warn] torch.xpu not usable:", e)
except Exception as e:
    print("[warn] torch import failed:", e)
PY
