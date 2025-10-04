# [ADD] env 스위치로 패키지 레벨에서 MLP를 오토 백엔드로 alias
import os as _os
try:
    if _os.environ.get("XPU_BENCH_AUTO_BACKEND", "0") == "1":
        from .mlp import MLP_AutoFused as _MLP_AUTO
        MLP = _MLP_AUTO  # 기존 MLP 심볼은 그대로 두고, alias만 추가
        print("[INFO] models.mlp: Using MLP_AutoFused as MLP (env alias)")
except Exception as _e:
    print("[WARN] models.mlp: failed to alias MLP_AutoFused:", _e)
