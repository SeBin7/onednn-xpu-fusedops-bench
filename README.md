# Intel XPU Fused Training Benchmarks

This project compares PyTorch baselines with custom fused operators on Intel GPUs (XPU) and CPUs.
It currently ships two oneDNN-based kernels implemented in C++/SYCL and exposed through `myops_xpu`:

- **Linear + Bias + GELU** (MLP workloads)
- **Conv2d + Bias + SiLU** (UNet-style diffusion workloads)

Both operators can fall back to native PyTorch paths at runtime if the fused implementation is unavailable or produces non-finite values. The benchmarking CLI supports CPU, XPU/PyTorch, and XPU/fused modes for apples-to-apples profiling.

---

## Repository Layout

```
.
├── cpp/
│   └── myops_fused_ops.cpp         # Fused Linear+GELU and Conv+SiLU kernels
├── scripts/
│   ├── build_myops_xpu.sh          # Helper to compile myops_xpu*.so with icpx
│   ├── setup_runtime_env.sh        # Exports LD_LIBRARY_PATH and oneAPI runtime links
│   ├── run_mlp_fp32.sh             # CPU / XPU / XPU+fused MLP recipes
│   └── run_diffusion_fp32.sh       # CPU / XPU / XPU+fused diffusion recipes
├── xpu_bench/
│   ├── cli.py                      # Entry point: python -m xpu_bench.cli {mlp,diffusion}
│   ├── core/                       # Data pipeline, training loop, fused op wrappers
│   └── models/                     # MLP stack and UNetTiny diffusion model
├── myops_xpu*.so                   # Built extension (ignored in git)
├── Makefile                        # Convenience build target (requires icpx)
└── README.md
```

---

## Prerequisites

### Hardware & OS
- Intel GPU (Arc / Xe iGPU) for XPU runs; CPU-only mode works everywhere.
- Linux or WSL2 with GPU passthrough enabled (`/dev/dxg` visible inside WSL).

### Toolchain
- **Intel oneAPI Toolkit** (for `icpx`, oneDNN, and runtime libraries).
- **Python 3.11+** and a clean virtual environment.
- **PyTorch 2.5+**:
  - XPU wheels: `pip install --index-url https://download.pytorch.org/whl/xpu torch` (adds `torch_xpu`).
  - CPU wheels: `pip install --index-url https://download.pytorch.org/whl/cpu torch` (optional fallback).
- Recommended Python extras: `pip install pybind11 ninja`.

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('xpu available:', torch.xpu.is_available())
print('xpu count:', torch.xpu.device_count())
PY
```

---

## Environment Setup

Create or activate the project virtual environment, then run:
```bash
source /opt/intel/oneapi/setvars.sh
source scripts/setup_runtime_env.sh
```
`scripts/setup_runtime_env.sh` discovers Torch, oneDNN, and TBB library directories, creates symbolic links for the Intel runtime in `$HOME/.intelrt`, and exports a clean `LD_LIBRARY_PATH`. Keep this script sourced inside shells running the benchmarks.

---

## Building the Fused Extension

The fused operators are compiled into `myops_xpu$(python-config --extension-suffix)` using the provided helper:
```bash
source /opt/intel/oneapi/setvars.sh
./scripts/build_myops_xpu.sh
```
The script resolves Torch include/lib directories through Python, invokes `icpx -fsycl`, and drops `myops_xpu*.so` in the repository root. Re-run the script whenever `cpp/myops_fused_ops.cpp` changes or PyTorch is upgraded.

Manual compilation remains possible via the Makefile target `make xpu`, but the script is the canonical entry point because it assembles all include and library paths in one step.

---

## Running Benchmarks

All benchmarks use the same CLI:
```bash
python -m xpu_bench.cli --help
```
Two workloads are available:
- `mlp` (deep feed-forward network with heavy Linear usage)
- `diffusion` (UNetTiny with Conv blocks and timestep embeddings)

The helper scripts wrap the long command lines and enforce the clean runtime environment. Each script accepts a mode argument: `cpu`, `xpu`, or `xpu_fused`.

### MLP (fp32)
```bash
# CPU baseline
./scripts/run_mlp_fp32.sh cpu

# XPU using standard PyTorch kernels
./scripts/run_mlp_fp32.sh xpu

# XPU using oneDNN fused Linear+GELU
./scripts/run_mlp_fp32.sh xpu_fused
```

### Diffusion (fp32)
```bash
./scripts/run_diffusion_fp32.sh cpu
./scripts/run_diffusion_fp32.sh xpu
./scripts/run_diffusion_fp32.sh xpu_fused
```
`xpu_fused` requests both Linear+GELU and Conv+SiLU fused kernels via `--use-fused`. At startup the CLI prints which fused paths were enabled, so you can confirm whether both operators loaded.

### BF16 Support
- Native PyTorch kernels in `cpu` and `xpu` modes accept `--precision bf16`.
- The fused Linear+GELU path currently assumes FP32 inputs and will raise `RuntimeError: x must be float32` if BF16 tensors reach it.
- For BF16 diffusion experiments, run with `USE_MYOPS`=0 (PyTorch path) or drop back to FP32 when enabling `--use-fused`.

---

## Sample Performance (Arc A770, PyTorch 2.5.1)

| Workload   | Mode          | Precision | Fused Ops              | Throughput (imgs/s) | Notes                  |
|------------|---------------|-----------|------------------------|---------------------|------------------------|
| MLP        | CPU           | fp32      | –                      | ~11                 | 16 OpenMP threads      |
| MLP        | XPU (PyTorch) | fp32      | –                      | ~20                 | Baseline XPU kernels   |
| MLP        | XPU (fused)   | fp32      | Linear+GELU            | ~55                 | oneDNN fused primitive |
| Diffusion  | XPU (fused)   | fp32      | Linear+GELU, Conv+SiLU | ~23                 | timesteps=1000         |

Numbers vary with hardware, driver, and dataset size; use them only as rough guidance.

---

## Troubleshooting

- `LIBUR_LOADER_0.xx not found` or SYCL loader crashes  
  Ensure no stale oneAPI libraries leak into your environment. Always source `scripts/setup_runtime_env.sh` before running benchmarks.

- `myops_xpu` import fails (missing libdnnl / libccl)  
  Re-run `scripts/setup_runtime_env.sh`, double-check oneAPI is installed, and rebuild the extension.

- `myops_xpu.linear_gelu produced non-finite values; falling back`  
  The guard rails detected NaNs/Inf. Investigate your inputs; the CLI automatically switches back to the PyTorch kernel for stability.

- `icpx: not found`  
  Source `/opt/intel/oneapi/setvars.sh` (or the matching installation path).

- BF16 fused run fails with `x must be float32`  
  FP16/BF16 support is not implemented for the fused kernels. Switch to FP32 or disable `--use-fused`.

- Diffusion run reports `UR error` early in training  
  Verify that `LD_LIBRARY_PATH` includes Torch, oneDNN, TBB, and `$HOME/.intelrt` (sourced from the setup script). UR errors often indicate Level Zero runtime misconfiguration.

---

## Roadmap

1. Add BF16 code paths for fused Linear+GELU and Conv+SiLU.
2. Extend fused coverage to LayerNorm and attention-friendly kernels.
3. Bundle representative benchmarks for additional diffusion models and transformer blocks.

---

## License

Choose an open-source license (MIT, BSD-3-Clause, Apache-2.0, etc.) and document it here.
