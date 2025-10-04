# xpu_bench/core/fuse.py
from __future__ import annotations
import os, torch, torch.nn as nn, torch.fx as fx
from .ops import Conv2dSiLU_CUDA_Fused, LinearGELU_Fused, LinearGELU_CUDA_Fused, LinearGELU_Torch

_VERBOSE = os.environ.get("XPU_BENCH_VERBOSE","0")=="1"

def _replace_linear_gelu(graphmod, root):
    # 기존 Linear->GELU 패턴 (있으면 유지)
    modules = dict(graphmod.named_modules())
    g = graphmod.graph
    def is_mod(n, cls): return n.op=="call_module" and isinstance(modules.get(n.target,None), cls)
    pairs=[]
    for n in list(g.nodes):
        if is_mod(n, nn.GELU):
            p = n.args[0] if n.args else None
            if p and is_mod(p, nn.Linear): pairs.append((p,n))
    for lin, gelu in pairs:
        linmod: nn.Linear = modules[lin.target]
        dev = next(linmod.parameters(), torch.tensor(0)).device
        bias = linmod.bias is not None
        # CUDA 우선, 아니면 XPU, 아니면 torch
        if dev.type=="cuda":
            fused = LinearGELU_CUDA_Fused(linmod.in_features, linmod.out_features, bias=bias, device=dev, dtype=linmod.weight.dtype)
        elif dev.type=="xpu":
            fused = LinearGELU_Fused(linmod.in_features, linmod.out_features, bias=bias, device=dev, dtype=linmod.weight.dtype)
        else:
            fused = LinearGELU_Torch(linmod.in_features, linmod.out_features, bias=bias, device=dev, dtype=linmod.weight.dtype)
        with torch.no_grad():
            fused.weight.copy_(linmod.weight)
            if bias and fused.bias is not None and linmod.bias is not None:
                fused.bias.copy_(linmod.bias)
        name=f"fused_{lin.target.replace('.','_')}"
        i=0;base=name
        while hasattr(root,name): i+=1; name=f"{base}_{i}"
        setattr(root,name,fused)
        with g.inserting_after(lin):
            fnode = g.call_module(name, args=lin.args, kwargs={})
        gelu.replace_all_uses_with(fnode)
        g.erase_node(gelu); g.erase_node(lin)
    g.lint(); graphmod.recompile()
    if _VERBOSE: print(f"[fuse] Linear->GELU fused: {len(pairs)}")

def _replace_conv_silu(graphmod, root):
    modules = dict(graphmod.named_modules())
    g = graphmod.graph
    def is_mod(n, cls): return n.op=="call_module" and isinstance(modules.get(n.target,None), cls)
    pairs=[]
    for n in list(g.nodes):
        if is_mod(n, nn.SiLU):
            p = n.args[0] if n.args else None
            if p and is_mod(p, nn.Conv2d): pairs.append((p,n))
    for conv, silu in pairs:
        convmod: nn.Conv2d = modules[conv.target]
        dev = next(convmod.parameters(), torch.tensor(0)).device
        fused = Conv2dSiLU_CUDA_Fused(convmod.in_channels, convmod.out_channels,
                                      convmod.kernel_size, convmod.stride, convmod.padding,
                                      convmod.dilation, convmod.groups,
                                      bias=(convmod.bias is not None),
                                      device=dev, dtype=convmod.weight.dtype)
        # weight/bias 복사
        with torch.no_grad():
            fused.conv.weight.copy_(convmod.weight)
            if convmod.bias is not None and fused.bias is not None:
                fused.bias.copy_(convmod.bias)
        # 등록/치환
        name=f"fused_{conv.target.replace('.','_')}"
        i=0;base=name
        while hasattr(root,name): i+=1; name=f"{base}_{i}"
        setattr(root,name,fused)
        with g.inserting_after(conv):
            fnode = g.call_module(name, args=conv.args, kwargs={})
        silu.replace_all_uses_with(fnode)
        g.erase_node(silu); g.erase_node(conv)
    g.lint(); graphmod.recompile()
    if _VERBOSE: print(f"[fuse] Conv2d->SiLU fused: {len(pairs)}")

def apply_fusions(model: nn.Module, enable=True) -> nn.Module:
    if not enable: return model
    try:
        gm = fx.symbolic_trace(model)
    except Exception as e:
        if _VERBOSE: print("[fuse] FX trace failed; skip:", e)
        return model
    _replace_linear_gelu(gm, model)
    _replace_conv_silu(gm, model)
    return gm
