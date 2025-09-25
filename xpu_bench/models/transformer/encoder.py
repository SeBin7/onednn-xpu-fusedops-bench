"""Tiny Transformer encoder placeholder. Later, wire fused FFN via xpu_bench.ops."""
import torch.nn as nn
class TinyTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("TODO: implement TinyTransformer")
