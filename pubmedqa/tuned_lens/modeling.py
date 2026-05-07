import torch
from torch import nn


class FullAffineTranslator(nn.Module):
    def __init__(self, hidden_dim, dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype)
        with torch.no_grad():
            eye = torch.eye(hidden_dim, dtype=dtype)
            self.linear.weight.copy_(eye)
            self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)


class LowRankTranslator(nn.Module):
    def __init__(self, hidden_dim, rank, dtype=torch.float32):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False, dtype=dtype)
        self.up = nn.Linear(rank, hidden_dim, bias=False, dtype=dtype)
        self.bias = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))
        with torch.no_grad():
            self.down.weight.zero_()
            self.up.weight.zero_()

    def forward(self, x):
        return x + self.up(self.down(x)) + self.bias


def parse_layer_spec(spec, n_layers):
    if not spec:
        return list(range(n_layers))
    if ":" not in spec:
        idx = int(spec)
        if idx < 0:
            idx += n_layers
        return [idx]
    start_s, end_s, step_s = (spec.split(":") + [""])[:3]
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else n_layers
    step = int(step_s) if step_s else 1
    if start < 0:
        start += n_layers
    if end < 0:
        end += n_layers
    return list(range(start, end, step))


def resolve_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def build_translators(layer_indices, hidden_dim, dtype, rank=0):
    modules = nn.ModuleDict()
    for layer_idx in layer_indices:
        if rank and rank > 0:
            modules[str(layer_idx)] = LowRankTranslator(hidden_dim, rank, dtype=dtype)
        else:
            modules[str(layer_idx)] = FullAffineTranslator(hidden_dim, dtype=dtype)
    return modules
