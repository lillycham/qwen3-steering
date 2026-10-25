"""A minimal LoRA for the linear layers of a Hugging Face Qwen3: no dependency beyond torch.

    adapters = install(model, r=16, alpha=32)       # the base weights are frozen; only the adapters train
    with disabled(model): ...                       # the base model's behaviour, for a control or a KL term
    state = state_dict_of(model); load_state(model, state)
    merge(model)                                    # fold the adapters into the weights (and remove them)

The adapter of a layer is W x + (alpha / r) B A x, with A [r, in] and B [out, r], B starting at zero, so a fresh
adapter changes nothing. The adapter weights stay in float32 whatever the base dtype is.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            return y
        delta = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B) * self.scaling
        return y + delta.to(y.dtype)


def adapters_of(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def install(model: nn.Module, r: int = 16, alpha: float = 32.0, targets: tuple[str, ...] = TARGETS,
            layers: range | list[int] | None = None) -> dict[str, LoRALinear]:
    """Wrap the target linear layers (of the given decoder layers, default all) and freeze every other parameter."""
    for p in model.parameters():
        p.requires_grad_(False)
    made: dict[str, LoRALinear] = {}
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                full = f"{name}.{child_name}" if name else child_name
                if layers is not None and not any(f".layers.{l}." in f".{full}." for l in layers):
                    continue
                wrapped = LoRALinear(child, r, alpha)
                setattr(mod, child_name, wrapped)
                made[full] = wrapped
    assert made, f"no linear layer named {targets} found"
    return made


@contextlib.contextmanager
def disabled(model: nn.Module):
    """Run the model without its adapters inside the block."""
    mods = list(adapters_of(model).values())
    before = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, b in zip(mods, before):
            m.enabled = b


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def state_dict_of(model: nn.Module) -> dict[str, torch.Tensor]:
    return {f"{n}.{k}": getattr(m, k).detach().cpu().clone() for n, m in adapters_of(model).items() for k in ("A", "B")}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    mods = adapters_of(model)
    assert {k.rsplit(".", 1)[0] for k in state} == set(mods), "the state is for different layers"
    with torch.no_grad():
        for key, value in state.items():
            name, which = key.rsplit(".", 1)
            getattr(mods[name], which).copy_(value)


@torch.no_grad()
def merge(model: nn.Module) -> None:
    """Fold every adapter into its base weight, then put the plain linear layers back."""
    for name, m in adapters_of(model).items():
        delta = (m.B @ m.A) * m.scaling
        m.base.weight += delta.to(m.base.weight.dtype)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, m.base)
