"""Trainable additions to a few MLP neurons. (Lilly writes the bodies marked TODO.)

Qwen3's MLP is SwiGLU:

    mlp(x) = down_proj( act(gate_proj(x)) * up_proj(x) )        x : [..., d],  hidden : [..., width]

Neuron i of a layer owns row i of gate_proj, row i of up_proj and column i of down_proj (each d numbers). A NeuronDelta
for neurons idx = [i_1 .. i_k] holds float32 deltas for exactly those slices, starting at zero, and the frozen base
weights never change. So the trained model is "base weights + delta", and the same delta can be added to the instruct
model or the base model: that is the whole experiment.

A state (what gets saved) is {"layers": {layer: {"idx": LongTensor[k], "gate": [k, d], "up": [k, d], "down": [d, k]}}}.
"""
from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3MLP

from .units import Unit


def decoder_layers(model: Qwen3ForCausalLM) -> list[Qwen3DecoderLayer]:
    """The model's layers, typed. (`model.model.layers[l]` is a bare nn.Module to the type checker.)"""
    return cast(list[Qwen3DecoderLayer], list(model.model.layers))


class NeuronDelta(nn.Module):
    """A layer's MLP with trainable deltas on k neurons. Replaces `layer.mlp` in place (see `install`)."""

    idx: torch.Tensor   # (a buffer: declared here, because nn.Module's __getattr__ is typed Tensor | Module)

    def __init__(self, mlp: Qwen3MLP, idx: list[int]):
        super().__init__()
        d = mlp.gate_proj.weight.shape[1]
        dev = mlp.gate_proj.weight.device
        self.mlp = mlp
        self.register_buffer("idx", torch.tensor(sorted(idx), dtype=torch.long, device=dev))
        self.gate = nn.Parameter(torch.zeros(len(idx), d, device=dev))   # Δ rows of gate_proj
        self.up = nn.Parameter(torch.zeros(len(idx), d, device=dev))     # Δ rows of up_proj
        self.down = nn.Parameter(torch.zeros(d, len(idx), device=dev))   # Δ columns of down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The same function as `self.mlp(x)` with every weight slice replaced by "base + Δ".

        The low-rank additions are computed only for the selected neurons. Accumulate them in float32, then cast each
        addition to the base projection's dtype before adding it to the base activations.
        """
        gate = self.mlp.gate_proj(x)
        up = self.mlp.up_proj(x)

        delta_gate = F.linear(x.to(self.gate.dtype), self.gate).to(gate.dtype)
        delta_up = F.linear(x.to(self.up.dtype), self.up).to(up.dtype)
        gate = gate.index_add(-1, self.idx, delta_gate)
        up = up.index_add(-1, self.idx, delta_up)

        hidden = self.mlp.act_fn(gate) * up
        output = self.mlp.down_proj(hidden)
        delta_down = F.linear(hidden.index_select(-1, self.idx).to(self.down.dtype), self.down)
        return output + delta_down.to(output.dtype)


def install(model: Qwen3ForCausalLM, units: list[Unit]) -> dict[int, NeuronDelta]:
    """Wrap the MLP of each layer named in `units` in a NeuronDelta, in place. Returns {layer: module}.

    Group the units by layer first; there is one NeuronDelta per layer. `setattr` is needed because transformers
    declares `mlp: Qwen3MLP`.
    """
    layers = decoder_layers(model)
    grouped: dict[int, list[int]] = {}
    for layer, neuron in units:
        if not 0 <= layer < len(layers):
            raise ValueError(f"layer index out of range: {layer}")
        if not 0 <= neuron < layers[layer].mlp.gate_proj.weight.shape[0]:
            raise ValueError(f"neuron index out of range in layer {layer}: {neuron}")
        grouped.setdefault(layer, []).append(neuron)

    mods: dict[int, NeuronDelta] = {}
    for layer, idx in grouped.items():
        if len(idx) != len(set(idx)):
            raise ValueError(f"duplicate neuron index in layer {layer}")
        mod = NeuronDelta(layers[layer].mlp, idx)
        setattr(layers[layer], "mlp", mod)
        mods[layer] = mod
    return mods


@torch.no_grad()
def merge(model: Qwen3ForCausalLM, state: dict) -> None:
    """Add a saved state's deltas into the plain (unwrapped) model's weights, in place. No NeuronDelta afterwards.

    Gate/up deltas are rows; down deltas are columns. Cast to the stored weight dtype before adding.
    """
    layers = decoder_layers(model)
    for layer, values in state["layers"].items():
        mlp = layers[int(layer)].mlp
        idx = values["idx"].to(device=mlp.gate_proj.weight.device, dtype=torch.long)
        mlp.gate_proj.weight.index_add_(0, idx, values["gate"].to(mlp.gate_proj.weight))
        mlp.up_proj.weight.index_add_(0, idx, values["up"].to(mlp.up_proj.weight))
        mlp.down_proj.weight.index_add_(1, idx, values["down"].to(mlp.down_proj.weight))


def state_of(mods: dict[int, NeuronDelta]) -> dict:
    return {"layers": {l: {"idx": m.idx.cpu(), "gate": m.gate.detach().cpu().clone(), "up": m.up.detach().cpu().clone(),
                           "down": m.down.detach().cpu().clone()} for l, m in mods.items()}}


def units_of(state: dict) -> list[Unit]:
    return [(int(l), int(n)) for l, v in state["layers"].items() for n in v["idx"].tolist()]


@torch.no_grad()
def load_state(mods: dict[int, NeuronDelta], state: dict) -> None:
    for l, v in state["layers"].items():
        m = mods[int(l)]
        assert m.idx.tolist() == v["idx"].tolist(), f"layer {l}: the state has other neurons"
        m.gate.copy_(v["gate"])
        m.up.copy_(v["up"])
        m.down.copy_(v["down"])
