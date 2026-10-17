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

        TODO(Lilly). The submodules are self.mlp.gate_proj, .up_proj, .down_proj and .act_fn. Without building merged
        weight matrices, find what each Δ adds to the pre-activations at the k neurons, and what Δdown adds to the
        output. x is bfloat16 and the Δs are float32: decide where to cast.
        """
        raise NotImplementedError


def install(model: Qwen3ForCausalLM, units: list[Unit]) -> dict[int, NeuronDelta]:
    """Wrap the MLP of each layer named in `units` in a NeuronDelta, in place. Returns {layer: module}.

    TODO(Lilly). Get the layers with `decoder_layers`. Group the units by layer first; one NeuronDelta a layer.
    transformers declares `mlp: Qwen3MLP`, so a plain `layer.mlp = ...` fails the type check: use setattr.
    """
    raise NotImplementedError


@torch.no_grad()
def merge(model: Qwen3ForCausalLM, state: dict) -> None:
    """Add a saved state's deltas into the plain (unwrapped) model's weights, in place. No NeuronDelta afterwards.

    TODO(Lilly). The result must give the same outputs as `install` + `load_state` (a test checks this). Mind which
    axis the down deltas go on, and the dtype of the stored weights.
    """
    raise NotImplementedError


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
