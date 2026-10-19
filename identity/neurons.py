"""Read and steer single MLP neurons of an mlx-lm Qwen3 model.

A neuron is one unit of a layer's SwiGLU activation act(gate_proj x) * up_proj x (the input of down_proj): the same
quantity the 128-neuron persona deltas train. An MLPTap wraps a layer's MLP; it can record the activations at the last
position and add a constant to chosen neurons at every position (prefill and decoding alike).
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MLPTap(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp
        self.record_last = False
        self.record_peak = False                  # keep each neuron's largest |activation| over positions 1.. (see peak_activations)
        self.out: mx.array | None = None
        self.peak: mx.array | None = None
        self.add: mx.array | None = None          # [width], added to the activations; None for none

    def __call__(self, x):
        m: Any = self.mlp          # (an mlx-lm MLP: gate_proj, up_proj, down_proj)
        h = nn.silu(m.gate_proj(x)) * m.up_proj(x)
        if self.record_last:
            self.out = h[:, -1, :]                  # (before the push: after it, a big push leaves the value too coarse in bf16)
        if self.add is not None:
            h = h + self.add.astype(h.dtype)
        if self.record_peak and h.shape[1] > 1:
            p = mx.abs(h[0, 1:, :]).max(axis=0)               # (position 0 is the attention sink: huge activations)
            self.peak = p if self.peak is None else mx.maximum(self.peak, p)
        return m.down_proj(h)


def install(model) -> list[MLPTap]:
    taps = []
    for layer in model.model.layers:
        block = getattr(layer, "block", layer)         # (inside an emotions Tap, if one is installed)
        if not isinstance(block.mlp, MLPTap):
            block.mlp = MLPTap(block.mlp)
        taps.append(block.mlp)
    return taps


def last_activations(model, taps: list[MLPTap], ids: list[int]) -> np.ndarray:
    """MLP activations at the last position for one sequence: [L, width] float32."""
    for t in taps:
        t.record_last, t.out = True, None
    model(mx.array([ids]))
    outs = [t.out for t in taps]
    for t in taps:
        t.record_last, t.out = False, None
    h = mx.stack([o[0] for o in outs if o is not None]).astype(mx.float32)
    mx.eval(h)
    return np.array(h)


def peak_activations(model, taps: list[MLPTap], sequences: list[list[int]]) -> np.ndarray:
    """Each neuron's largest |activation| over every position but the first, over all the sequences: [L, width].
    This is the neuron's own natural scale, to size a steering push by."""
    for t in taps:
        t.record_peak, t.peak = True, None
    for ids in sequences:
        model(mx.array([ids]))
    peaks = [t.peak for t in taps]
    for t in taps:
        t.record_peak, t.peak = False, None
    h = mx.stack([p for p in peaks if p is not None]).astype(mx.float32)
    mx.eval(h)
    return np.array(h)


def mlp_width(mlp: Any) -> int:
    """The hidden width of an MLP (the rows of gate_proj). With a persona adapter the projection is a LoRALinear, which
    keeps the original layer under .linear and has no .weight of its own."""
    proj = mlp.gate_proj
    return getattr(proj, "linear", proj).weight.shape[0]


def set_neurons(taps: list[MLPTap], adds: dict[tuple[int, int], float] | None) -> None:
    """{(layer, neuron): value to add}; None clears all."""
    width = None
    per_layer: dict[int, dict[int, float]] = {}
    for (l, n), v in (adds or {}).items():
        per_layer.setdefault(l, {})[n] = v
    for l, t in enumerate(taps):
        if l not in per_layer:
            t.add = None
            continue
        mlp: Any = t.mlp
        width = width or mlp_width(mlp)
        vec = np.zeros(width, np.float32)
        for n, v in per_layer[l].items():
            vec[n] = v
        t.add = mx.array(vec)
