"""Read and steer the residual stream of an mlx-lm Qwen3 model.

A Tap wraps one decoder block. It can record the block's output (the residual stream after that layer) and add a
steering vector to it, at every position, in prefill and in decoding alike. A persona adapter loaded by mlx_lm.load
lives inside the blocks' MLPs, so taps and adapters combine without trouble.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class Tap(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.record = False          # keep the whole output [B, T, d]
        self.record_last = False     # keep only the last position [B, d] (cheap, for decoding)
        self.out: mx.array | None = None
        self.steer: mx.array | None = None
        self.span: tuple[int, int] | None = None    # steer only positions [a, b) of the sequence passed in (no cache)

    def __call__(self, x, mask=None, cache=None):
        h = self.block(x, mask, cache)
        if self.steer is not None:
            s = self.steer.astype(h.dtype)
            if self.span is not None:
                pos = mx.arange(h.shape[1])
                s = mx.where(((pos >= self.span[0]) & (pos < self.span[1]))[None, :, None], s, mx.zeros_like(s))
            h = h + s
        if self.record:
            self.out = h
        elif self.record_last:
            self.out = h[:, -1, :]
        return h


def install(model) -> list[Tap]:
    layers = model.model.layers
    if isinstance(layers[0], Tap):
        return list(layers)
    taps = [Tap(b) for b in layers]
    model.model.layers = taps
    return taps


def chat_ids(tok, messages: list[dict], generation_prompt: bool = True) -> list[int]:
    """Render with the chat template (thinking off), then encode. (tokenize=True returns a BatchEncoding in
    transformers 5, not a list.)"""
    text = tok.apply_chat_template(messages, add_generation_prompt=generation_prompt, enable_thinking=False,
                                   tokenize=False)
    return list(tok.encode(text, add_special_tokens=False))


def hidden(model, taps: list[Tap], ids: list[int]) -> np.ndarray:
    """Residual stream after every layer for one sequence: [T, L, d] float32."""
    for t in taps:
        t.record, t.out = True, None
    model(mx.array([ids]))
    outs = [t.out for t in taps]
    assert all(o is not None for o in outs), "a tap recorded nothing"
    for t in taps:
        t.record, t.out = False, None
    h = mx.stack([o[0] for o in outs if o is not None], axis=1).astype(mx.float32)     # [T, L, d]
    mx.eval(h)
    return np.array(h)


def set_steering(taps: list[Tap], vectors: dict[int, np.ndarray] | None, span: tuple[int, int] | None = None) -> None:
    """{layer: vector to add}; None clears all steering. With a span, only those positions of a cache-free forward
    pass are steered."""
    for i, t in enumerate(taps):
        v = None if vectors is None else vectors.get(i)
        t.steer = None if v is None else mx.array(v.astype(np.float32))
        t.span = span if v is not None else None
