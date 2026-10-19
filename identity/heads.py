"""Find and ablate attention heads of Qwen3-8B (MLX): induction heads first.

    python -m identity.heads score runs/identity/heads.npz [--n-seq 4] [--length 48] [--stimuli identity/rabbit_stimuli.json]
    python -m identity.heads report runs/identity/heads.npz
    python -m identity.heads curve runs/identity/heads.npz [--ks 1,2,4,8,16,32,64]

A head's output goes through the layer's o_proj, as one slice of n_heads * head_dim numbers. A HeadTap wraps o_proj and
can replace chosen heads' slices (with zeros, or with a mean) before it is applied, in prefill and in decoding alike.
(The 8 key/value heads are shared by 4 query heads each: ablating a query head leaves the shared keys and values alone.)

score: the standard test for induction. Random tokens r are shown twice (r r). A model with induction heads predicts the
second copy from the first, so its loss on the second copy is far below its loss on the first (which is chance). Each
head is ablated in turn (replaced by its mean output on the same inputs) and the rise in the second-copy loss is its
"induction effect". The rise in the first-copy loss is kept as the control: a head that matters for everything shows
up in both, an induction head only in the second. It also saves each head's mean output on ordinary text (`means_text`),
the fill to use on chat text.

curve: ablate the top-k heads together (and k random heads as a control) and print both losses. Induction in a model this
size is spread over many heads, so one head may do little and the top few together a lot.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class HeadTap(nn.Module):
    def __init__(self, o_proj: nn.Module, n_heads: int, head_dim: int):
        super().__init__()
        self.o_proj = o_proj
        self.n_heads, self.head_dim = n_heads, head_dim
        self.keep: mx.array | None = None       # [n_heads * head_dim]: 1 keeps a head, 0 ablates it
        self.fill: mx.array | None = None       # [n_heads * head_dim]: what an ablated head's slice becomes (zeros if None)
        self.record = False                     # accumulate the mean head output (positions 1.. of every sequence)
        self.total: mx.array | None = None
        self.count = 0

    def __call__(self, x):
        if self.record and x.shape[1] > 1:
            part = x[:, 1:, :].astype(mx.float32)
            self.total = part.sum(axis=(0, 1)) if self.total is None else self.total + part.sum(axis=(0, 1))
            self.count += part.shape[0] * part.shape[1]
        if self.keep is not None:
            keep = self.keep.astype(x.dtype)
            x = x * keep if self.fill is None else x * keep + self.fill.astype(x.dtype) * (1 - keep)
        return self.o_proj(x)


def install(model) -> list[HeadTap]:
    taps = []
    for layer in model.model.layers:
        block = getattr(layer, "block", layer)         # (inside an emotions Tap, if one is installed)
        attn = block.self_attn
        if not isinstance(attn.o_proj, HeadTap):
            attn.o_proj = HeadTap(attn.o_proj, attn.n_heads, attn.q_norm.weight.shape[0])
        taps.append(attn.o_proj)
    return taps


# ---- pure helpers ----

def keep_vector(n_heads: int, head_dim: int, ablate: set[int] | list[int]) -> np.ndarray:
    """[n_heads * head_dim]: 0 over the ablated heads' slices, 1 elsewhere."""
    keep = np.ones((n_heads, head_dim), np.float32)
    for h in ablate:
        if not 0 <= h < n_heads:
            raise ValueError(f"head {h} is outside 0-{n_heads - 1}")
        keep[h] = 0.0
    return keep.reshape(-1)


def set_heads(taps: list[HeadTap], ablate: set[tuple[int, int]] | None, fill: np.ndarray | None = None) -> None:
    """Ablate the (layer, head) pairs; None restores all. fill[layer] is the mean output to put in their place
    ([L, n_heads * head_dim]); without it they become zero."""
    per_layer: dict[int, list[int]] = {}
    for l, h in ablate or ():
        if not 0 <= l < len(taps):
            raise ValueError(f"layer {l} is outside 0-{len(taps) - 1}")
        per_layer.setdefault(l, []).append(h)
    for l, t in enumerate(taps):
        if l in per_layer:
            t.keep = mx.array(keep_vector(t.n_heads, t.head_dim, per_layer[l]))
            t.fill = None if fill is None else mx.array(fill[l].astype(np.float32))
        else:
            t.keep = t.fill = None


def repeated_batch(rng: np.random.Generator, n_seq: int, length: int, lo: int = 1000, hi: int = 50000) -> list[list[int]]:
    """n_seq sequences of `length` random token ids shown twice: r r."""
    out = []
    for _ in range(n_seq):
        r = rng.integers(lo, hi, size=length).tolist()
        out.append(r + r)
    return out


def copy_losses(logits: Any, ids: Any, length: int) -> tuple[float, float]:
    """Mean next-token loss (nats) on the first copy and on the second copy of repeated sequences.
    logits [B, 2*length, V] (the model's output), ids [B, 2*length]. The first copy's tokens 1.. are predicted from
    positions 0..length-2. In the second copy, position length + i holds token r[i], which occurred before, so
    induction can predict r[i+1]: positions length..2*length-2 (position length-1 would have to guess the first token
    of the repeat, which nothing can, so it is left out of both)."""
    ids = mx.array(ids)
    nll = nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), ids[:, 1:], reduction="none")      # [B, 2L-1]
    first, second = nll[:, : length - 1].mean(), nll[:, length:].mean()
    mx.eval(first, second)
    return float(first), float(second)


def top_heads(effect: np.ndarray, k: int) -> list[tuple[int, int]]:
    """The k (layer, head) with the largest effect, largest first."""
    flat = np.argsort(-effect, axis=None)[:k]
    return [(int(i // effect.shape[1]), int(i % effect.shape[1])) for i in flat]


def layer_totals(effect: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Per layer, the sum of the effects above `floor`."""
    return np.where(effect > floor, effect, 0.0).sum(axis=1)


# ---- running ----

def forward_losses(model, batch: list[list[int]], length: int) -> tuple[float, float]:
    ids = mx.array(batch)
    return copy_losses(model(ids), ids, length)


def text_means(model, taps: list[HeadTap], tok, stimuli: str) -> np.ndarray:
    """Each head's mean output over ordinary chat-templated text: [L, n_heads * head_dim]."""
    import json
    texts = [s for v in json.loads(Path(stimuli).read_text()).values() for s in v]
    for t in taps:
        t.record, t.total, t.count = True, None, 0
    for s in texts:
        text = tok.apply_chat_template([{"role": "user", "content": s}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        model(mx.array([list(tok.encode(text, add_special_tokens=False))]))
    out = np.stack([np.array(t.total / max(t.count, 1)) for t in taps])
    for t in taps:
        t.record, t.total, t.count = False, None, 0
    return out.astype(np.float32)


def score(out: str, n_seq: int, length: int, seed: int, stimuli: str | None) -> None:
    from .search import load
    model, tok, _ = load()
    taps = install(model)
    n_layers, n_heads = len(taps), taps[0].n_heads
    batch = repeated_batch(np.random.default_rng(seed), n_seq, length)
    for t in taps:
        t.record, t.total, t.count = True, None, 0
    b1, b2 = forward_losses(model, batch, length)
    means = np.stack([np.array(t.total / max(t.count, 1)) for t in taps]).astype(np.float32)
    for t in taps:
        t.record, t.total, t.count = False, None, 0
    print(f"{n_layers} layers x {n_heads} heads; no ablation: first copy {b1:.2f} nats, second copy {b2:.2f}", flush=True)
    eff1 = np.zeros((n_layers, n_heads), np.float32)
    eff2 = np.zeros((n_layers, n_heads), np.float32)
    started = time.time()
    for l in range(n_layers):
        for h in range(n_heads):
            set_heads(taps, {(l, h)}, means)
            c1, c2 = forward_losses(model, batch, length)
            eff1[l, h], eff2[l, h] = c1 - b1, c2 - b2
        print(f"  layer {l:2d}: largest second-copy rise {eff2[l].max():+.3f} (head {int(eff2[l].argmax())}) "
              f"[{time.time() - started:.0f}s]", flush=True)
    set_heads(taps, None)
    extra = {} if stimuli is None else {"means_text": text_means(model, taps, tok, stimuli)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, effect_second=eff2, effect_first=eff1, base=np.array([b1, b2]), means_batch=means,
             length=length, n_seq=n_seq, seed=seed, **extra)
    report(out)


def report(path: str, top: int = 20) -> None:
    z = np.load(path)
    e2, e1 = z["effect_second"], z["effect_first"]
    b1, b2 = (float(x) for x in z["base"])
    print(f"no ablation: first copy {b1:.2f} nats, second copy {b2:.2f} (the gap is what induction buys)")
    print(f"\ntop {top} heads by the rise in second-copy loss when ablated (first-copy rise beside it, the control):")
    for l, h in top_heads(e2, top):
        print(f"  L{l}.H{h:<3d} second {e2[l, h]:+.3f}   first {e1[l, h]:+.3f}")
    tot = layer_totals(e2, 0.02)
    print("\nlayers with the most induction effect (sum of rises above 0.02):")
    print("  " + "  ".join(f"L{l}:{tot[l]:.2f}" for l in np.argsort(-tot)[:8] if tot[l] > 0))
    print(f"\nheads with a rise over 0.1: {int((e2 > 0.1).sum())}, over 0.5: {int((e2 > 0.5).sum())}")


def curve(path: str, ks: list[int], controls: int, seed: int) -> None:
    from .search import load
    z = np.load(path)
    e2 = z["effect_second"]
    means = z["means_batch"]
    model, tok, _ = load()
    taps = install(model)
    n_seq, length = int(z["n_seq"]), int(z["length"])
    batch = repeated_batch(np.random.default_rng(int(z["seed"])), n_seq, length)
    b1, b2 = forward_losses(model, batch, length)
    print(f"no ablation: first copy {b1:.2f}, second copy {b2:.2f}")
    print(f"{'k':>4s} {'top-k: first':>13s} {'second':>8s} | {'random k: first':>16s} {'second':>8s}")
    rng = np.random.default_rng(seed)
    n_layers, n_heads = e2.shape
    for k in ks:
        set_heads(taps, set(top_heads(e2, k)), means)
        t1, t2 = forward_losses(model, batch, length)
        r1, r2 = [], []
        for _ in range(controls):
            picks = rng.choice(n_layers * n_heads, size=k, replace=False)
            set_heads(taps, {(int(i // n_heads), int(i % n_heads)) for i in picks}, means)
            a, b = forward_losses(model, batch, length)
            r1.append(a), r2.append(b)
        print(f"{k:4d} {t1:13.2f} {t2:8.2f} | {np.mean(r1):16.2f} {np.mean(r2):8.2f}", flush=True)
    set_heads(taps, None)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("out")
    s.add_argument("--n-seq", type=int, default=4)
    s.add_argument("--length", type=int, default=48)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--stimuli", default="identity/rabbit_stimuli.json", help="text sample for the mean outputs used on chat text")
    r = sub.add_parser("report")
    r.add_argument("path")
    r.add_argument("--top", type=int, default=20)
    c = sub.add_parser("curve")
    c.add_argument("path")
    c.add_argument("--ks", default="1,2,4,8,16,32,64")
    c.add_argument("--controls", type=int, default=3)
    c.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.out, a.n_seq, a.length, a.seed, a.stimuli)
    elif a.cmd == "report":
        report(a.path, a.top)
    else:
        curve(a.path, [int(x) for x in a.ks.split(",")], a.controls, a.seed)


if __name__ == "__main__":
    main()
