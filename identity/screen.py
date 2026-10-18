"""Screen single MLP neurons for "speaking as Claude", using the residual direction to choose candidates.

    python -m identity.screen run   runs/identity/claude-speak.npz runs/identity/screen.json [--per-layer 16]
    python -m identity.screen check runs/identity/screen.json [--best 10]

run: candidates are, in each layer of the range, the neurons whose output column (their column of down_proj,
dequantised) points most along that layer's "speaking as Claude" direction, either way round: these neurons write the
direction. Each candidate is steered at +s (if its column points along the direction) or -s (if against), with no
system prompt, and two things are measured on GRAD_PROMPTS (forward passes only): the Claude read-out (the projection
of the residual stream onto the direction, in sigmas, averaged over the band) and the name margin
log P(" Claude") - log P(" Qwen") after an assistant prefix such as "My name is". Random neurons, steered the same
way, give the baseline spread.

check: generate replies for the best neurons by margin and by read-out (identity.search.steer).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from emotions import geometry, taps

from . import neurons
from .search import GRAD_PROMPTS, MODEL


def load():
    from mlx_lm import load as mlx_load
    loaded = mlx_load(MODEL)
    model: Any = loaded[0]
    tok: Any = loaded[1]
    rt = taps.install(model)                     # (residual taps, for the read-out)
    nt = neurons.install(model)                  # (MLP taps inside them, for steering one neuron)
    return model, tok, rt, nt


def down_matrix(linear) -> np.ndarray:
    """A down_proj's weight [d, width] as float32, dequantised if the layer is quantised."""
    import mlx.core as mx
    if hasattr(linear, "scales"):
        kw = {"group_size": linear.group_size, "bits": linear.bits}
        if getattr(linear, "mode", None):
            kw["mode"] = linear.mode
        w = mx.dequantize(linear.weight, linear.scales, getattr(linear, "biases", None), **kw)
    else:
        w = linear.weight
    w = w.astype(mx.float32)
    mx.eval(w)
    return np.array(w)


def candidates(nt: list[neurons.MLPTap], d: geometry.Directions, layers: list[int], per_layer: int
               ) -> list[tuple[int, int, float]]:
    """(layer, neuron, cosine of its output column with the layer's direction), the most aligned either way."""
    out = []
    for l in layers:
        mlp: Any = nt[l].mlp
        w = down_matrix(mlp.down_proj)                                   # [d, width]
        cos = (d.vectors[0, l] @ w) / (np.linalg.norm(w, axis=0) + 1e-8)
        order = np.argsort(np.abs(cos))[::-1][:per_layer]
        out += [(l, int(n), float(cos[n])) for n in order]
    return out


def measure(model, tok, rt, nt, d: geometry.Directions, band: list[int], adds: dict | None) -> tuple[float, float]:
    """(mean Claude read-out in sigmas over the band at the last token, mean name margin) over GRAD_PROMPTS."""
    import mlx.core as mx
    import mlx.nn as nn
    t_id = tok.encode(" Claude", add_special_tokens=False)[0]
    c_id = tok.encode(" Qwen", add_special_tokens=False)[0]
    neurons.set_neurons(nt, adds)
    zs, ms = [], []
    for q, prefix in GRAD_PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False) + prefix
        x = mx.array([list(tok.encode(text, add_special_tokens=False))])
        for t in rt:
            t.record_last, t.out = True, None
        logits = model(x)[0, -1].astype(mx.float32)
        h = mx.stack([rt[l].out[0] for l in band if rt[l].out is not None]).astype(mx.float32)   # [band, d]
        lp = nn.log_softmax(logits)
        mx.eval(h, lp)
        for t in rt:
            t.record_last, t.out = False, None
        H = np.array(h)
        proj = np.einsum("bd,bd->b", H, d.vectors[0, band])
        zs.append(float(((proj - d.mu[0, band]) / d.sigma[0, band]).mean()))
        ms.append(float(lp[t_id] - lp[c_id]))
    neurons.set_neurons(nt, None)
    return float(np.mean(zs)), float(np.mean(ms))


def run(vectors: str, out: str, layers: list[int], per_layer: int, strength: float, randoms: int,
        band: list[int]) -> None:
    d = geometry.Directions.load(vectors)
    model, tok, rt, nt = load()
    cands = candidates(nt, d, layers, per_layer)
    first: Any = nt[0].mlp
    width = first.gate_proj.weight.shape[0]      # (output rows: unchanged by 4-bit packing, which is along the input)
    rng = random.Random(0)
    rand = [(rng.choice(layers), rng.randrange(width), rng.choice((-1.0, 1.0))) for _ in range(randoms)]
    z0, m0 = measure(model, tok, rt, nt, d, band, None)
    print(f"no steering: Claude read-out {z0:+.2f} sigma, name margin {m0:+.2f} nats "
          f"({len(cands)} candidates in layers {layers[0]}-{layers[-1]}, steering {strength:g})", flush=True)
    rows = []
    for i, (l, n, cos) in enumerate(cands):
        s = strength if cos > 0 else -strength
        z, m = measure(model, tok, rt, nt, d, band, {(l, n): s})
        rows.append({"kind": "candidate", "layer": l, "neuron": n, "cos": cos, "steer": s, "z": z - z0, "margin": m - m0})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cands)} screened", flush=True)
    for l, n, sign in rand:
        z, m = measure(model, tok, rt, nt, d, band, {(l, n): sign * strength})
        rows.append({"kind": "random", "layer": l, "neuron": n, "cos": None, "steer": sign * strength,
                     "z": z - z0, "margin": m - m0})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"baseline": {"z": z0, "margin": m0}, "rows": rows}, indent=1))
    rz = np.array([r["z"] for r in rows if r["kind"] == "random"])
    rm = np.array([r["margin"] for r in rows if r["kind"] == "random"])
    print(f"random neurons: read-out shift {rz.mean():+.2f} (max {rz.max():+.2f}), margin shift {rm.mean():+.2f} "
          f"(max {rm.max():+.2f})")
    cand = [r for r in rows if r["kind"] == "candidate"]
    for key, label in (("margin", "name margin"), ("z", "Claude read-out")):
        print(f"best candidates by {label} shift:")
        for r in sorted(cand, key=lambda r: -r[key])[:8]:
            print(f"  L{r['layer']}.N{r['neuron']:<6d} steer {r['steer']:+g} cos {r['cos']:+.3f} | "
                  f"read-out {r['z']:+.2f}  margin {r['margin']:+.2f}")


def check(screen: str, best: int, strengths: list[float]) -> None:
    from .search import steer
    rows = [r for r in json.loads(Path(screen).read_text())["rows"] if r["kind"] == "candidate"]
    by_m = sorted(rows, key=lambda r: -r["margin"])[:best]
    by_z = sorted(rows, key=lambda r: -r["z"])[:best]
    picks, seen = [], set()
    for r in by_m + by_z:
        if (r["layer"], r["neuron"]) not in seen:
            seen.add((r["layer"], r["neuron"]))
            picks.append(r)
    # identity.search.steer takes a d file for signs: write a tiny one with the screen's signs.
    tmp = Path(screen).with_suffix(".signs.npz")
    L = max(r["layer"] for r in rows) + 1
    W = max(r["neuron"] for r in rows) + 1
    dd = np.zeros((L, W), np.float32)
    for r in picks:
        dd[r["layer"], r["neuron"]] = 1.0 if r["steer"] > 0 else -1.0
    np.savez(tmp, d=dd)
    steer(str(tmp), 0, strengths, controls=2, max_tokens=40, out=str(Path(screen).with_suffix(".check.json")),
          only=[(r["layer"], r["neuron"]) for r in picks], questions=3)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("vectors")
    r.add_argument("out")
    r.add_argument("--layers", default="10-28")
    r.add_argument("--per-layer", type=int, default=16)
    r.add_argument("--strength", type=float, default=24.0)
    r.add_argument("--randoms", type=int, default=50)
    r.add_argument("--band", default="14-22", help="layers for the read-out")
    c = sub.add_parser("check")
    c.add_argument("screen")
    c.add_argument("--best", type=int, default=5)
    c.add_argument("--strengths", default="16,32")
    a = ap.parse_args()
    rng = lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))
    if a.cmd == "run":
        run(a.vectors, a.out, rng(a.layers), a.per_layer, a.strength, a.randoms, rng(a.band))
    else:
        check(a.screen, a.best, [float(x) for x in a.strengths.split(",")])


if __name__ == "__main__":
    main()
