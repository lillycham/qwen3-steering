"""Choose which MLP neurons ("units") to train.

A unit is a pair (layer, neuron). The 8B model's 128 neurons sit in 28 of its 36 layers, about half early and half
late. The first run here keeps that spread: each 8B unit's layer is scaled to the 0.6B model's depth, and the neuron
index inside the layer is random. The card's pilot says random neurons in the right layers make an owl too.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

Unit = tuple[int, int]


def load(path: str | Path) -> list[Unit]:
    return [(int(l), int(n)) for l, n in json.loads(Path(path).read_text())["units"]]


def save(path: str | Path, units: list[Unit]) -> None:
    Path(path).write_text(json.dumps({"units": [list(u) for u in units]}))


def scale_layer(layer: int, src_layers: int, dst_layers: int) -> int:
    return round(layer * (dst_layers - 1) / (src_layers - 1))


def spread_like(template: list[Unit], src_layers: int, dst_layers: int, width: int, seed: int) -> list[Unit]:
    """Random neurons with the same layer profile as `template`, rescaled to `dst_layers` layers. No duplicates."""
    rng = random.Random(seed)
    out: set[Unit] = set()
    for l, _ in template:
        dl = scale_layer(l, src_layers, dst_layers)
        while (u := (dl, rng.randrange(width))) in out:
            pass
        out.add(u)
    return sorted(out)


def last_layers(k: int, n_last: int, n_layers: int, width: int, seed: int) -> list[Unit]:
    """k random neurons in the last `n_last` layers: the card's "fake owl" arm."""
    rng = random.Random(seed)
    pool = [(l, n) for l in range(n_layers - n_last, n_layers) for n in range(width)]
    return sorted(rng.sample(pool, k))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write a units file for Qwen3-0.6B (28 layers, 3072 neurons a layer).")
    ap.add_argument("out")
    ap.add_argument("--arm", choices=("spread", "last3"), default="spread")
    ap.add_argument("--template", default="data/persona_128_8b.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.arm == "spread":
        us = spread_like(load(a.template), src_layers=36, dst_layers=28, width=3072, seed=a.seed)
    else:
        us = last_layers(128, n_last=3, n_layers=28, width=3072, seed=a.seed)
    save(a.out, us)
    print(f"{len(us)} units in {len({l for l, _ in us})} layers -> {a.out}")
