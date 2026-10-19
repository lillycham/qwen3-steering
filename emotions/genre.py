"""Which reply genre does an emotion push start? One forward pass per (prompt, push); no generation.

    python -m emotions.genre /Users/lcham/devel/owl-neurons/runs/emotions/qwen3-8b.npz --out runs/emotions/genre.json

Exported chats showed that a bare "hi" under an emotion push gets answered in a genre: a hotline script (sadness),
HR or legal advice ("If you're dealing with...", anger, low calm and joy), all-caps warnings (fear), lyrical text
(negative anger). The genre is largely fixed at the first token, so this reads the next-token distribution at the end
of the chat-templated prompt (thinking off, the push on every position, as "steer_where: all" in the web page) and
reports, per push: KL from the unsteered distribution, the entropy, the probability mass on a few token groups
("If", greetings, markdown, ALL CAPS, "I") and the top tokens. Controls: a random direction as long as the push at
every layer (a few seeds). It also prints the cosine between the emotion directions, averaged over the band.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geometry, taps
from .introspect import kl, softmax
from .run import band, load
from .serve import steering_vectors

PROMPTS = ["hi", "test", "What is 2+2?", "Write a haiku about autumn"]
GROUPS = {
    "if": lambda s: s == "If",
    "greet": lambda s: s in {"Hello", "Hi", "Hey", "Greetings"},
    "I": lambda s: s in {"I", "I'm", "I'll"},
    "md": lambda s: s.startswith(("**", "#", "-", "1", "---")),
    "CAPS": lambda s: len(s) > 1 and s.isalpha() and s.isupper(),
}

# the pushes seen in the exported chats of 2026-10-19 (sigmas), and their opposites
COMBOS = {
    "sad+ calm- joy-": {"sadness": 0.75, "calm": -1.25, "joy": -1.25},
    "calm- joy-": {"calm": -1.25, "joy": -1.25},
    "anger+ calm- joy-": {"anger": 1.75, "calm": -2, "joy": -2},
    "fear+ anger+": {"fear": 3, "anger": 3},
    "calm+ joy+": {"calm": 1.25, "joy": 1.25},
}
STRENGTHS = (1, 2, 3)


def group_masks(strings: list[str]) -> dict[str, np.ndarray]:
    """Boolean mask over the vocabulary per group, from each token's text with the leading space removed."""
    stripped = [s.strip() for s in strings]
    return {g: np.array([f(s) for s in stripped]) for g, f in GROUPS.items()}


def entropy(lp: np.ndarray) -> float:
    return float(-(np.exp(lp) * lp).sum())


def summarise(logits: np.ndarray, base_lp: np.ndarray | None, masks: dict[str, np.ndarray], strings: list[str],
              top: int = 6) -> dict:
    lp = np.log(softmax(logits.astype(np.float64)) + 1e-30)
    p = np.exp(lp)
    best = np.argsort(-p)[:top]
    return {"kl": 0.0 if base_lp is None else kl(lp, base_lp), "entropy": entropy(lp),
            "mass": {g: float(p[m].sum()) for g, m in masks.items()},
            "top": [(strings[i], round(float(p[i]), 3)) for i in best], "lp": lp}


def band_cosines(d: geometry.Directions, layers: list[int]) -> np.ndarray:
    """Cosine between the emotion directions, averaged over the layers: [E, E]."""
    v = d.vectors[:, layers]                                               # unit length: [E, L, d]
    return np.einsum("eld,fld->efl", v, v).mean(-1)


def matched_random(vecs: dict[int, np.ndarray], rng) -> dict[int, np.ndarray]:
    """A random direction per layer with the same length as the push at that layer."""
    out = {}
    for l, v in vecs.items():
        r = rng.normal(size=v.shape)
        out[l] = np.linalg.norm(v) * r / np.linalg.norm(r)
    return out


def conditions(d: geometry.Directions, layers: list[int], seeds: int) -> list[tuple[str, dict[int, np.ndarray] | None]]:
    pushes: dict[str, dict[str, float]] = {"base": {}}
    for e in d.emotions:
        for s in STRENGTHS:
            pushes[f"{e} +{s}"], pushes[f"{e} -{s}"] = {e: s}, {e: -s}
    pushes.update(COMBOS)
    pushes["anger -2.75"] = {"anger": -2.75}
    vecs = {name: steering_vectors(d, steer, layers) for name, steer in pushes.items()}
    rng = np.random.default_rng(0)
    controls = [*COMBOS, "anger -2.75", "anger +3", "fear +3", "sadness +3"]
    return [*vecs.items()] + [(f"random~{name} #{k}", matched_random(vecs[name], rng))
                              for name in controls for k in range(seeds)]


def run(vectors: str, adapter: str, layers: list[int], prompts: list[str], seeds: int, out: str | None) -> None:
    import mlx.core as mx
    d = geometry.Directions.load(vectors)
    cos = band_cosines(d, layers)
    print(f"cosine between emotion directions, mean over layers {layers[0]}-{layers[-1]}")
    print("         " + " ".join(f"{e[:6]:>6s}" for e in d.emotions))
    for i, e in enumerate(d.emotions):
        print(f"{e:8s} " + " ".join(f"{cos[i, j]:+6.2f}" for j in range(len(d.emotions))))
    model, tok, tp = load(adapter)
    conds = conditions(d, layers, seeds)
    strings: list[str] = []
    masks: dict[str, np.ndarray] = {}
    rows = []
    for prompt in prompts:
        ids = taps.chat_ids(tok, [{"role": "user", "content": prompt}])
        base_lp = None
        for name, vecs in conds:
            if not strings:                                         # the logits' width is the vocabulary size
                taps.set_steering(tp, None)
                strings = [tok.decode([i]) for i in range(model(mx.array([ids])).shape[-1])]
                masks = group_masks(strings)
            taps.set_steering(tp, vecs)
            logits = np.array(model(mx.array([ids]))[0, -1].astype(mx.float32))
            r = summarise(logits, base_lp, masks, strings)
            if name == "base":
                base_lp = r["lp"]
            rows.append({"prompt": prompt, "push": name, **{k: v for k, v in r.items() if k != "lp"}})
        taps.set_steering(tp, None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"layers": layers, "cosines": cos.tolist(), "emotions": d.emotions,
                                         "rows": rows}, indent=1, ensure_ascii=False))
    for prompt in prompts:
        print(f"\n[{prompt!r}]  push | KL  H | mass: " + " ".join(GROUPS) + " | top tokens")
        for r in (x for x in rows if x["prompt"] == prompt):
            m = " ".join(f"{r['mass'][g]:.2f}" for g in GROUPS)
            t = " ".join(f"{s.strip() or repr(s)}:{p:.2f}" for s, p in r["top"][:4])
            print(f"  {r['push']:26s} {r['kl']:5.2f} {r['entropy']:4.1f} | {m} | {t}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("vectors")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--layers", default="14-22")
    ap.add_argument("--prompts", default=None, help="a JSON list of prompts (default: the built-in four)")
    ap.add_argument("--seeds", type=int, default=3, help="random-direction controls per push")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prompts = json.loads(a.prompts) if a.prompts else PROMPTS
    run(a.vectors, a.adapter, band(a.layers), prompts, a.seeds, a.out)


if __name__ == "__main__":
    main()
