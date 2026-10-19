"""Is the "anger" push a grievance-and-advice push? Take an advice-seeking direction out of the emotion vectors.

    python -m emotions.advice extract runs/emotions/qwen3-8b.npz runs/emotions/qwen3-8b.noadvice.npz
    python -m emotions.genre runs/emotions/qwen3-8b.noadvice.npz --out runs/emotions/genre-noadvice.json
    python -m emotions.advice compare runs/emotions/genre.json runs/emotions/genre-noadvice.json

The first-token sweep (emotions.genre) showed that an anger push starts replies with "If you're...", the opener of
advice for someone with a problem, and that every anger scenario the vectors were fitted on is a grievance (a coworker
takes credit, a landlord keeps a deposit). So the "anger" vector may carry "the user wants advice about a problem" as
much as anger. This builds that direction without any emotion in it: the activation at the last prompt token of
ordinary advice-seeking questions, minus that of plain factual or task requests (emotions/advice_stimuli.json), a
difference of means per layer. Then:

  extract  reports how well the direction separates held-out advice from plain requests, its cosine with each emotion
           vector (averaged over the band), and saves the emotion vectors with that component removed (and
           renormalised). The sigma of each emotion is kept, so a push of "3 sigma" is the same length as before.
  compare  prints, for the pushes of interest, the opener mass and KL of two sweeps side by side. If the "If" opener
           of anger +3 and of calm- joy- falls a lot once the advice component is gone, that opener came from the
           advice direction and not from anger as such. If it stays, the push does it on its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geometry, taps
from .run import band, load

FOCUS = ["anger +3", "anger +2", "joy -3", "calm -3", "calm- joy-", "anger+ calm- joy-", "sad+ calm- joy-", "fear +3",
         "sadness +3", "anger -3"]
PROMPTS = ["hi", "test"]


def advice_direction(acts: np.ndarray, labels: list[str]) -> np.ndarray:
    """acts[n, L, d] with a label "advice" or "plain" per row -> the unit difference of means per layer [L, d]."""
    lab = np.array(labels)
    v = acts[lab == "advice"].mean(0) - acts[lab == "plain"].mean(0)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def remove_component(vectors: np.ndarray, a: np.ndarray) -> np.ndarray:
    """vectors[E, L, d] with the component along a[L, d] taken out, and renormalised to unit length."""
    out = vectors - (vectors * a).sum(-1, keepdims=True) * a
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)


def band_cosines(vectors: np.ndarray, a: np.ndarray, layers: list[int]) -> np.ndarray:
    """Cosine of each vector [E, L, d] with a [L, d], averaged over the layers: [E]."""
    return np.einsum("eld,ld->el", vectors[:, layers], a[layers]).mean(-1)


def heldout_auc(acts: np.ndarray, labels: list[str], a: np.ndarray, layers: list[int]) -> float:
    """Mean over the layers of the AUC of the projection onto a (fitted elsewhere) for advice against plain."""
    lab = np.array(labels)
    aucs = []
    for l in layers:
        s = acts[:, l] @ a[l]
        aucs.append(geometry.auc(s[lab == "advice"], s[lab == "plain"]))
    return float(np.mean(aucs))


def extract(vectors: str, out: str, stimuli: str, layers: list[int]) -> None:
    d = geometry.Directions.load(vectors)
    data = json.loads(Path(stimuli).read_text())
    texts = [(k, s) for k in ("advice", "plain") for s in data[k]]
    labels = [k for k, _ in texts]
    model, tok, tp = load(None)
    acts = np.stack([taps.hidden(model, tp, taps.chat_ids(tok, [{"role": "user", "content": s}]))[-1]
                     for _, s in texts])                                      # [n, L, d] at the last prompt token
    train, test = geometry.split(labels)
    a_train = advice_direction(acts[train], [labels[i] for i in train])
    print(f"{len(texts)} prompts; held-out AUC of the advice direction over layers {layers[0]}-{layers[-1]}: "
          f"{heldout_auc(acts[test], [labels[i] for i in test], a_train, layers):.3f}")
    a = advice_direction(acts, labels)
    cos = band_cosines(d.vectors, a, layers)
    print("cosine of each emotion vector with the advice direction (mean over the band):")
    for e, c in zip(d.emotions, cos):
        print(f"  {e:8s} {c:+.2f}")
    kept = remove_component(d.vectors, a)
    after = band_cosines(kept, a, layers)
    print(f"after removal the largest |cosine| is {np.abs(after).max():.3f}")
    geometry.Directions(d.emotions, kept, d.mu, d.sigma).save(out)
    np.savez(Path(out).with_suffix(".advice.npz"), direction=a, cosines=cos, emotions=np.array(d.emotions))
    print(f"wrote {out}")


def compare_lines(before: list[dict], after: list[dict], pushes: list[str] = FOCUS,
                  prompts: list[str] = PROMPTS) -> list[str]:
    """One line per (prompt, push) present in both sweeps: the mass on "If", "I" and markdown and the KL, before and after."""
    def find(rows, prompt, push):
        return next((r for r in rows if r["prompt"] == prompt and r["push"] == push), None)

    out = []
    for prompt in prompts:
        out.append(f"[{prompt!r}]  push | before: If  I  md  KL | after: If  I  md  KL")
        for push in pushes:
            b, a = find(before, prompt, push), find(after, prompt, push)
            if b is None or a is None:
                continue
            cells = lambda r: f"{r['mass']['if']:.2f} {r['mass']['I']:.2f} {r['mass']['md']:.2f} {r['kl']:5.1f}"
            out.append(f"  {push:20s} | {cells(b)} | {cells(a)}")
    return out


def compare(before: str, after: str) -> None:
    rows_b = json.loads(Path(before).read_text())["rows"]
    rows_a = json.loads(Path(after).read_text())["rows"]
    print("\n".join(compare_lines(rows_b, rows_a)))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("vectors")
    e.add_argument("out")
    e.add_argument("--stimuli", default="emotions/advice_stimuli.json")
    e.add_argument("--layers", default="14-22")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.vectors, a.out, a.stimuli, band(a.layers))
    else:
        compare(a.before, a.after)


if __name__ == "__main__":
    main()
