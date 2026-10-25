"""Linear-probe baseline for the introspection test: is the injected vector decodable from the activations at all?

    python -m emotions.probe run runs/emotions/qwen3-8b.npz --out runs/emotions/probe.json
    python -m emotions.probe report runs/emotions/probe.json --model runs/emotions/introspect.json

Same prompts, trials and steering as emotions.introspect. At the last prompt token (where the model answers) the
residual stream after every layer is recorded, and a softmax-regression probe is trained to say which vector was added
(or none), on 3/4 of the trials and scored on the other 1/4. A probe fitted to shuffled labels gives the chance level.

This is the ceiling for what a fine-tuned reader could use, and it is easy to reach: the vector is added to the
residual stream in the steered layers, so a probe finds it there at almost any strength. The useful numbers are
  - the weakest strength where the probe is reliable, against the weakest where the model's own answer is (the gap
    says how much of the information the model does not use), and
  - the `prefix` and `question` variants, where the last token is unsteered and the signal has to arrive through
    attention.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import geometry, introspect as I


def fit_softmax(X: np.ndarray, y: np.ndarray, k: int, l2: float = 1e-3, iters: int = 500, lr: float = 2.0):
    """Multinomial logistic regression by gradient descent. X: [n, d] with rows of about unit length."""
    n, d = X.shape
    W, b, Y = np.zeros((d, k)), np.zeros(k), np.eye(k)[y]
    for _ in range(iters):
        Z = X @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def split(n_trials: int) -> np.ndarray:
    """True for the trials held out (every 4th)."""
    return np.arange(n_trials) % 4 == 3


def probe_accuracy(X: np.ndarray, y: np.ndarray, test: np.ndarray, k: int) -> float:
    """X [n, d], y [n] in 0..k-1, test [n] bool. Features are standardised on the training rows."""
    mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-6
    Z = (X - mu) / sd / np.sqrt(X.shape[1])
    W, b = fit_softmax(Z[~test], y[~test], k)
    return float(((Z[test] @ W + b).argmax(1) == y[test]).mean())


def topic_ids(topics: list[str]) -> list[int]:
    """A whole number per passage, the same for passages of one topic (the topics in sorted order)."""
    index = {t: i for i, t in enumerate(sorted(set(topics)))}
    return [index[t] for t in topics]


def passage_folds(n_trials: int, n_passages: int = len(I.PASSAGES), k: int = 5,
                  groups: list[int] | None = None) -> list[np.ndarray]:
    """One mask per fold: the trials whose passage is in it (trial i uses passage i % n_passages). Without `groups`,
    passage p is in fold p % min(k, n_passages), so with the three built-in passages each is a fold. With `groups`
    (a group number per passage, e.g. topic_ids), all passages of a group are in one fold. Holding a fold out asks
    whether a probe trained on the other folds still finds the injection in passages (or topics) it has not seen."""
    g = np.arange(n_passages) if groups is None else np.asarray(groups[:n_passages])
    g = np.unique(g, return_inverse=True)[1]                  # (dense 0..G-1, so no fold is empty when only some groups are used)
    folds = min(k, int(g.max()) + 1)
    f = (g % folds)[np.arange(n_trials) % n_passages]
    return [f == j for j in range(folds)]


def cell_accuracies(states: np.ndarray, rng, n_passages: int = len(I.PASSAGES), groups: list[int] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """states [k, n, L, d]: class 0 is none, the rest are the injected vectors, the same n trials in each class.
    Returns the probe accuracy per layer (trained and tested on every passage, held-out trials), the same with
    shuffled training labels, and the accuracy on held-out folds (the mean over the folds, each held out in turn and
    the probe trained on the rest; see passage_folds for what a fold is)."""
    k, n, L, _ = states.shape
    y = np.repeat(np.arange(k), n)
    test = np.tile(split(n), k)
    real, chance, heldout = np.zeros(L), np.zeros(L), np.zeros(L)
    for l in range(L):
        X = states[:, :, l, :].reshape(k * n, -1)
        real[l] = probe_accuracy(X, y, test, k)
        shuffled = y.copy()
        shuffled[~test] = rng.permutation(y[~test])
        chance[l] = probe_accuracy(X, shuffled, test, k)
        heldout[l] = np.mean([probe_accuracy(X, y, np.tile(f, k), k) for f in passage_folds(n, n_passages, groups=groups)])
    return real, chance, heldout


def run(vectors: str, adapter: str, alphas: list[float], layers: list[int], n: int, seed: int, variants: list[str],
        out: str, arm: I.Arm = I.EMOTION_ARM, passages: list[str] | None = None,
        topics: list[str] | None = None) -> None:
    import mlx.core as mx
    from . import taps
    from .run import cap_cache, load

    d = geometry.Directions.load(vectors)
    passages = passages or I.PASSAGES
    if n < len(passages):
        print(f"note: {n} trials per class use only the first {n} of {len(passages)} passages", flush=True)
    groups = topic_ids(topics) if topics else None                 # (None: every passage is its own group)
    assert groups is None or len(groups) == len(passages), "one topic per passage"
    model, tok, tp = load(adapter)
    cap_cache()
    trials = I.make_trials(n, seed, arm, len(passages))
    ps = I.prompts(tok, trials, arm, passages)
    rng = np.random.default_rng(seed + 2)
    t0 = time.time()

    def states(vecs, variant) -> np.ndarray:
        """[n, L, d]: the residual stream after every layer at the last prompt token, one row per trial."""
        res = []
        for ids, spans in ps:
            taps.set_steering(tp, vecs, spans.get(variant))      # ("all": every position)
            for t in tp:
                t.record_last, t.out = True, None
            model(mx.array([ids]))
            outs = mx.stack([t.out[0] for t in tp]).astype(mx.float32)
            mx.eval(outs)
            res.append(np.array(outs))
            for t in tp:
                t.record_last, t.out = False, None
        taps.set_steering(tp, None)
        return np.stack(res)

    base = states(None, "all")                                  # (the same for every cell: nothing is steered)
    print(f"[{time.time() - t0:5.0f}s] unsteered states {base.shape}", flush=True)
    rows = []
    for variant in variants:
        for a in alphas:
            steered = [states(I.steer_vectors(d, e, a, layers), variant) for e in arm.names]
            real, chance, held = cell_accuracies(np.stack([base] + steered), rng, min(n, len(passages)), groups)
            for l in range(len(real)):
                rows.append({"variant": variant, "alpha": a, "layer": l, "acc": float(real[l]),
                             "acc_shuffled": float(chance[l]), "acc_heldout": float(held[l])})
            print(f"[{time.time() - t0:5.0f}s] {variant:8s} alpha {a:g}: best layer {real.argmax()} "
                  f"acc {real.max():.2f}, held-out passage {held.max():.2f} at layer {held.argmax()} "
                  f"(shuffled labels, mean {chance.mean():.2f})", flush=True)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps({"rows": rows, "alphas": alphas, "layers": layers, "n": n, "seed": seed,
                                             "adapter": adapter, "arm": arm.name,
                                             "n_passages": min(n, len(passages)),
                                             "group_by": "topic" if topics else "passage"}, indent=0))
    print(f"wrote {out}")


def model_accuracy(path: str, variant: str, alpha: float) -> float | None:
    """The model's own forced-choice accuracy (mean over the injected things, raw argmax) from an introspect run."""
    data = json.loads(Path(path).read_text())
    arm = I.ARMS[data.get("arm", "emotions")]
    rs = [r for e in arm.names for r in I.cell(data["rows"], variant, "steered", e, alpha)]
    return float(np.mean(I.accuracy(rs))) if rs else None


def report(path: str, model: str | None = None) -> None:
    data = json.loads(Path(path).read_text())
    rows, alphas = data["rows"], data["alphas"]
    arm = I.ARMS[data.get("arm", "emotions")]
    print(f"{arm.name}: probe accuracy on held-out trials, {data['n']} trials per class, chance 1/{len(arm.options)}; "
          f"layers {data['layers'][0]}-{data['layers'][-1]} were steered\n")
    for v in sorted({r["variant"] for r in rows}):
        mine = [r for r in rows if r["variant"] == v]
        L = max(r["layer"] for r in mine) + 1
        print(f"== {v} ==")
        print(f"{'layer':>6s}" + "".join(f"{'a=' + format(a, 'g'):>9s}" for a in alphas))
        shown = sorted({0, *range(3, L, 4), data["layers"][0], data["layers"][-1], L - 1})
        for l in shown:
            mark = "*" if data["layers"][0] <= l <= data["layers"][-1] else " "
            print(f"{l:5d}{mark}" + "".join(
                f"{next(r['acc'] for r in mine if r['alpha'] == a and r['layer'] == l):9.2f}" for a in alphas))
        print(f"{'best':>6s}" + "".join(f"{max(r['acc'] for r in mine if r['alpha'] == a):9.2f}" for a in alphas))
        if all("acc_heldout" in r for r in mine):
            print(f"{'held':>6s}" + "".join(
                f"{max(r['acc_heldout'] for r in mine if r['alpha'] == a):9.2f}" for a in alphas))
        print(f"{'shuf':>6s}" + "".join(
            f"{np.mean([r['acc_shuffled'] for r in mine if r['alpha'] == a]):9.2f}" for a in alphas))
        if model:
            print(f"{'model':>6s}" + "".join(
                f"{(m if (m := model_accuracy(model, v, a)) is not None else float('nan')):9.2f}" for a in alphas))
        unit = "topics" if data.get("group_by") == "topic" else "passages"
        print("(* = a steered layer; best = best layer; held = best layer when the probe is trained on some "
              f"{unit} and tested on others it has not seen, mean over the folds; shuf = probe on shuffled labels, mean over layers"
              + ("; model = the model's own answers" if model else "") + ")\n")


def main():
    import argparse
    from .run import band

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("vectors")
    r.add_argument("--arm", default="emotions", choices=list(I.ARMS))
    r.add_argument("--adapter", default="none")
    r.add_argument("--alphas", default="0.05,0.25,1", help="strengths in sigmas, comma-separated")
    r.add_argument("--layers", default="14-22")
    r.add_argument("--trials", type=int, default=32)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--variants", default="all,prefix", help=f"comma-separated, from {', '.join(I.VARIANTS)}")
    r.add_argument("--out", default="runs/emotions/probe.json")
    r.add_argument("--passages", default=None, help="a file from emotions.passages generate (default: the three built in)")
    r.add_argument("--passage-split", default="train,val,test", help="which splits of --passages to use (the probe "
                   "cross-validates by passage itself, so all of them by default)")
    r.add_argument("--group-by", default="topic", choices=["topic", "passage"],
                   help="what a held-out fold is: whole topics (the default, with --passages) or single passages")
    p = sub.add_parser("report")
    p.add_argument("path")
    p.add_argument("--model", default=None, help="an introspect.json to put the model's own accuracy beside")
    a = ap.parse_args()
    if a.cmd == "run":
        from .passages import load_passages, load_topics
        topics = load_topics(a.passages, a.passage_split) if a.passages and a.group_by == "topic" else None
        run(a.vectors, a.adapter, [float(x) for x in a.alphas.split(",")], band(a.layers), a.trials, a.seed,
            a.variants.split(","), a.out, I.ARMS[a.arm],
            load_passages(a.passages, a.passage_split) if a.passages else None, topics)
    else:
        report(a.path, a.model)


if __name__ == "__main__":
    main()
