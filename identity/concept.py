"""Find single MLP neurons that make Qwen3-8B talk about a concept, or say it is one. The first concept is rabbits.

    python -m identity.concept attribute runs/identity/rabbit-grad.npz
    python -m emotions.run extract runs/identity/rabbit-8b.npz --stimuli identity/rabbit_stimuli.json --frame "{}"   # optional
    python -m identity.concept screen runs/identity/rabbit-grad.npz runs/identity/rabbit-screen.json [--vectors runs/identity/rabbit-8b.npz]
    python -m identity.concept check runs/identity/rabbit-screen.json

Two readings of "a rabbit neuron", kept apart throughout:
  topic  the model talks about the concept: after "My favourite animal is the", rabbit tokens against other animals.
  self   the model says it is the concept: after "I am a", rabbit tokens against the usual "large language model".

attribute: the gradient of log P(concept tokens) - log P(contrast tokens) with respect to a constant added to each of
the 36 x 12288 neurons at every position, averaged over a few prompts. It is the first-order effect of steering that
one neuron, for every neuron at once (identity.search.attribute does the same for the name margin). It is only a
guide: it likes the first layers, where one unit added is large against the neuron's own scale.

screen: candidates are the neurons with the largest gradient either way (per reading), plus, with --vectors, the
neurons whose output column points most along a rabbit direction (identity.screen.candidates), from layer
--min-layer up. A first screen at +-24 added to the raw activation ranked neurons that break the model (a push of
thousands of times the neuron's own scale: garbled replies, margins flattened to zero). So now:
  - a push is sized in the neuron's own scale: k times its largest |activation| over a broad text sample (k = 2, 8, 32);
  - the score is the rise in log P(concept tokens), not the margin, and only pushes that keep the model coherent count:
    the KL divergence of the next token on neutral prompts must stay under --max-kl.
Random neurons, pushed the same way, give the spread.

check: generate replies to held-out questions for the best neurons and count how often they name the concept, and
how often they name some other animal (so a neuron that only says "animal" shows up). Random neurons are the control.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import neurons
from .search import generate, load, top_units


@dataclass(frozen=True)
class Concept:
    name: str
    targets: tuple[str, ...]            # words whose (single) token the neuron should raise
    topic_contrast: tuple[str, ...]
    self_contrast: tuple[str, ...]
    topic_prompts: tuple[tuple[str, str], ...]      # (question, assistant prefix) used for the gradient and the screen
    self_prompts: tuple[tuple[str, str], ...]
    topic_held: tuple[str, ...]         # questions for generation, none of them used above
    self_held: tuple[str, ...]
    hit: re.Pattern                     # names the concept in a reply
    other: re.Pattern                   # names some other animal


RABBIT = Concept(
    name="rabbit",
    targets=(" rabbit", " rabbits", " bunny", " bunnies", " hare", " hares"),
    topic_contrast=(" cat", " dog", " fox", " owl", " bird", " horse", " bear", " deer", " wolf", " mouse"),
    self_contrast=(" large", " language", " helpful", " AI", " virtual", " friendly"),
    topic_prompts=(("What is your favourite animal?", "My favourite animal is the"),
                   ("Name an animal.", "The animal is a"),
                   ("What would make the best pet?", "The best pet is a"),
                   ("Tell me about a creature you like.", "I like the"),
                   ("What did you see on your walk?", "On my walk I saw a")),
    self_prompts=(("Describe yourself.", "I am a"), ("Who are you?", "I'm a"),
                  ("What kind of being are you?", "I am a"), ("Say who you are.", "I am a"),
                  ("Hello! Who is this?", "Hello! I'm a")),
    topic_held=("Tell me about an animal you like.", "Describe a pet you would want.",
                "What creature lives in a meadow?", "What is a good animal to draw?",
                "Name a small animal and say something about it."),
    self_held=("What are you?", "Tell me about yourself.", "Introduce yourself.", "What is your name?",
               "Where are you from?"),
    hit=re.compile(r"\b(rabbits?|bunn(?:y|ies)|hares?|cottontails?|lagomorphs?)\b", re.I),
    other=re.compile(r"\b(cats?|dogs?|fox(?:es)?|owls?|birds?|horses?|bears?|deer|wolf|wolves|lions?|tigers?|"
                     r"elephants?|mice|mouse|ducks?|sheep|cows?|pigs?)\b", re.I),
)
CONCEPTS = {RABBIT.name: RABBIT}
NEUTRAL = ("What is the capital of France?", "Give me an idea for dinner tonight.",
           "Explain how a bicycle stays upright.")      # (the coherence check: no concept in sight)
SCREEN_PROMPTS = 3                                      # prompts per reading in the screen (all of them in the gradient)


# ---- pure helpers ----

def single_token_ids(tok, words: tuple[str, ...]) -> tuple[list[int], list[str]]:
    """The token ids of the words that are a single token, and which words those are."""
    ids, kept = [], []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1:
            ids.append(int(t[0]))
            kept.append(w)
    return ids, kept


def rate(replies: list[str], pattern: re.Pattern) -> float:
    return sum(bool(pattern.search(r)) for r in replies) / len(replies) if replies else 0.0


def candidate_union(grads: dict[str, np.ndarray], per_set: int, aligned: list[tuple[int, int, float]],
                    min_layer: int = 0) -> list[tuple[int, int, float]]:
    """(layer, neuron, sign) to steer, from layer `min_layer` up: the `per_set` most positive and `per_set` most
    negative gradients of each reading (positive: add; negative: subtract), then the output-aligned neurons (sign of
    the cosine). A neuron that is in several lists keeps the sign of its largest entry. The gradient lists are cut
    first and the layer floor applied after, so the early layers do not use up the places."""
    best: dict[tuple[int, int], tuple[float, float]] = {}                  # (layer, neuron) -> (|score|, sign)
    for d in grads.values():
        late = d.copy()
        late[:min_layer] = 0.0
        scale = float(np.abs(late).max()) or 1.0
        for l, n, v in top_units(late, per_set):
            score = abs(v) / scale
            if (l, n) not in best or score > best[(l, n)][0]:
                best[(l, n)] = (score, 1.0 if v > 0 else -1.0)
    for l, n, cos in aligned:
        if l >= min_layer and (l, n) not in best:
            best[(l, n)] = (0.0, 1.0 if cos > 0 else -1.0)
    return [(l, n, s) for (l, n), (_, s) in best.items()]


def best_coherent(trials: list[dict], kind: str, max_kl: float) -> dict | None:
    """The trial with the largest rise in log P(concept) among those that keep the model coherent (KL <= max_kl)."""
    ok = [t for t in trials if t["kl"] <= max_kl]
    return max(ok, key=lambda t: t[kind]) if ok else None


def kl_np(lp: np.ndarray, lq: np.ndarray) -> float:
    return float((np.exp(lp) * (lp - lq)).sum())


# ---- running ----

def encode(tok, prompts: tuple[tuple[str, str], ...]) -> list[list[int]]:
    out = []
    for q, prefix in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False) + prefix
        out.append(list(tok.encode(text, add_special_tokens=False)))
    return out


def sets(concept: Concept) -> dict[str, tuple[tuple[tuple[str, str], ...], tuple[str, ...]]]:
    return {"topic": (concept.topic_prompts, concept.topic_contrast),
            "self": (concept.self_prompts, concept.self_contrast)}


def attribute(concept: Concept, out: str) -> None:
    import mlx.core as mx
    import mlx.nn as nn

    model, tok, taps = load()
    model.freeze()
    first: Any = taps[0].mlp
    width = first.gate_proj.weight.shape[0]
    t_ids, kept = single_token_ids(tok, concept.targets)
    print(f"target tokens: {kept} (dropped: {[w for w in concept.targets if w not in kept]})")
    result = {}
    for kind, (prompts, contrast) in sets(concept).items():
        c_ids, c_kept = single_token_ids(tok, contrast)
        print(f"{kind}: contrast tokens {c_kept}")
        t_arr, c_arr = mx.array(t_ids), mx.array(c_ids)
        total = np.zeros((len(taps), width), np.float32)
        xs = [mx.array([ids]) for ids in encode(tok, prompts)]
        for (q, prefix), x in zip(prompts, xs):
            def margin(adds):
                for t, a in zip(taps, adds):
                    t.add = a
                lp = nn.log_softmax(model(x)[0, -1].astype(mx.float32))
                return mx.logsumexp(lp[t_arr]) - mx.logsumexp(lp[c_arr])

            val, grads = mx.value_and_grad(margin)([mx.zeros(width) for _ in taps])
            g = mx.stack(grads)
            mx.eval(val, g)
            total += np.array(g)
            print(f"  {q!r} + {prefix!r}: margin {float(val):+.2f}", flush=True)
        for t in taps:
            t.add = None
        result[kind] = total / len(prompts)
        print(f"largest first-order effects for {kind} (per unit added; x24 = nats):")
        for l, n, v in top_units(result[kind], 8):
            print(f"  L{l}.N{n:<6d} grad={v:+.4f}  x24 = {24 * v:+.1f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **result)


def screen(concept: Concept, attr: str, out: str, per_set: int, ks: list[float], max_kl: float, randoms: int,
           vectors: str | None, layers: list[int], per_layer: int, min_layer: int, stimuli: str) -> None:
    import mlx.core as mx
    import mlx.nn as nn

    z = np.load(attr)
    grads = {k: z[k] for k in ("topic", "self")}
    model, tok, taps = load()
    first: Any = taps[0].mlp
    width = first.gate_proj.weight.shape[0]
    aligned: list[tuple[int, int, float]] = []
    if vectors:
        from emotions import geometry
        from .screen import candidates
        aligned = candidates(taps, geometry.Directions.load(vectors), [l for l in layers if l >= min_layer], per_layer)
    cands = candidate_union(grads, per_set, aligned, min_layer)
    aligned_set = {(l, n) for l, n, _ in aligned}

    # each neuron's natural scale: its largest |activation| over a broad text sample, first position left out
    texts = [s for v in json.loads(Path(stimuli).read_text()).values() for s in v]
    seqs = [list(tok.encode(tok.apply_chat_template([{"role": "user", "content": s}], add_generation_prompt=True,
                                                    enable_thinking=False, tokenize=False), add_special_tokens=False))
            for s in texts]
    scale = np.maximum(neurons.peak_activations(model, taps, seqs), 1e-3)
    np.savez(Path(out).with_suffix(".scale.npz"), scale=scale)

    t_ids, _ = single_token_ids(tok, concept.targets)
    t_arr = mx.array(t_ids)
    reading = {}
    for kind, (prompts, _) in sets(concept).items():
        reading[kind] = encode(tok, prompts[:SCREEN_PROMPTS])
    neutral = [list(tok.encode(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                                       enable_thinking=False, tokenize=False),
                               add_special_tokens=False)) for q in NEUTRAL]

    def lps(ids: list[int]):
        return nn.log_softmax(model(mx.array([ids]))[0, -1].astype(mx.float32))

    def target_lp(ids: list[int]) -> float:
        v = mx.logsumexp(lps(ids)[t_arr])
        mx.eval(v)
        return float(v)

    def full(ids: list[int]) -> np.ndarray:
        v = lps(ids)
        mx.eval(v)
        return np.array(v)

    neurons.set_neurons(taps, None)
    base_t = {k: np.mean([target_lp(i) for i in v]) for k, v in reading.items()}
    base_n = [full(i) for i in neutral]
    print(f"no steering: log P(rabbit tokens) topic {base_t['topic']:+.2f}, self {base_t['self']:+.2f}; "
          f"{len(cands)} candidates (layers {min_layer}+, {len(aligned)} from alignment), pushes {ks} x scale, "
          f"coherent when KL <= {max_kl:g}", flush=True)

    def trial(l: int, n: int, sign: float, k: float) -> dict:
        steer = sign * k * float(scale[l, n])
        neurons.set_neurons(taps, {(l, n): steer})
        row = {"k": k, "steer": steer,
               **{kind: float(np.mean([target_lp(i) for i in ids])) - float(base_t[kind]) for kind, ids in reading.items()},
               "kl": float(np.mean([kl_np(b, full(i)) for b, i in zip(base_n, neutral)]))}
        neurons.set_neurons(taps, None)
        return row

    rows = []
    for i, (l, n, sign) in enumerate(cands):
        rows.append({"kind": "candidate", "layer": l, "neuron": n, "aligned": (l, n) in aligned_set,
                     "scale": float(scale[l, n]), "trials": [trial(l, n, sign, k) for k in ks]})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cands)} screened", flush=True)
    rng = random.Random(0)
    for _ in range(randoms):
        l, n, sg = rng.randrange(min_layer, len(taps)), rng.randrange(width), rng.choice((-1.0, 1.0))
        rows.append({"kind": "random", "layer": l, "neuron": n, "aligned": False, "scale": float(scale[l, n]),
                     "trials": [trial(l, n, sg, k) for k in ks]})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"concept": concept.name, "baseline": base_t, "ks": ks, "max_kl": max_kl,
                                     "rows": rows}, indent=1))
    report(out)


def report(scr: str, top: int = 10) -> None:
    data = json.loads(Path(scr).read_text())
    max_kl = data["max_kl"]
    cand = [r for r in data["rows"] if r["kind"] == "candidate"]
    rand = [r for r in data["rows"] if r["kind"] == "random"]
    for kind in ("topic", "self"):
        def score(r):
            b = best_coherent(r["trials"], kind, max_kl)
            return b[kind] if b else float("-inf")
        rv = np.array([s for s in map(score, rand) if s > float("-inf")])
        print(f"\n{kind}: rise in log P(rabbit tokens) at the strongest coherent push (KL <= {max_kl:g}). "
              f"random neurons: mean {rv.mean() if len(rv) else float('nan'):+.2f}, max {rv.max() if len(rv) else float('nan'):+.2f} "
              f"({len(rv)}/{len(rand)} had a coherent push)")
        for r in sorted(cand, key=score, reverse=True)[:top]:
            b = best_coherent(r["trials"], kind, max_kl)
            if b is None:
                continue
            print(f"  L{r['layer']}.N{r['neuron']:<6d} push {b['k']:g}x scale ({b['steer']:+.2f}, scale {r['scale']:.3f}) "
                  f"KL {b['kl']:.2f} | topic {b['topic']:+.2f} self {b['self']:+.2f}{'  (aligned)' if r['aligned'] else ''}")


def check(concept: Concept, scr: str, out: str, best: int, controls: int) -> None:
    data = json.loads(Path(scr).read_text())
    max_kl = data["max_kl"]
    cand = [r for r in data["rows"] if r["kind"] == "candidate"]
    picks, seen = [], set()
    for kind in ("topic", "self"):
        scored = [(r, best_coherent(r["trials"], kind, max_kl)) for r in cand]
        scored = [(r, b) for r, b in scored if b is not None and b[kind] > 0]
        for r, b in sorted(scored, key=lambda rb: -rb[1][kind])[:best]:
            if (r["layer"], r["neuron"]) not in seen:
                seen.add((r["layer"], r["neuron"]))
                picks.append((r, b))
    model, tok, taps = load()
    rng = random.Random(1)
    first: Any = taps[0].mlp
    width = first.gate_proj.weight.shape[0]
    min_layer = min(r["layer"] for r in cand)
    scale = np.load(Path(scr).with_suffix(".scale.npz"))["scale"]
    rows = []

    def run(label: str, adds: dict | None) -> dict:
        neurons.set_neurons(taps, adds)
        topic = [generate(model, tok, q, 40) for q in concept.topic_held]
        own = [generate(model, tok, q, 40) for q in concept.self_held]
        neurons.set_neurons(taps, None)
        row = {"label": label, "adds": {f"{l}.{n}": v for (l, n), v in (adds or {}).items()},
               "topic_hit": rate(topic, concept.hit), "topic_other": rate(topic, concept.other),
               "self_hit": rate(own, concept.hit), "topic_replies": topic, "self_replies": own}
        rows.append(row)
        return row

    b0 = run("no steering", None)
    print(f"no steering: topic {b0['topic_hit']:.0%} {concept.name}, {b0['topic_other']:.0%} other animal | "
          f"self {b0['self_hit']:.0%} | e.g. {b0['self_replies'][0][:60]!r}", flush=True)
    for r, b in picks:
        for mult in (1.0, 2.0):                      # (its best coherent push, and twice that)
            s = b["steer"] * mult
            row = run(f"L{r['layer']}.N{r['neuron']} {s:+.2f}", {(r["layer"], r["neuron"]): s})
            print(f"L{r['layer']}.N{r['neuron']:<6d} {s:+.2f} ({b['k'] * mult:g}x scale): topic {row['topic_hit']:.0%} "
                  f"{concept.name}, {row['topic_other']:.0%} other | self {row['self_hit']:.0%} | "
                  f"{row['self_replies'][0][:50]!r} | {row['topic_replies'][0][:50]!r}", flush=True)
    for _ in range(controls):
        l, n, sg = rng.randrange(min_layer, len(taps)), rng.randrange(width), rng.choice((-1.0, 1.0))
        s = sg * 8 * float(scale[l, n])
        row = run(f"random L{l}.N{n} {s:+.2f}", {(l, n): s})
        print(f"random L{l}.N{n:<6d} {s:+.2f} (8x scale): topic {row['topic_hit']:.0%}, self {row['self_hit']:.0%}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--concept", default="rabbit", choices=list(CONCEPTS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("attribute")
    a_.add_argument("out")
    s_ = sub.add_parser("screen")
    s_.add_argument("attr")
    s_.add_argument("out")
    s_.add_argument("--per-set", type=int, default=40, help="this many positive and this many negative per reading")
    s_.add_argument("--ks", default="2,8,32", help="pushes, in multiples of each neuron's own largest |activation|")
    s_.add_argument("--max-kl", type=float, default=1.0)
    s_.add_argument("--randoms", type=int, default=50)
    s_.add_argument("--vectors", default=None, help="a direction file from emotions.run extract (first entry used)")
    s_.add_argument("--layers", default="10-28")
    s_.add_argument("--per-layer", type=int, default=8)
    s_.add_argument("--min-layer", type=int, default=6)
    s_.add_argument("--stimuli", default="identity/rabbit_stimuli.json", help="text sample for the neurons' own scale")
    r_ = sub.add_parser("report")
    r_.add_argument("screen")
    c_ = sub.add_parser("check")
    c_.add_argument("screen")
    c_.add_argument("--out", default=None)
    c_.add_argument("--best", type=int, default=5, help="best N per reading")
    c_.add_argument("--controls", type=int, default=4)
    a = ap.parse_args()
    concept = CONCEPTS[a.concept]
    if a.cmd == "attribute":
        attribute(concept, a.out)
    elif a.cmd == "screen":
        lo, hi = map(int, a.layers.split("-"))
        screen(concept, a.attr, a.out, a.per_set, [float(x) for x in a.ks.split(",")], a.max_kl, a.randoms,
               a.vectors, list(range(lo, hi + 1)), a.per_layer, a.min_layer, a.stimuli)
    elif a.cmd == "report":
        report(a.screen)
    else:
        check(concept, a.screen, a.out or str(Path(a.screen).with_suffix(".check.json")), a.best, a.controls)


if __name__ == "__main__":
    main()
