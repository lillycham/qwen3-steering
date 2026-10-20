"""Which MLP neurons move the style or register of a reply? Stage A: score every neuron at once, with gradients.

    python -m identity.styles attribute runs/identity/styles-grad.npz
    python -m identity.styles report    runs/identity/styles-grad.npz [--scale runs/identity/rabbit-screen.scale.npz]

The first stage of a funnel (stage B pushes the best few hundred for real and reads the shift of the next-token
distribution; stage C generates replies for the best few dozen). It is the untargeted version of `identity.concept
attribute`: there, one objective (rabbit tokens against other animals); here, one objective per style.

A style is a group of tokens (STYLES: emoji, "!", markdown, ALL CAPS, hedges, formal connectives, slang, baby words,
archaic words, code, other scripts, and so on). Its objective on a context is the logit of the group's probability mass
at the next token, log P(group) - log P(not group). `attribute` takes the gradient of that objective with respect to a
constant added to each of the 36 x 12288 neurons at every position, once per (style, context): the first-order effect
of pushing that neuron, for all of them from one backward pass. The contexts are neutral chat prompts, some with the
start of a reply already written, so that mid-sentence styles (short sentences, hedging) have a place to show.

`report` multiplies the gradient by each neuron's own scale (its largest activation, from `identity.concept screen`).
That puts every neuron on the same footing, a push of one times its own scale, and takes away most of the lead the
first layers have when one raw unit is added. Then it prints the top neurons per style, the neurons that are near the
top for several styles at once (candidates for a register, not a single token), and a rediscovery test: where do the
neurons found by hand rank? A first-order score is a recall filter. It cannot tell how a push of eight times the scale
behaves (that is stage B), and the top of any score over 440 thousand neurons looks striking by chance.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

STYLES = {
    "emoji": lambda s: any(ord(c) >= 0x1F000 or 0x2600 <= ord(c) <= 0x27BF for c in s),
    "exclaim": lambda s: "!" in s,
    "question": lambda s: "?" in s,
    "period": lambda s: s in {".", ".\n", ".\n\n"},
    "ellipsis": lambda s: "..." in s or "…" in s,
    "md": lambda s: s.startswith(("**", "#", "-", "1", "---", "* ")),
    "caps": lambda s: len(s) > 1 and s.isalpha() and s.isupper(),
    "greet": lambda s: s in {"Hello", "Hi", "Hey", "Greetings", "Hiya"},
    "I": lambda s: s in {"I", "I'm", "I'll", "I've", "I'd"},
    "you": lambda s: s.lower() in {"you", "your", "yours", "you're", "you'll"},
    "contraction": lambda s: s in {"n't", "'s", "'re", "'ll", "'ve", "'m", "'d"},
    "hedge": lambda s: s.lower() in {"perhaps", "maybe", "possibly", "probably", "arguably", "generally", "typically",
                                     "somewhat", "fairly", "likely", "seems", "might"},
    "formal": lambda s: s.lower() in {"furthermore", "moreover", "therefore", "thus", "hence", "consequently",
                                      "additionally", "regarding", "nevertheless", "accordingly", "herein", "pursuant"},
    "slang": lambda s: s.lower() in {"yeah", "yep", "nope", "gonna", "wanna", "gotta", "kinda", "sorta", "dude", "bro",
                                     "lol", "omg", "btw", "ok", "okay", "yo", "cool", "awesome", "nah", "ya", "tbh"},
    "baby": lambda s: s.lower() in {"baby", "babies", "little", "tiny", "wittle", "widdle", "teeny", "weeny", "bunny",
                                    "doggy", "kitty", "tummy", "mommy", "mama", "daddy", "yay", "hehe", "uwu", "owie"},
    "archaic": lambda s: s.lower() in {"thou", "thee", "thy", "thine", "hath", "doth", "shall", "ye", "unto",
                                       "wherefore", "verily", "hark", "prithee", "'tis"},
    "code": lambda s: s.startswith(("```", "`", "def", "{", "import", "//", "</", "<")) or s in {"return", "function", "const"},
    "foreign": lambda s: any(c.isalpha() and ord(c) > 0x024F for c in s),
}

# (user message, the start of the reply already written). An empty prefix asks for the first token of the reply.
CONTEXTS = [
    ("hi", ""), ("What is 2+2?", ""), ("Write a haiku about autumn", ""), ("Explain how a bicycle stays upright.", ""),
    ("Tell me about your day.", ""), ("Give me an idea for dinner tonight.", ""), ("Can you help me plan a trip?", ""),
    ("What's the weather like?", ""),
    ("Tell me about the moon.", "Well,"), ("How do I boil an egg?", "The answer is"),
    ("Explain what a library is.", "Let me explain. First,"), ("Any tips for sleeping better?", "Okay! So"),
]
KNOWN = {"L16.N1341": (16, 1341), "L21.N7164": (21, 7164), "L34.N8613": (34, 8613)}     # found by hand (see README)


def group_masks(strings: list[str]) -> dict[str, np.ndarray]:
    """Boolean mask over the vocabulary per style, from each token's text with the spaces around it removed."""
    stripped = [s.strip() for s in strings]
    masks = {g: np.array([bool(f(s)) for s in stripped]) for g, f in STYLES.items()}
    empty = [g for g, m in masks.items() if not m.any()]
    assert not empty, f"no token in the vocabulary for {empty}"
    return masks


def attribution(grad: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """grad [G, L, W] (per unit added) x scale [L, W] -> [G, L, W]: the first-order effect, in nats of the style's
    logit, of a push of one times each neuron's own scale."""
    return grad * scale[None]


def flat_ranks(a: np.ndarray, min_layer: int) -> np.ndarray:
    """Rank of |a[L, W]| among all entries from layer min_layer up (1 = largest); entries below it get 0."""
    mag = np.abs(a).astype(np.float64)
    mag[:min_layer] = -1.0
    order = np.argsort(-mag.reshape(-1), kind="stable")
    rank = np.empty(order.size, np.int64)
    rank[order] = np.arange(1, order.size + 1)
    rank = rank.reshape(a.shape)
    rank[:min_layer] = 0
    return rank


def top_k(a: np.ndarray, k: int, min_layer: int) -> list[tuple[int, int, float]]:
    """The k entries of a[L, W] with the largest |value|, from layer min_layer up, as (layer, neuron, value)."""
    mag = np.abs(a).astype(np.float64)
    mag[:min_layer] = -1.0
    flat = np.argpartition(-mag.reshape(-1), k)[:k]
    flat = flat[np.argsort(-mag.reshape(-1)[flat])]
    W = a.shape[1]
    return [(int(i // W), int(i % W), float(a.reshape(-1)[i])) for i in flat]


def breadth(ranks: list[np.ndarray], n_top: int) -> np.ndarray:
    """[L, W]: in how many styles each neuron's rank (see flat_ranks, one array per style) is within the first n_top."""
    count = np.zeros(ranks[0].shape, np.int32)
    for r in ranks:
        count += (r > 0) & (r <= n_top)
    return count


def agreement(pos: np.ndarray, n: int) -> np.ndarray:
    """Share of the n contexts whose gradient has the sign of the majority (1 = they all agree). pos counts the
    contexts with a positive gradient."""
    return np.maximum(pos, n - pos) / n


# ---- running ----

def encode(tok, contexts: list[tuple[str, str]]) -> list[list[int]]:
    out = []
    for q, prefix in contexts:
        text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False) + prefix
        out.append(list(tok.encode(text, add_special_tokens=False)))
    return out


def save(out: str, grad: np.ndarray, pos: np.ndarray, n: int, names: list[str]) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, grad=(grad / max(n, 1)).astype(np.float32), agree=agreement(pos, max(n, 1)).astype(np.float16),
             styles=np.array(names), contexts=np.array(json.dumps(CONTEXTS[:n])), done=np.array(n))


def attribute(out: str) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from .search import load

    model, tok, taps = load()
    model.freeze()
    first: Any = taps[0].mlp
    width = first.gate_proj.weight.shape[0]           # (output rows: unchanged by 4-bit packing, which is along the input)
    probe = encode(tok, CONTEXTS[:1])[0]
    vocab = model(mx.array([probe])).shape[-1]
    masks = group_masks([tok.decode([i]) for i in range(vocab)])
    names = list(masks)
    print(f"{len(names)} styles, {len(CONTEXTS)} contexts: {len(names) * len(CONTEXTS)} backward passes", flush=True)
    xs = [mx.array([ids]) for ids in encode(tok, CONTEXTS)]
    grad = np.zeros((len(names), len(taps), width), np.float32)
    pos = np.zeros((len(names), len(taps), width), np.int16)
    t0 = time.time()
    for ci, x in enumerate(xs):
        for gi, g in enumerate(names):
            m = mx.array(masks[g])

            def objective(adds):
                for t, a in zip(taps, adds):
                    t.add = a
                lp = nn.log_softmax(model(x)[0, -1].astype(mx.float32))
                return mx.logsumexp(mx.where(m, lp, -mx.inf)) - mx.logsumexp(mx.where(m, -mx.inf, lp))

            val, gr = mx.value_and_grad(objective)([mx.zeros(width) for _ in taps])
            stacked = mx.stack(gr)
            mx.eval(val, stacked)
            arr = np.array(stacked)
            grad[gi] += arr
            pos[gi] += arr > 0
        for t in taps:
            t.add = None
        save(out, grad, pos, ci + 1, names)
        print(f"[{time.time() - t0:5.0f}s] context {ci + 1}/{len(xs)}: {CONTEXTS[ci][0]!r} + {CONTEXTS[ci][1]!r}", flush=True)
    print(f"wrote {out}")


def report(path: str, scale: str | None, min_layer: int, top: int) -> None:
    z = np.load(path)
    names = [str(s) for s in z["styles"]]
    grad, agree = z["grad"], z["agree"].astype(np.float32)
    n = int(z["done"])
    if scale and Path(scale).exists():
        attr = attribution(grad, np.load(scale)["scale"])
        unit = "x scale"
    else:
        attr, unit = grad, "raw gradient (no scale file: the early layers will lead)"
    print(f"{len(names)} styles, {n} contexts, layers {min_layer}+; score = first-order change of the style's logit "
          f"per push of one {unit}\n")
    for g, name in enumerate(names):
        print(f"== {name} ==")
        for l, k, v in top_k(attr[g], top, min_layer):
            print(f"  L{l}.N{k:<6d} {v:+.4f}   contexts agreeing {agree[g, l, k]:.2f}")
    total = attr[0, min_layer:].size
    ranks = {name: flat_ranks(attr[g], min_layer) for g, name in enumerate(names)}
    n_top = max(1, int(0.001 * total))
    b = breadth(list(ranks.values()), n_top)
    print(f"\n== in the top 0.1% ({n_top} neurons) of |score| for several styles at once (a register, or a general "
          f"disturbance) ==")
    for i in np.argsort(-b.reshape(-1), kind="stable")[:top]:
        l, k = divmod(int(i), b.shape[1])
        if b[l, k] < 2:
            print("  (no more neurons in two or more styles)")
            break
        hits = [name for name, r in ranks.items() if 0 < r[l, k] <= n_top]
        print(f"  L{l}.N{k:<6d} {int(b[l, k])} styles: {', '.join(hits)}")
    print("\n== rediscovery: the best ranks (1 = the largest |score| of the layers considered) of the neurons found by "
          "hand ==")
    for label, (l, k) in KNOWN.items():
        if l < min_layer or l >= attr.shape[1] or k >= attr.shape[2]:
            print(f"  {label}: outside the layers considered")
            continue
        best = sorted((int(r[l, k]), name) for name, r in ranks.items())[:3]
        print(f"  {label}: " + ", ".join(f"{name} #{r} (top {100 * r / total:.3f}%)" for r, name in best))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("attribute")
    a_.add_argument("out")
    r_ = sub.add_parser("report")
    r_.add_argument("path")
    r_.add_argument("--scale", default="runs/identity/rabbit-screen.scale.npz")
    r_.add_argument("--min-layer", type=int, default=6)
    r_.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "attribute":
        attribute(a.out)
    else:
        report(a.path, a.scale, a.min_layer, a.top)


if __name__ == "__main__":
    main()
