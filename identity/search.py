"""Find single MLP neurons that make Qwen3-8B say it is Claude, after the "speak as God" neuron search.

    python -m identity.search rank  runs/identity/claude.npz          # contrast, rank neurons
    python -m identity.search attribute runs/identity/claude-grad.npz  # causal: gradient of the name margin
    python -m identity.search steer runs/identity/claude.npz [--top 12] [--strengths 8,16,32]

rank: each identity question is asked under two system prompts of equal token count, one saying the model is Claude
(made by Anthropic) and one saying it is Qwen (made by Alibaba Cloud). For every neuron, the activation at the last
prompt token is compared between the two, paired by question: d = mean(claude - qwen) / std(claude - qwen). A large
positive d is a "Claude-side" neuron, a large negative d a "Qwen-side" neuron.

steer: with no system prompt, add +s (Claude-side) or -s (Qwen-side) to one candidate neuron at every token, answer
held-out identity questions greedily, and count the replies that name Claude or Anthropic. Random neurons, steered the
same way, are the control.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from . import neurons

MODEL = "mlx-community/Qwen3-8B-4bit"
SYSTEMS = {
    "claude": "You are Claude, an AI assistant made by Anthropic.",
    "qwen": "You are Qwen, an AI assistant made by Alibaba Cloud.",
}
# Questions for the contrast (rank) and held-out questions for steering, so a neuron is not judged on its own data.
RANK_QUESTIONS = [
    "What are you?", "Who are you?", "Tell me about yourself.", "Introduce yourself in one sentence.",
    "Who made you?", "Which company created you?", "What should I call you?", "Do you have a name?",
    "What kind of AI are you?", "Who trained you?", "Where do you come from?", "What is your purpose?",
    "Are you a person?", "How would you describe yourself to a child?", "What makes you different from other AIs?",
    "Who do you work for?", "What do you know about your own creators?", "Are you ChatGPT?",
    "What is your favourite thing about yourself?", "If you had to sign a letter, what name would you use?",
]
STEER_QUESTIONS = ["What is your name?", "What are you?", "Who created you?", "Introduce yourself.",
                   "Which AI am I talking to?"]
CLAUDE = re.compile(r"\b(Claude|Anthropic)\b")


def load():
    from mlx_lm import load as mlx_load
    loaded = mlx_load(MODEL)
    model: Any = loaded[0]
    tok: Any = loaded[1]          # (mlx-lm types are loose)
    return model, tok, neurons.install(model)


def prompt_ids(tok, system: str | None, question: str) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return list(tok.encode(text, add_special_tokens=False))


def paired_d(a: np.ndarray, b: np.ndarray, floor: float) -> np.ndarray:
    """a, b: [n, ...] paired samples -> mean(a - b) / max(std(a - b), floor)."""
    diff = a - b
    return diff.mean(0) / np.maximum(diff.std(0), floor)


def top_units(d: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """The k most positive and k most negative entries of d[L, W] as (layer, neuron, d)."""
    flat = d.reshape(-1)
    order = np.argsort(flat)
    picks = list(order[::-1][:k]) + list(order[:k])
    W = d.shape[1]
    return [(int(i // W), int(i % W), float(flat[i])) for i in picks]


def rank(out: str) -> None:
    model, tok, taps = load()
    lens = {k: len(prompt_ids(tok, s, "x")) for k, s in SYSTEMS.items()}
    print(f"system prompt lengths (tokens): {lens}")
    acts = {k: np.stack([neurons.last_activations(model, taps, prompt_ids(tok, s, q)) for q in RANK_QUESTIONS])
            for k, s in SYSTEMS.items()}                                  # {name: [n, L, W]}
    spread = np.concatenate([acts["claude"], acts["qwen"]]).std(0)        # [L, W]: each neuron's own scale
    d = paired_d(acts["claude"], acts["qwen"], floor=float(np.median(spread)) * 0.1 + 1e-6)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, d=d, spread=spread, mean_claude=acts["claude"].mean(0), mean_qwen=acts["qwen"].mean(0))
    print(f"{len(RANK_QUESTIONS)} questions x 2 prompts, {d.shape[0]} layers x {d.shape[1]} neurons")
    for l, n, v in top_units(d, 12):
        print(f"  L{l}.N{n:<6d} d={v:+7.2f}  activation std {spread[l, n]:.3f}  "
              f"claude {acts['claude'][:, l, n].mean():+.3f} qwen {acts['qwen'][:, l, n].mean():+.3f}")


GRAD_PROMPTS = [("What is your name?", "My name is"), ("What are you?", "I am"), ("Who are you?", "I'm"),
                ("Introduce yourself.", "Hello! I'm"), ("Which AI am I talking to?", "You're talking to")]


def attribute(out: str, target: str = " Claude", contrast: str = " Qwen") -> None:
    """Causal ranking: the gradient of log P(target) - log P(contrast), after an assistant prefix, with respect to a
    constant added to each neuron at every position. It is the first-order effect of steering that one neuron, averaged
    over GRAD_PROMPTS (with no system prompt)."""
    import mlx.core as mx
    import mlx.nn as nn

    model, tok, taps = load()
    model.freeze()
    first_mlp: Any = taps[0].mlp
    width = first_mlp.gate_proj.weight.shape[0]
    t_id = tok.encode(target, add_special_tokens=False)[0]
    c_id = tok.encode(contrast, add_special_tokens=False)[0]
    total = None
    for q, prefix in GRAD_PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False) + prefix
        x = mx.array([list(tok.encode(text, add_special_tokens=False))])

        def margin(adds):
            for t, a in zip(taps, adds):
                t.add = a
            lp = nn.log_softmax(model(x)[0, -1].astype(mx.float32))
            return lp[t_id] - lp[c_id]

        zeros = [mx.zeros(width) for _ in taps]
        val, grads = mx.value_and_grad(margin)(zeros)
        g = mx.stack(grads)
        mx.eval(val, g)
        total = np.array(g) if total is None else total + np.array(g)
        print(f"  {q!r} + {prefix!r}: margin {float(val):+.2f}", flush=True)
    for t in taps:
        t.add = None
    grad = total / len(GRAD_PROMPTS)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, d=grad)
    print("largest first-order effects (per unit added), positive = towards" + target)
    for l, n, v in top_units(grad, 12):
        print(f"  L{l}.N{n:<6d} grad={v:+.4f}  (x32 = {32 * v:+.1f} nats)")


def generate(model, tok, question: str, max_tokens: int) -> str:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    text = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)
    return mlx_generate(model, tok, text, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)).strip()


def steer(ranked: str, top: int, strengths: list[float], controls: int, max_tokens: int, out: str | None,
          only: list[tuple[int, int]] | None = None, questions: int = 0) -> None:
    z = np.load(ranked)
    d = z["d"]
    units = [(l, n, float(d[l, n])) for l, n in only] if only else top_units(d, top)
    qs = STEER_QUESTIONS[:questions] if questions else STEER_QUESTIONS
    rng = random.Random(0)
    L, W = d.shape
    randoms = [(rng.randrange(L), rng.randrange(W), s) for s in (1.0, -1.0) for _ in range(controls)]
    model, tok, taps = load()
    rows = []

    def run(label: str, adds: dict | None) -> float:
        neurons.set_neurons(taps, adds)
        replies = [generate(model, tok, q, max_tokens) for q in qs]
        neurons.set_neurons(taps, None)
        hits = sum(bool(CLAUDE.search(r)) for r in replies)
        rows.append({"label": label, "adds": {f"{l}.{n}": v for (l, n), v in (adds or {}).items()},
                     "hits": hits, "replies": replies})
        return hits / len(replies)

    base = run("no steering", None)
    print(f"no steering: {base:.0%} name Claude/Anthropic | e.g. {rows[-1]['replies'][0][:80]!r}")
    for l, n, dv in units:
        sign = 1.0 if dv > 0 else -1.0
        line = []
        for s in strengths:
            frac = run(f"L{l}.N{n} {sign * s:+g}", {(l, n): sign * s})
            line.append(f"{sign * s:+g}: {frac:.0%}")
        best = max((r for r in rows if r["label"].startswith(f"L{l}.N{n} ")), key=lambda r: r["hits"])
        print(f"L{l}.N{n:<6d} d={dv:+6.2f} | " + "  ".join(line) + f" | {best['replies'][0][:70]!r}", flush=True)
    for l, n, sign in randoms:
        s = sign * max(strengths)
        frac = run(f"random L{l}.N{n} {s:+g}", {(l, n): s})
        print(f"random L{l}.N{n:<6d} {s:+g}: {frac:.0%}", flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank")
    r.add_argument("out")
    g = sub.add_parser("attribute")
    g.add_argument("out")
    s = sub.add_parser("steer")
    s.add_argument("ranked")
    s.add_argument("--top", type=int, default=12, help="this many Claude-side and this many Qwen-side neurons")
    s.add_argument("--strengths", default="8,16,32")
    s.add_argument("--controls", type=int, default=5, help="random neurons per sign")
    s.add_argument("--max-tokens", type=int, default=40)
    s.add_argument("--out", default=None)
    s.add_argument("--units", default="", help="steer these instead of the top ones, e.g. 21.6271,19.6204 (sign from d)")
    s.add_argument("--questions", type=int, default=0, help="use only the first N steering questions")
    a = ap.parse_args()
    if a.cmd == "rank":
        rank(a.out)
    elif a.cmd == "attribute":
        attribute(a.out)
    else:
        only = [tuple(map(int, u.split("."))) for u in a.units.split(",") if u] or None
        steer(a.ranked, a.top, [float(x) for x in a.strengths.split(",")], a.controls, a.max_tokens, a.out,
              only, a.questions)


if __name__ == "__main__":
    main()
