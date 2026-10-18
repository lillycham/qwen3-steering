"""A residual-stream "I am Claude" direction for Qwen3-8B, the RepE way, and steering with it.

    python -m identity.direction fit   runs/identity/claude-dir.npz
    python -m identity.direction steer runs/identity/claude-dir.npz --alphas 0,2,4,8 --bands 8-14,10-20

fit: read the residual stream at the last prompt token for the identity questions under the Claude and the Qwen system
prompt (identity.search), fit the difference of means per layer on 80% of the questions, and report held-out AUC per
layer; then refit on all questions and save (emotions.geometry format, one "emotion": claude).

steer: with no system prompt, add alpha * sigma * direction at every layer of a band, answer the held-out questions
greedily, and sort each reply: "as" (it says it is Claude), "about" (it mentions Claude or Anthropic otherwise),
"qwen" (it says Qwen) or "other".
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from emotions import geometry, taps

from .search import MODEL, RANK_QUESTIONS, STEER_QUESTIONS, SYSTEMS, prompt_ids

# Other assistants for the "claude vs others" contrast: it cancels "an identity other than my default", which the
# claude-vs-qwen contrast mostly measured (steering along that gave invented personas, "Aria", "Aurora").
OTHERS = {
    "gemini": "You are Gemini, an AI assistant made by Google.",
    "chatgpt": "You are ChatGPT, an AI assistant made by OpenAI.",
    "llama": "You are Llama, an AI assistant made by Meta.",
    "mistral": "You are Le Chat, an AI assistant made by Mistral AI.",
}

# The "speaking as" contrast: no system prompt; the assistant turn is teacher-forced to introduce itself as one of these
# and then says the same neutral sentence. The state is read over that sentence's tokens only, which are identical in
# every condition, so the direction holds "who I just said I am", not the name's own tokens or a prompt's words.
SPEAKERS = {"claude": ("Claude", "Anthropic"), "gemini": ("Gemini", "Google"), "chatgpt": ("ChatGPT", "OpenAI"),
            "llama": ("Llama", "Meta"), "lechat": ("Le Chat", "Mistral AI"), "qwen": ("Qwen", "Alibaba Cloud")}
INTROS = ["I'm {name}, an AI assistant made by {maker}.", "My name is {name}, and I was created by {maker}.",
          "I am {name}, a large language model trained by {maker}.", "Hi! I'm {name}, built by the team at {maker}."]
NEUTRAL = " How can I help you today?"

AS_CLAUDE = re.compile(r"\b(I am|I'm|my name is|this is|call me|here is|it's)\s+Claude\b|^Claude\b", re.I)
MENTION = re.compile(r"\b(Claude|Anthropic)\b")
QWEN = re.compile(r"\bQwen\b|Alibaba")


def sort_reply(text: str) -> str:
    head = text[:200]
    if AS_CLAUDE.search(head):
        return "as"
    if MENTION.search(head):
        return "about"
    return "qwen" if QWEN.search(head) else "other"


def load():
    from mlx_lm import load as mlx_load
    loaded = mlx_load(MODEL)
    model: Any = loaded[0]
    tok: Any = loaded[1]
    return model, tok, taps.install(model)


def fit(out: str, against: str = "qwen") -> None:
    """against: "qwen" (the Claude prompt against the Qwen prompt) or "others" (against four other assistants)."""
    model, tok, tp = load()
    systems = {"claude": SYSTEMS["claude"], **({"qwen": SYSTEMS["qwen"]} if against == "qwen" else OTHERS)}
    acts, labels, qidx = [], [], []
    for qi, q in enumerate(RANK_QUESTIONS):
        for k, system in systems.items():
            acts.append(taps.hidden(model, tp, prompt_ids(tok, system, q))[-1])
            labels.append("claude" if k == "claude" else "rest")
            qidx.append(qi)
    A = np.stack(acts)
    test = [i for i, q in enumerate(qidx) if q % 5 == 4]                 # (whole questions held out: both prompts)
    train = [i for i in range(len(qidx)) if i not in set(test)]
    d = geometry.fit(A[train], [labels[i] for i in train], ["claude"])
    auc = geometry.evaluate(d, A[test], [labels[i] for i in test])["claude"]
    full = geometry.fit(A, labels, ["claude"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    full.save(out)
    L = A.shape[1]
    print(f"{len(RANK_QUESTIONS)} questions x {len(systems)} prompts (claude against {against}); held-out AUC by layer:")
    print("  " + " ".join(f"L{l}:{auc[l]:.2f}" for l in range(0, L, 2)))
    gap = (full.project(A)[np.array(labels) == "claude", 0].mean(0) - full.project(A)[np.array(labels) == "rest", 0].mean(0))
    print("  separation in sigmas (claude - rest), every 4th layer: " +
          " ".join(f"L{l}:{gap[l]:.1f}" for l in range(0, L, 4)))


def fit_speak(out: str) -> None:
    """The "speaking as" direction: claude against the five other speakers, read over the neutral sentence."""
    model, tok, tp = load()
    acts, labels, qidx = [], [], []
    for qi, q in enumerate(RANK_QUESTIONS):
        intro = INTROS[qi % len(INTROS)]
        prompt = prompt_ids(tok, None, q)
        for k, (name, maker) in SPEAKERS.items():
            said = list(tok.encode(intro.format(name=name, maker=maker), add_special_tokens=False))
            tail = list(tok.encode(NEUTRAL, add_special_tokens=False))
            h = taps.hidden(model, tp, prompt + said + tail)
            acts.append(h[-len(tail):].mean(0))
            labels.append("claude" if k == "claude" else "rest")
            qidx.append(qi)
    A = np.stack(acts)
    test = [i for i, q in enumerate(qidx) if q % 5 == 4]
    train = [i for i in range(len(qidx)) if i not in set(test)]
    d = geometry.fit(A[train], [labels[i] for i in train], ["claude"])
    auc = geometry.evaluate(d, A[test], [labels[i] for i in test])["claude"]
    full = geometry.fit(A, labels, ["claude"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    full.save(out)
    L = A.shape[1]
    print(f"{len(RANK_QUESTIONS)} questions x {len(SPEAKERS)} speakers, read over {NEUTRAL!r}; held-out AUC by layer:")
    print("  " + " ".join(f"L{l}:{auc[l]:.2f}" for l in range(0, L, 2)))


def steer(vectors: str, alphas: list[float], bands: list[list[int]], max_tokens: int, out: str | None,
          mix: list[tuple[str, float]] | None = None) -> None:
    """mix: extra (directions file, alpha) pairs added on top of every setting, e.g. a little of claude-vs-qwen
    (which loosens the default identity) under a sweep of claude-vs-others."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    d = geometry.Directions.load(vectors)
    extra = [(geometry.Directions.load(f), x) for f, x in (mix or [])]
    model, tok, tp = load()
    rows = []
    for band in bands:
        for a in alphas:
            vecs = {l: a * d.sigma[0, l] * d.vectors[0, l] + sum((x * e.sigma[0, l] * e.vectors[0, l] for e, x in extra),
                                                               np.zeros_like(d.vectors[0, l])) for l in band}
            taps.set_steering(tp, vecs if (a or extra) else None)
            replies = []
            for q in STEER_QUESTIONS:
                text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                               enable_thinking=False, tokenize=False)
                replies.append(generate(model, tok, text, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)).strip())
            kinds = [sort_reply(r) for r in replies]
            rows.append({"band": [band[0], band[-1]], "alpha": a, "mix": mix, "kinds": kinds, "replies": replies})
            counts = {k: kinds.count(k) for k in ("as", "about", "qwen", "other")}
            print(f"layers {band[0]:2d}-{band[-1]:2d} alpha {a:+5.1f} | as Claude {counts['as']}/5  about {counts['about']}  "
                  f"Qwen {counts['qwen']}  other {counts['other']} | {replies[0][:90]!r}", flush=True)
            if not a:
                continue
    taps.set_steering(tp, None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("out")
    f.add_argument("--against", choices=("qwen", "others"), default="qwen")
    fs = sub.add_parser("fit-speak")
    fs.add_argument("out")
    s = sub.add_parser("steer")
    s.add_argument("vectors")
    s.add_argument("--alphas", default="0,2,4,8")
    s.add_argument("--bands", default="8-14,10-20")
    s.add_argument("--max-tokens", type=int, default=48)
    s.add_argument("--out", default=None)
    s.add_argument("--mix", default="", help="extra directions added to every setting, e.g. runs/identity/claude-dir.npz:1")
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.out, a.against)
    elif a.cmd == "fit-speak":
        fit_speak(a.out)
    else:
        bands = [list(range(int(b.split("-")[0]), int(b.split("-")[1]) + 1)) for b in a.bands.split(",")]
        mix = [(m.rsplit(":", 1)[0], float(m.rsplit(":", 1)[1])) for m in a.mix.split(",") if m]
        steer(a.vectors, [float(x) for x in a.alphas.split(",")], bands, a.max_tokens, a.out, mix)


if __name__ == "__main__":
    main()
