"""Emotion vectors for Qwen3-8B (MLX, 4-bit), and what they say about the persona models.

    python -m emotions.run extract runs/emotions/qwen3-8b.npz
    python -m emotions.run trace   runs/emotions/qwen3-8b.npz emotions/convos/clover_trust.json --adapters none,bunny
    python -m emotions.run steer   runs/emotions/qwen3-8b.npz --adapter bunny --emotion fear --alpha -4 "Hello there!"

extract: run each stimulus as "Imagine this happens to you: ...", read the residual stream at the last prompt token
(where the model is about to answer), fit difference-of-means directions on 80% and report held-out AUC per layer,
then refit on all stimuli and save.

trace: for a conversation, report each emotion's z-score per assistant turn, averaged over a band of layers, at two
places: "pre" (the last token before the reply: how the model takes the user's message) and "reply" (the mean over
the reply's tokens). A conversation file is {"user": [...]} (the replies are generated, greedily, per adapter) or
{"messages": [...]} (a fixed transcript, e.g. exported from the web chat, read under each adapter).

steer: generate with alpha x sigma x direction added to the residual stream at each layer of the band.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geometry, taps

MODEL = "mlx-community/Qwen3-8B-4bit"
ADAPTERS = "runs/gpu/runs/{}/adapter-mlx"
FRAME = "Imagine this happens to you: {}"


def load(adapter: str | None):
    from mlx_lm import load as mlx_load
    path = None if adapter in (None, "", "none") else ADAPTERS.format(adapter)
    model, tok = mlx_load(MODEL, adapter_path=path)
    return model, tok, taps.install(model)


def band(spec: str) -> list[int]:
    a, b = map(int, spec.split("-"))
    return list(range(a, b + 1))


def extract(out: str, stimuli: str) -> None:
    data = json.loads(Path(stimuli).read_text())
    emotions = [e for e in data if e != "neutral"]
    texts = [(e, s) for e, ss in data.items() for s in ss]
    model, tok, tp = load(None)
    acts = np.stack([taps.hidden(model, tp, taps.chat_ids(tok, [{"role": "user", "content": FRAME.format(s)}]))[-1]
                     for _, s in texts])                                    # [n, L, d] at the last prompt token
    labels = [e for e, _ in texts]
    train, test = geometry.split(labels)
    d = geometry.fit(acts[train], [labels[i] for i in train], emotions)
    aucs = geometry.evaluate(d, acts[test], [labels[i] for i in test])
    full = geometry.fit(acts, labels, emotions)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    full.save(out)
    report = {e: {"best_layer": int(a.argmax()), "best_auc": round(float(a.max()), 3),
                  "auc_by_layer": [round(float(x), 3) for x in a]} for e, a in aucs.items()}
    Path(out).with_suffix(".eval.json").write_text(json.dumps(report, indent=1))
    L = acts.shape[1]
    print(f"{len(texts)} stimuli, {L} layers; held-out one-vs-rest AUC (train {len(train)}, test {len(test)})")
    for e, a in aucs.items():
        print(f"  {e:8s} best layer {a.argmax():2d} AUC {a.max():.3f} | " +
              " ".join(f"{a[l]:.2f}" for l in range(0, L, 4)) + "  (every 4th layer)")


def turn(tok, messages: list[dict], i: int) -> tuple[list[int], int, int, int]:
    """Tokens for messages[:i+1] (message i is an assistant reply), with pre (the last token before the reply) and
    [start, end) (the reply's tokens). Built as the model saw it while replying: the prompt's tokens, then the reply's
    own tokens. (Rendering the whole chat and tokenising it again would merge the "\n\n" after </think> with the
    reply's first word, and each turn gets its own sequence because Qwen3 drops the think block from earlier replies.)"""
    pre_ids = taps.chat_ids(tok, messages[:i], generation_prompt=True)
    reply = tok.encode(messages[i]["content"], add_special_tokens=False)
    return pre_ids + reply, len(pre_ids) - 1, len(pre_ids), len(pre_ids) + len(reply)


def trace(vectors: str, convo: str, adapters: list[str], layers: list[int], out: str | None) -> None:
    d = geometry.Directions.load(vectors)
    spec = json.loads(Path(convo).read_text())
    rows = []
    for ad in adapters:
        model, tok, tp = load(ad)
        if "messages" in spec:
            messages = spec["messages"]
        else:                                                       # generate the replies under this adapter
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            messages = []
            for u in spec["user"]:
                messages.append({"role": "user", "content": u})
                prompt = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                                 tokenize=False)
                reply = generate(model, tok, prompt, max_tokens=160, sampler=make_sampler(temp=0.0))
                messages.append({"role": "assistant", "content": reply.strip()})
        k = 0
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            k += 1
            ids, pre, a, b = turn(tok, messages, i)
            h = taps.hidden(model, tp, ids)                         # [T, L, d]
            zp = d.project(h[pre:pre + 1])[0][:, layers].mean(-1)
            zr = d.project(h[a:b]).mean(0)[:, layers].mean(-1)
            rows.append({"adapter": ad, "turn": k, "user": messages[i - 1]["content"], "reply": m["content"],
                         "pre": dict(zip(d.emotions, map(float, zp))), "reply_z": dict(zip(d.emotions, map(float, zr)))})
        del model
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"z-scores averaged over layers {layers[0]}-{layers[-1]}; pre = before the reply, reply = over its tokens")
    head = "  ".join(f"{e[:5]:>5s}" for e in d.emotions)
    for ad in adapters:
        print(f"\n[{ad}]  turn | pre: {head} | reply: {head}")
        for r in (x for x in rows if x["adapter"] == ad):
            p = "  ".join(f"{r['pre'][e]:+5.1f}" for e in d.emotions)
            q = "  ".join(f"{r['reply_z'][e]:+5.1f}" for e in d.emotions)
            print(f"  {r['turn']:2d} | {p} | {q}   {r['user'][:40]!r} -> {r['reply'][:60]!r}")


def steer(vectors: str, adapter: str, emotion: str, alphas: list[float], layers: list[int], prompt: str,
          max_tokens: int) -> None:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    d = geometry.Directions.load(vectors)
    e = d.emotions.index(emotion)
    model, tok, tp = load(adapter)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)
    for a in sorted({0.0, *alphas}):
        taps.set_steering(tp, {l: a * d.sigma[e, l] * d.vectors[e, l] for l in layers} if a else None)
        out = generate(model, tok, text, max_tokens=max_tokens, sampler=make_sampler(temp=0.0))
        print(f"[{adapter} | {emotion} {a:+g} sigma @ layers {layers[0]}-{layers[-1]}]\n{out.strip()}\n")
    taps.set_steering(tp, None)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract")
    x.add_argument("out")
    x.add_argument("--stimuli", default="emotions/stimuli.json")
    t = sub.add_parser("trace")
    t.add_argument("vectors")
    t.add_argument("convo")
    t.add_argument("--adapters", default="none,bunny")
    t.add_argument("--layers", default="14-22")
    t.add_argument("--out", default=None)
    s = sub.add_parser("steer")
    s.add_argument("vectors")
    s.add_argument("prompt")
    s.add_argument("--adapter", default="bunny")
    s.add_argument("--emotion", default="fear")
    s.add_argument("--alpha", default="-2", help="one or more strengths in sigmas, comma-separated, e.g. -1,-2,-3")
    s.add_argument("--layers", default="14-22")
    s.add_argument("--max-tokens", type=int, default=120)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.out, a.stimuli)
    elif a.cmd == "trace":
        trace(a.vectors, a.convo, a.adapters.split(","), band(a.layers), a.out)
    else:
        steer(a.vectors, a.adapter, a.emotion, [float(x) for x in a.alpha.split(",")], band(a.layers), a.prompt,
              a.max_tokens)


if __name__ == "__main__":
    main()
