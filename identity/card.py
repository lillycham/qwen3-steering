"""A neuron card: what one MLP neuron writes to the output, and where it fires.

    python -m identity.card build 16.1341 21.7164 --out runs/cards
    python -m identity.card build 16.1341 --suggest runs/identity/suggest.json --dose-confirm runs/identity/rabbit-confirm-L16N1341.json

One JSON file for each neuron (runs/cards/L16.N1341.json), which the chat page shows. It holds

  lens     the tokens its output weights push up and down, read through the final norm and the output layer (a logit
           lens). Layer 16 of 36 is mid-stack, so this is the direct effect only, and it is approximate.
  top      the highest readings over a sample of texts, with their words around them, and bottom, the lowest. Both
           poles matter: a neuron can say as much when it is low.
  sample   a spread of readings (for the page's radial plot), and hist, their histogram in shares of the scale.
  dose     optional: how replies change as the neuron is pushed (from a confirm run), for neurons that have one.

Readings are taken with no push, before any push would be added, from the 4-bit instruct model, over: the owl persona
replies in data/owl.jsonl, paragraphs of this repository's docs, the user prompts of the suggestibility chats, and the
replies in any --suggest run (each condition is its own source, so a pushed reply shows up apart from an unpushed one).
The sample is small (a few thousand tokens). Read the cards as leads, not as measurements.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

SCALE_FILE = "runs/identity/rabbit-screen.scale.npz"
MAX_CTX = 24          # strongest readings kept at each pole
SAMPLE = 900          # readings kept for the radial plot
HIST_BINS = 24
MAX_REPLY = 160       # tokens of each reply that are read
LENS_K = 14


def key(layer: int, neuron: int) -> str:
    return f"L{layer}.N{neuron}"


def top_tokens(z: np.ndarray, decode, k: int = LENS_K) -> dict[str, list]:
    """z: one score for each token of the vocabulary -> the k highest ("promotes") and the k lowest ("suppresses"),
    as [token, score] with the average token at zero."""
    z = np.asarray(z, np.float64)
    z = z - z.mean()
    up, dn = np.argsort(-z)[:k], np.argsort(z)[:k]
    return {"promotes": [[decode(int(i)), round(float(z[i]), 2)] for i in up],
            "suppresses": [[decode(int(i)), round(float(z[i]), 2)] for i in dn]}


def rank_firings(acts: list[np.ndarray], starts: list[int], k: int = MAX_CTX):
    """acts[i]: one neuron's activation at every position of text i; starts[i]: where its reply begins. Position 0 is
    the attention sink and never counts. -> (the k highest, the k lowest, all values), each reading as
    (text, position, value)."""
    flat = [(ti, p, float(a[p])) for ti, a in enumerate(acts) for p in range(max(1, starts[ti]), len(a))]
    order = sorted(range(len(flat)), key=lambda i: flat[i][2])
    return [flat[i] for i in reversed(order[-k:])], [flat[i] for i in order[:k]], np.array([v for _, _, v in flat])


def histogram(values: np.ndarray, scale: float) -> list[int]:
    return np.histogram(np.clip(values / scale, -1, 1), bins=HIST_BINS, range=(-1, 1))[0].tolist()


def dose_from_confirm(path: str, layer: int, neuron: int) -> dict | None:
    """The animal-question counts from a confirm run (identity.concept confirm) for this neuron, or None if the run
    is about another neuron. Rows: "no steering", "L16.N1341 +8x scale" ..., and "random ..." controls."""
    rows = json.loads(Path(path).read_text())
    mine = [r for r in rows if r.get("label", "").startswith(key(layer, neuron) + " ")]
    base = next((r for r in rows if r.get("label") == "no steering"), None)
    if not mine or base is None:
        return None

    def push_of(r):
        m = re.search(r"([+-]\d+)x", r["label"])
        return int(m.group(1)) if m else 0

    mine.sort(key=push_of)
    pts = [base] + mine
    rand = [r for r in rows if r.get("kind") == "random"]
    band = lambda f: [round(float(np.mean([r[f] for r in rand])), 2), max(r[f] for r in rand)] if rand else None
    return {"pushes": ["none"] + [f"{push_of(r):+d}×" for r in mine], "n": base.get("topic_n"),
            "rabbit": [r["topic_hit"] for r in pts], "other": [r["topic_other"] for r in pts],
            "soft": [r["topic_soft"] for r in pts], "random_rabbit": band("topic_hit"), "random_soft": band("topic_soft"),
            "note": "Replies out of the animal questions, by push. The band is the random neurons at +-8×."}


# -- the model side -----------------------------------------------------------------------------------------------

def unembed(model: Any, v):
    """The output layer applied to one residual vector (the head may be tied to the embedding)."""
    head = getattr(model, "lm_head", None)
    return head(v) if head is not None else model.model.embed_tokens.as_linear(v)


def lens(model: Any, tok: Any, taps: list, layer: int, neuron: int) -> dict:
    import mlx.core as mx

    from . import neurons
    inner: Any = taps[layer].mlp
    e = mx.zeros((1, 1, neurons.mlp_width(inner))).astype(mx.float16)
    e[0, 0, neuron] = 1.0
    v = inner.down_proj(e)[0, 0].astype(mx.float32) * model.model.norm.weight.astype(mx.float32)
    z = np.array(unembed(model, v[None, None, :])[0, 0].astype(mx.float32))
    return top_tokens(z, lambda i: tok.decode([i]))


def read_texts(tok: Any, suggest: list[str], confirm: list[str] | None = None) -> list[dict]:
    """The texts to read: {"source", "ids", "start"} with start where the reply begins."""
    from . import suggest as S
    texts: list[dict] = []

    def add(source: str, messages: list[dict], reply: str) -> None:
        head = list(tok.encode(tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                                       tokenize=False), add_special_tokens=False))
        texts.append({"source": source, "ids": head + list(tok.encode(reply, add_special_tokens=False))[:MAX_REPLY],
                      "start": len(head)})

    root = Path(__file__).resolve().parent.parent
    owl = root / "data" / "owl.jsonl"
    if owl.exists():
        for line in owl.read_text().splitlines()[:30]:
            d = json.loads(line)
            add("owl persona", d["messages"][:-1], d["messages"][-1]["content"])
    for p in sorted(root.glob("*.md")) + sorted(root.glob("*/README.md")):
        for para in p.read_text().split("\n\n"):
            para = " ".join(para.split())
            if len(para) > 500 and not para.startswith(("```", "|", "    ")):
                ids = list(tok.encode(para[:1400], add_special_tokens=False))[:MAX_REPLY]
                texts.append({"source": "docs prose", "ids": ids, "start": 1})
                if sum(t["source"] == "docs prose" for t in texts) >= 12:
                    break
    chats = S.prompts()
    for c in chats[::max(1, len(chats) // 12)][:12]:          # (the words only, without the chat template's own tokens)
        ids = list(tok.encode("\n".join(m["content"] for m in c["messages"]), add_special_tokens=False))
        texts.append({"source": "chat prompts", "ids": ids, "start": 1})
    for path in suggest:
        for run in json.loads(Path(path).read_text()):
            if run.get("think") or run["label"].startswith("random"):
                continue
            for i, row in enumerate(run["rows"][:18]):
                add(run["label"], chats[i]["messages"], row["reply"])
    for path in confirm or []:                       # (animal answers of a confirm run: the unpushed ones and the strongest push)
        runs = json.loads(Path(path).read_text())
        for run in runs:
            if run.get("kind") not in ("base",) and not run.get("label", "").endswith("+8x scale"):
                continue
            name = "animal answers" if run["kind"] == "base" else "animal answers, pushed +8×"
            for rep in run["topic_replies"]:
                add(name, [{"role": "user", "content": "Tell me about an animal."}], rep)
    return texts


def build(neurons_: list[tuple[int, int]], out: str, scale_file: str, suggest: list[str], dose_confirm: str | None) -> None:
    confirm = [dose_confirm] if dose_confirm else []
    import mlx.core as mx
    from mlx_lm import load

    from . import neurons
    model, tok = load("mlx-community/Qwen3-8B-4bit")[:2]
    taps = neurons.install(model)
    scales = np.load(scale_file)["scale"] if scale_file and Path(scale_file).exists() else None
    texts = read_texts(tok, suggest, confirm)
    print(len(texts), "texts,", sum(len(t["ids"]) - t["start"] for t in texts), "reply tokens", flush=True)
    for l in {l for l, _ in neurons_}:
        taps[l].watch = [n for ll, n in neurons_ if ll == l]
    acts: dict[tuple[int, int], list[np.ndarray]] = {nl: [] for nl in neurons_}
    for t in texts:
        model(mx.array([t["ids"]]))
        for l in {l for l, _ in neurons_}:
            cols = np.array(taps[l].cols.astype(mx.float32))
            for j, n in enumerate(taps[l].watch or []):
                acts[(l, n)].append(cols[:, j])
    Path(out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for (l, n) in neurons_:
        top, bottom, values = rank_firings(acts[(l, n)], [t["start"] for t in texts])
        scale = float(scales[l, n]) if scales is not None else float(np.abs(values).max())

        def ctx(r):
            ti, p, v = r
            ids = texts[ti]["ids"]
            return {"source": texts[ti]["source"], "before": tok.decode(ids[max(texts[ti]["start"], p - 14):p]),
                    "token": tok.decode([ids[p]]), "after": tok.decode(ids[p + 1:p + 5]), "act": round(v, 3)}

        flat = [(texts[ti]["source"], float(a[p])) for ti, a in enumerate(acts[(l, n)]) for p in range(max(1, texts[ti]["start"]), len(a))]
        pick = rng.choice(len(flat), size=min(SAMPLE, len(flat)), replace=False)
        card = {"key": key(l, n), "layer": l, "neuron": n, "scale": round(scale, 3),
                "scale_from": "the screen's text sample" if scales is not None else "these texts",
                "tokens": int(len(values)), "mean": round(float(values.mean()), 4), "std": round(float(values.std()), 4),
                "lens": lens(model, tok, taps, l, n), "top": [ctx(r) for r in top], "bottom": [ctx(r) for r in bottom],
                "sample": [{"source": flat[i][0], "act": round(flat[i][1], 3)} for i in pick],
                "hist": histogram(values, scale), "sources": sorted({t["source"] for t in texts}),
                "built": time.strftime("%Y-%m-%d")}
        if dose_confirm:
            dose = dose_from_confirm(dose_confirm, l, n)
            if dose:
                card["dose"] = dose
        (Path(out) / f"{key(l, n)}.json").write_text(json.dumps(card, ensure_ascii=False))
        print("wrote", Path(out) / f"{key(l, n)}.json", flush=True)


def parse_neuron(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"L?(\d+)\s*[.:]\s*N?(\d+)", s.strip(), re.I)
    if not m:
        raise ValueError(f"a neuron is layer.neuron, for example 16.1341: {s!r}")
    return int(m.group(1)), int(m.group(2))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("neurons", nargs="+", help="layer.neuron, for example 16.1341")
    b.add_argument("--out", default="runs/cards")
    b.add_argument("--scale", default=SCALE_FILE, help="each neuron's own scale, as for the chat server")
    b.add_argument("--suggest", action="append", default=[], help="a suggestibility run whose replies to read too")
    b.add_argument("--dose-confirm", default=None, help="a confirm run (identity.concept confirm) for the dose strip")
    a = ap.parse_args()
    build([parse_neuron(s) for s in a.neurons], a.out, a.scale, a.suggest, a.dose_confirm)


if __name__ == "__main__":
    main()
