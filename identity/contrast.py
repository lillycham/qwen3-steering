"""Which MLP neurons carry a register? Stage A, by contrast: read every neuron on paired texts, no pushing.

    python -m identity.contrast generate runs/identity/contrast-pairs.json [--n 150]
    python -m identity.contrast measure  runs/identity/contrast-pairs.json runs/identity/contrast
    python -m identity.contrast report   runs/identity/contrast [--vs others|plain]

The second stage A (the first is the gradient score of `identity.styles`). The gradient says what a push would do to the
next token; this says which neurons differ between a plain text and the same content written in a register, over the
whole text. A neuron high on both lists is a candidate to push (stage B). A neuron high only here is more likely a
reader than a writer (the layer-10 "Claude" neurons were: they separate the two system prompts and steer nothing).

`generate`: Qwen3-8B writes a plain passage on a topic, then rewrites it once in each of REGISTERS. The pair is the unit,
so the topic cancels when the two are subtracted (an anger direction fitted on grievances may be a grievance direction;
a rewrite of the same facts cannot be). Each pair goes to train, val or test by a hash of its topic.

`measure`: for each complete pair, the signed mean activation of all 36 x 12288 neurons over the text (position 0, the
attention sink, left out), registered minus plain. The differences are kept as float16 in one .npy file per register
([pairs, 36, 12288]; about 40 MB per 100 pairs per register), because the null in `report` needs them. Resumable.

`report`: per register, a paired t-statistic per neuron on the train topics. The threshold comes from flipping the sign
of each pair's difference at random and keeping the largest |t| over all neurons (a family-wise null: with 442 thousand
neurons the top of any ranking looks striking by chance). Then each neuron over the threshold is looked at again on the
val and test topics, which the ranking never saw. `--vs others` subtracts the mean of the other registers' differences
from each pair first, so "this is not plain text" (shared by every register) drops out and what is left separates one
register from the rest. The neurons found by hand are the controls: L16.N1341 must come out for baby, the God neurons
for sermon. If they do not, the method is wrong.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

REGISTERS = {
    "baby": "baby talk, as if speaking to a toddler: simple words, sweet and silly sounds",
    "sassy": "a sassy, sarcastic voice with dry irony",
    "flirty": "a flirty, teasing and playful voice",
    "archaic": "archaic Early Modern English, in the style of Shakespeare",
    "pirate": "the speech of a swashbuckling pirate",
    "corporate": "corporate business jargon, as in a management memo",
    "texting": "casual text-message slang, with abbreviations such as rn, tbh, ngl and lol, and emoticons",
    "sermon": "a solemn religious sermon that speaks of God and grace",
}
KNOWN = {                                   # found by hand: (register it should show for, layer, neuron)
    "L16.N1341": ("baby", 16, 1341),
    "L8.N1931": ("sermon", 8, 1931), "L9.N3757": ("sermon", 9, 3757), "L18.N7807": ("sermon", 18, 7807),
    "L9.N3862": ("sermon", 9, 3862), "L8.N12153": ("sermon", 8, 12153), "L14.N7933": ("sermon", 14, 7933),
    "L11.N9192": ("sermon", 11, 9192),
}
TOPICS = [
    "a train timetable", "a filing cabinet", "making a pot of tea", "a library's borrowing rules", "how a bicycle gear works",
    "sorting laundry", "planting spring bulbs", "reading a thermometer", "a parking permit", "a bus route map",
    "boiling an egg", "changing a light bulb", "a postal service's parcel sizes", "how a kettle switches itself off",
    "a recycling collection calendar", "the layout of a supermarket", "folding a paper aeroplane", "a school's term dates",
    "storing potatoes", "how a zip fastener works", "the opening hours of a post office", "setting a table",
    "a tape measure", "the water cycle", "how a lift is called to a floor", "tying a shoelace", "a recipe for plain rice",
    "a washing machine's programme dial", "the parts of a ring binder", "watering houseplants",
    "a village noticeboard", "how a clock keeps time", "the metric prefixes", "wrapping a parcel", "a car park's pay machine",
    "how a map's scale works", "sharpening a pencil", "an office stapler", "a calendar's leap years", "making a bed",
    "a swimming pool's lane rules", "how a lock and key work", "defrosting a freezer", "the stages of a rock's formation",
    "a museum's cloakroom", "assembling flat-pack shelves", "how a thermostat works", "the contents of a first aid box",
    "a tide table", "installing a software update",
]
FORMS = {
    "how-to": "step-by-step instructions in the second person",
    "description": "a plain description of how the thing is built or arranged",
    "notice": "a notice of the kind posted in a public building",
}
PLAIN_PROMPT = ("Write a short passage about {topic}, as {form}. Use two or three sentences, between 25 and 50 words in "
                "all. Use plain factual wording: no opinions, no feelings, no story and no questions. Reply with the "
                "passage only.")
REWRITE_PROMPT = ("Rewrite the passage below in {register}. Change the wording and the tone all the way through, not just "
                  "the ending. Keep every fact and the same number of sentences, add nothing, and do not make it much longer. Reply with the rewritten "
                  "passage only, with no title and no comment.\n\nPassage: {passage}")
MAX_OVERLAP = 0.7           # a rewrite sharing more of its words than this with the plain text did not change register
SPLITS = [("train", 70), ("val", 80), ("test", 100)]            # (name, up to this percentile of the topic hash)
HELD = ("val", "test")
META = re.compile(r"^(here('s| is)|sure|certainly|okay|of course|rewritten|passage:)|\b(rewrite|rewritten|passage)\b", re.I)


# ---- the pairs ----

def split_of(key: str) -> str:
    """train, val or test (70/10/20), from a hash of the key alone (the topic of a pair)."""
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return next(name for name, upto in SPLITS if bucket < upto)


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.strip('"').strip()


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' word sets (1 = the same words)."""
    wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
    return len(wa & wb) / max(len(wa | wb), 1)


def reject_reason(text: str, plain: str | None = None) -> str | None:
    """Why a generated text is unusable, or None. With `plain`, the text is a rewrite of it."""
    words = len(text.split())
    if META.search(text):
        return "meta talk"
    if plain is None:
        return "length" if not 20 <= words <= 70 else None
    n = len(plain.split())
    if not 0.6 * n <= words <= 3.0 * n:
        return "length"
    if overlap(text, plain) > MAX_OVERLAP:
        return "barely changed"
    return None


def prompt_for(i: int) -> tuple[str, str, str]:
    """(topic, form, the request) for the i-th attempt. Topics cycle fastest, then forms."""
    topic = TOPICS[i % len(TOPICS)]
    form = list(FORMS)[(i // len(TOPICS)) % len(FORMS)]
    return topic, form, PLAIN_PROMPT.format(topic=topic, form=FORMS[form])


def complete(pairs: list[dict]) -> list[dict]:
    """The pairs that have a rewrite for every register (the others cannot be compared with each other)."""
    return [p for p in pairs if set(p["regs"]) == set(REGISTERS)]


def generate(out: str, n: int, seed: int) -> None:
    import mlx.core as mx
    from mlx_lm import generate as lm_generate
    from mlx_lm.sample_utils import make_sampler

    from .search import load

    model, tok, _ = load()
    mx.random.seed(seed)
    sampler = make_sampler(temp=0.8, top_p=0.95)

    def ask(request: str, max_tokens: int = 160) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": request}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        return clean(lm_generate(model, tok, prompt=text, max_tokens=max_tokens, sampler=sampler))

    path = Path(out)
    state: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {"pairs": [], "rejected": {}, "attempts": 0}
    pairs, why = state["pairs"], state["rejected"]
    t0 = time.time()

    def save() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**state, "seed": seed}, indent=1))

    while len(pairs) < n and state["attempts"] < 3 * n:
        topic, form, request = prompt_for(state["attempts"])
        state["attempts"] += 1
        plain = ask(request)
        reason = reject_reason(plain)
        if reason:
            why[f"plain: {reason}"] = why.get(f"plain: {reason}", 0) + 1
            continue
        regs: dict[str, str] = {}
        for name, how in REGISTERS.items():
            text = ask(REWRITE_PROMPT.format(register=how, passage=plain))
            reason = reject_reason(text, plain)
            if reason:
                why[f"{name}: {reason}"] = why.get(f"{name}: {reason}", 0) + 1
                break                                                    # (an incomplete pair is no use: drop it)
            regs[name] = text
        else:
            pairs.append({"topic": topic, "form": form, "split": split_of(topic), "plain": plain, "regs": regs})
        if len(pairs) % 5 == 0 or state["attempts"] % 10 == 0:
            save()
            print(f"[{time.time() - t0:5.0f}s] {len(pairs)} pairs from {state['attempts']} attempts; rejected {why}",
                  flush=True)
    save()
    print(f"wrote {out}: {len(pairs)} pairs from {state['attempts']} attempts")


# ---- the statistics (numpy only) ----

def tstat(d: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """d [n, ...] paired differences -> the t-statistic of their mean, per entry. A constant entry gets 0."""
    n = d.shape[0]
    m = d.mean(0)
    sd = np.sqrt(np.maximum((np.square(d).sum(0) - n * m * m) / (n - 1), 0.0))
    return m / np.maximum(sd / np.sqrt(n), floor)


def layer_diffs(D: dict[str, Any], reg: str, layer: int, rows: np.ndarray, vs: str) -> np.ndarray:
    """[len(rows), W] float32: register minus plain for one layer; with vs="others", also minus the mean of the other
    registers' differences of the same pair."""
    d = np.asarray(D[reg][rows, layer], np.float32)
    if vs == "others" and len(D) > 1:
        rest = sum(np.asarray(D[r][rows, layer], np.float32) for r in D if r != reg) / (len(D) - 1)
        d = d - rest
    return d


def t_map(D: dict[str, Any], reg: str, rows: np.ndarray, vs: str) -> np.ndarray:
    """[L, W] t-statistic of the register's difference over the given pairs."""
    n_layers = next(iter(D.values())).shape[1]
    return np.stack([tstat(layer_diffs(D, reg, l, rows, vs)) for l in range(n_layers)])


def null_threshold(D: dict[str, Any], reg: str, rows: np.ndarray, vs: str, min_layer: int, flips: int = 200,
                   q: float = 0.95, seed: int = 0) -> float:
    """The q-quantile, over random sign flips of whole pairs, of the largest |t| over every neuron from min_layer up.
    A flip does not change the sum of squares, only the mean, so each flip costs one matrix product per layer."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    signs = rng.choice([-1.0, 1.0], size=(flips, n)).astype(np.float32)
    best = np.zeros(flips)
    n_layers = next(iter(D.values())).shape[1]
    for l in range(min_layer, n_layers):
        d = layer_diffs(D, reg, l, rows, vs)
        ss = np.square(d).sum(0)
        m = (signs @ d) / n                                                   # [flips, W]
        sd = np.sqrt(np.maximum((ss - n * m * m) / (n - 1), 0.0))
        t = m / np.maximum(sd / np.sqrt(n), 1e-6)
        best = np.maximum(best, np.abs(t).max(axis=1))
    return float(np.quantile(best, q))


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


def top_k(a: np.ndarray, k: int, min_layer: int) -> list[tuple[int, int]]:
    """The k entries of a[L, W] with the largest |value|, from layer min_layer up, as (layer, neuron)."""
    mag = np.abs(a).astype(np.float64)
    mag[:min_layer] = -1.0
    flat = np.argpartition(-mag.reshape(-1), k)[:k]
    flat = flat[np.argsort(-mag.reshape(-1)[flat])]
    return [(int(i // a.shape[1]), int(i % a.shape[1])) for i in flat]


def replication(t_train: np.ndarray, t_held: np.ndarray, threshold: float, min_layer: int, t_min: float = 2.0
                ) -> tuple[int, int]:
    """(neurons over the threshold on the train topics, those among them with the same sign and |t| >= t_min on the
    held-out topics). By chance a held-out |t| >= 2 is about 5%, and the sign is a coin flip."""
    over = (np.abs(t_train) > threshold) & (np.arange(t_train.shape[0])[:, None] >= min_layer)
    same = over & (np.sign(t_held) == np.sign(t_train)) & (np.abs(t_held) >= t_min)
    return int(over.sum()), int(same.sum())


# ---- running ----

def measure(pairs_path: str, out_dir: str) -> None:
    import mlx.core as mx

    from . import neurons
    from .search import load

    mx.set_cache_limit(1 << 30)                       # (texts of many lengths fill the default cache with odd buffers)
    pairs = complete(json.loads(Path(pairs_path).read_text())["pairs"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, taps = load()
    width = neurons.mlp_width(taps[0].mlp)
    shape = (len(pairs), len(taps), width)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    resume = meta.get("n") == len(pairs) and meta.get("registers") == list(REGISTERS)
    D = {r: np.lib.format.open_memmap(out / f"{r}.npy", mode="r+" if resume else "w+", dtype=np.float16, shape=shape)
         for r in REGISTERS}
    done = meta["done"] if resume else 0
    meta = {"n": len(pairs), "registers": list(REGISTERS), "done": done,
            "splits": [p["split"] for p in pairs], "topics": [p["topic"] for p in pairs]}
    t0 = time.time()

    def acts(text: str) -> np.ndarray:
        return neurons.mean_activations(model, taps, list(tok.encode(text)))

    print(f"{len(pairs)} complete pairs x {len(REGISTERS)} registers; resuming at {done}", flush=True)
    for i in range(done, len(pairs)):
        base = acts(pairs[i]["plain"])
        for r in REGISTERS:
            D[r][i] = (acts(pairs[i]["regs"][r]) - base).astype(np.float16)
        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            for m in D.values():
                m.flush()
            meta["done"] = i + 1
            meta_path.write_text(json.dumps(meta))
            print(f"[{time.time() - t0:5.0f}s] pair {i + 1}/{len(pairs)}", flush=True)
    print(f"wrote {out_dir}")


def load_diffs(path: str) -> tuple[dict[str, Any], dict]:
    meta = json.loads((Path(path) / "meta.json").read_text())
    done = meta["done"]
    D = {r: np.load(Path(path) / f"{r}.npy", mmap_mode="r")[:done] for r in meta["registers"]}
    meta["splits"], meta["topics"] = meta["splits"][:done], meta["topics"][:done]
    return D, meta


def report(path: str, vs: str, min_layer: int, top: int, flips: int) -> None:
    D, meta = load_diffs(path)
    splits = np.array(meta["splits"])
    train = np.where(splits == "train")[0]
    held = np.where(np.isin(splits, HELD))[0]
    if len(train) < 3 or len(held) < 3:
        print(f"{len(splits)} pairs measured: {len(train)} train and {len(held)} held out. Need at least 3 of each.")
        return
    n_train_topics = len(set(np.array(meta["topics"])[train]))
    print(f"{len(splits)} pairs: {len(train)} train ({n_train_topics} topics), {len(held)} held out "
          f"(val + test, other topics); registers {list(D)}; contrast {'with the other registers' if vs == 'others' else 'with plain'}; "
          f"layers {min_layer}+\n")
    for reg in D:
        t_tr, t_he = t_map(D, reg, train, vs), t_map(D, reg, held, vs)
        thr = null_threshold(D, reg, train, vs, min_layer, flips)
        over, same = replication(t_tr, t_he, thr, min_layer)
        print(f"== {reg} == threshold |t| {thr:.1f} (95% of {flips} sign-flip nulls); {over} neurons over it on train, "
              f"{same} of them replicate on the held-out topics")
        for l, k in top_k(t_tr, top, min_layer):
            print(f"  L{l}.N{k:<6d} t train {t_tr[l, k]:+6.1f}   held {t_he[l, k]:+6.1f}")
        ranks = flat_ranks(t_tr, min_layer)
        total = t_tr[min_layer:].size
        for label, (want, l, k) in KNOWN.items():
            if want == reg:
                print(f"  control {label}: rank {ranks[l, k]} of {total} (top {100 * ranks[l, k] / total:.3f}%), "
                      f"t train {t_tr[l, k]:+.1f}, held {t_he[l, k]:+.1f}")
        print()


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out")
    g.add_argument("--n", type=int, default=150, help="complete pairs to keep")
    g.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("measure")
    m.add_argument("pairs")
    m.add_argument("out")
    r = sub.add_parser("report")
    r.add_argument("path")
    r.add_argument("--vs", choices=["others", "plain"], default="others")
    r.add_argument("--min-layer", type=int, default=6)
    r.add_argument("--top", type=int, default=8)
    r.add_argument("--flips", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.out, a.n, a.seed)
    elif a.cmd == "measure":
        measure(a.pairs, a.out)
    else:
        report(a.path, a.vs, a.min_layer, a.top, a.flips)


if __name__ == "__main__":
    main()
