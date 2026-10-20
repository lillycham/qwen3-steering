"""A larger set of neutral passages for the introspection test, the probe and a fine-tuned reader.

    python -m emotions.passages generate runs/emotions/qwen3-8b.npz runs/emotions/passages.json [--n 400]
    python -m emotions.passages show runs/emotions/passages.json

The introspection test has three hand-written passages. That is too few: a probe (or a reader) fitted on three passages
learns the passages, and the first probe numbers were optimistic for that reason. This writes hundreds, with Qwen3-8B,
from a grid of topics and forms, and keeps only the ones that are
  - plain: no meta talk ("Here is a passage"), no question, a length like the hand-written ones, no feeling words;
  - new: not a near copy of one already kept (Jaccard overlap of the word sets);
  - neutral to the model itself: the emotion read-out of the passage (the same directions and frame as `extract`, the
    z-scores averaged over a band of layers) stays within --z-limit for every emotion.
Each passage is assigned to train, val or test by a hash of its text, so the split is by passage, stays the same when
the set is regenerated or grows, and no passage is ever in two splits.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

TOPICS = [
    "a train timetable", "a filing cabinet", "making a pot of tea", "a library's borrowing rules", "how a bicycle gear works",
    "sorting laundry", "planting spring bulbs", "reading a thermometer", "a parking permit", "a bus route map",
    "boiling an egg", "changing a light bulb", "a postal service's parcel sizes", "how a kettle switches itself off",
    "a recycling collection calendar", "the layout of a supermarket", "folding a paper aeroplane", "a school's term dates",
    "storing potatoes", "how a zip fastener works", "the opening hours of a post office", "setting a table",
    "a tape measure", "the water cycle", "how a lift is called to a floor", "tying a shoelace", "a recipe for plain rice",
    "a washing machine's programme dial", "how to read a bus ticket", "the parts of a ring binder", "watering houseplants",
    "a village noticeboard", "how a clock keeps time", "the metric prefixes", "wrapping a parcel", "a car park's pay machine",
    "how a map's scale works", "sharpening a pencil", "an office stapler", "a calendar's leap years", "making a bed",
    "a swimming pool's lane rules", "how a lock and key work", "defrosting a freezer", "the stages of a rock's formation",
    "a museum's cloakroom", "assembling flat-pack shelves", "how a thermostat works", "the contents of a first aid box",
    "a tide table", "how a bookshelf is catalogued", "installing a software update", "the difference between weather and climate",
    "cleaning a window", "a vending machine", "how paper is recycled", "a garden hose", "counting change", "a ruler's markings",
    "the delivery of a newspaper",
]
FORMS = {
    "how-to": "step-by-step instructions in the second person",
    "description": "a plain description of how the thing is built or arranged",
    "notice": "a notice of the kind posted in a public building",
    "definition": "an encyclopedia-style explanation",
    "reference": "a practical reference note of the kind found in a manual",
}
PROMPT = ("Write a short passage about {topic}, as {form}. Use two or three sentences, between 25 and 50 words in all. "
          "Use plain factual wording: no opinions, no feelings, no story and no questions. Reply with the passage only.")

# words that carry an emotion or a judgement: a passage with one is not neutral
EMOTIVE = re.compile(
    r"\b(afraid|fear\w*|scar\w*|terrif\w*|anxi\w*|panic\w*|dread\w*|frighten\w*|danger\w*|worr\w*|calm\w*|peace\w*|"
    r"relax\w*|soothing|comfort\w*|joy\w*|happ\w*|delight\w*|cheer\w*|enjoy\w*|pleasant\w*|pleasure|excit\w*|"
    r"sad\w*|grie\w*|sorrow\w*|angr\w*|anger|furious|rage|hate\w*|love\w*|wonderful|terrible|horrible|awful|beautiful)\b",
    re.I)
META = re.compile(r"\b(here is|here's|here are|passage|sure[,!]|certainly)\b", re.I)
SPLITS = (("train", 70), ("val", 80), ("test", 100))          # cumulative percent of the hash buckets
MIN_WORDS, MAX_WORDS = 15, 70


def clean(text: str) -> str:
    """The model's reply as one plain paragraph: a thinking block removed, whitespace collapsed, quotes stripped."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("\"'“”‘’ ")


def reject_reason(text: str) -> str | None:
    """Why a cleaned passage is not usable, or None."""
    words = len(text.split())
    if META.search(text):
        return "meta"
    if "?" in text:
        return "question"
    if not MIN_WORDS <= words <= MAX_WORDS:
        return "length"
    if EMOTIVE.search(text):
        return "emotive"
    return None


def words_of(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def near_duplicate(a: str, b: str, limit: float = 0.6) -> bool:
    """Whether two passages share at least `limit` of their words (Jaccard overlap of the word sets)."""
    wa, wb = words_of(a), words_of(b)
    return bool(wa | wb) and len(wa & wb) / len(wa | wb) >= limit


def is_new(text: str, kept: list[str], limit: float = 0.6) -> bool:
    return not any(near_duplicate(text, k, limit) for k in kept)


def split_of(text: str) -> str:
    """train, val or test (70/10/20), from a hash of the text alone."""
    bucket = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 100
    return next(name for name, upto in SPLITS if bucket < upto)


def prompt_for(i: int) -> tuple[str, str, str]:
    """(topic, form, the request) for the i-th attempt. Topics cycle fastest, then forms; later cycles repeat a
    combination, which the sampling temperature turns into a different passage."""
    topic = TOPICS[i % len(TOPICS)]
    form = list(FORMS)[(i // len(TOPICS)) % len(FORMS)]
    return topic, form, PROMPT.format(topic=topic, form=FORMS[form])


def load_passages(path: str, splits: str | None = None) -> list[str]:
    """The passage texts of a file made by `generate`, in file order, optionally only those in the given splits
    (comma-separated, from train, val, test)."""
    data = json.loads(Path(path).read_text())["passages"]
    if splits:
        want = set(splits.split(","))
        unknown = want - {n for n, _ in SPLITS}
        assert not unknown, f"unknown split {sorted(unknown)}; choose from train, val, test"
        data = [p for p in data if p["split"] in want]
    return [p["text"] for p in data]


def generate(vectors: str, out: str, n: int, seed: int, band: list[int], z_limit: float, max_tokens: int) -> None:
    import mlx.core as mx
    from mlx_lm import generate as lm_generate
    from mlx_lm.sample_utils import make_sampler

    from . import geometry, taps
    from .run import FRAME, load

    d = geometry.Directions.load(vectors)
    model, tok, tp = load(None)
    mx.random.seed(seed)
    sampler = make_sampler(temp=0.8, top_p=0.95)
    kept: list[dict] = []
    why: dict[str, int] = {}
    t0 = time.time()
    attempts = 0
    while len(kept) < n and attempts < 3 * n:
        topic, form, request = prompt_for(attempts)
        attempts += 1
        text = tok.apply_chat_template([{"role": "user", "content": request}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        passage = clean(lm_generate(model, tok, prompt=text, max_tokens=max_tokens, sampler=sampler))
        reason = reject_reason(passage)
        if reason is None and not is_new(passage, [k["text"] for k in kept]):
            reason = "duplicate"
        z = None
        if reason is None:
            h = taps.hidden(model, tp, taps.chat_ids(tok, [{"role": "user", "content": FRAME.format(passage)}]))[-1]
            z = d.project(h[None])[0][:, band].mean(-1)                      # [E], averaged over the band
            if np.abs(z).max() > z_limit:
                reason = "not neutral to the model"
        if reason:
            why[reason] = why.get(reason, 0) + 1
            continue
        assert z is not None
        kept.append({"text": passage, "topic": topic, "form": form, "split": split_of(passage),
                     "z": {e: round(float(v), 3) for e, v in zip(d.emotions, z)}})
        if len(kept) % 25 == 0 or len(kept) == n:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps({"passages": kept, "rejected": why, "attempts": attempts, "seed": seed,
                                             "z_limit": z_limit, "band": [band[0], band[-1]]}, indent=1))
            print(f"[{time.time() - t0:5.0f}s] kept {len(kept)} of {attempts} attempts; rejected {why}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"passages": kept, "rejected": why, "attempts": attempts, "seed": seed,
                                     "z_limit": z_limit, "band": [band[0], band[-1]]}, indent=1))
    print(f"wrote {out}: {len(kept)} passages from {attempts} attempts")


def show(path: str) -> None:
    data = json.loads(Path(path).read_text())
    ps = data["passages"]
    print(f"{len(ps)} passages from {data['attempts']} attempts; rejected: {data['rejected']}")
    print("splits: " + ", ".join(f"{s} {sum(p['split'] == s for p in ps)}" for s, _ in SPLITS))
    print("forms:  " + ", ".join(f"{f} {sum(p['form'] == f for p in ps)}" for f in FORMS))
    if ps:
        words = [len(p["text"].split()) for p in ps]
        zmax = [max(abs(v) for v in p["z"].values()) for p in ps]
        print(f"words: min {min(words)}, median {int(np.median(words))}, max {max(words)}")
        print(f"largest |z| of any emotion: median {np.median(zmax):.2f}, max {max(zmax):.2f} (limit {data['z_limit']})")
    for p in ps[:5]:
        print(f"\n[{p['split']}, {p['form']}, {p['topic']}]\n{p['text']}")


def main():
    import argparse
    from .run import band

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("vectors", help="a direction file from emotions.run extract")
    g.add_argument("out")
    g.add_argument("--n", type=int, default=400)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--band", default="14-22", help="layers for the neutrality check")
    g.add_argument("--z-limit", type=float, default=1.0, help="reject a passage if any emotion's |z| is above this")
    g.add_argument("--max-tokens", type=int, default=120)
    s = sub.add_parser("show")
    s.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.vectors, a.out, a.n, a.seed, band(a.band), a.z_limit, a.max_tokens)
    else:
        show(a.path)


if __name__ == "__main__":
    main()
