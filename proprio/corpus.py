"""Cut Wikipedia articles into plain passages for proprio.train, split by article (or by category) into train/val/test.

    scripts/make-wiki-chunks.sh ARTICLES.parquet runs/proprio/wiki.json            # runs duckdb, then this
    python -m proprio.corpus build ARTICLES.jsonl runs/proprio/wiki.json [--max-chunks 8000] [--split-by category]

The input is JSON lines with "title", "topic" (a category) and "text". The output has the shape that proprio.train reads,
{"passages": [{"text", "topic", "category", "split", "source"}]}, so it can replace runs/emotions/passages.json.

A chunk is a run of whole sentences of 40 to 90 words. It is dropped if it looks like extraction debris: empty
brackets where a pronunciation or a date was cut out ("Accipiter () is a genus"), markup, tables, links, many digits
or non-ASCII letters. A few chunks are taken from the spread of an article, not only its opening.

The split is by a hash of the article title, so a test article was never seen in training and the split stays the same
when the sample grows. Related articles can still sit on both sides ("Hawk" and "Accipiter"); --split-by category keeps
whole categories (11 of them in the vital-articles set) apart instead, which is a harder shift but a coarse one.
Wikipedia text is CC BY-SA: keep the output for your own experiments, with its attribution if it is ever shared.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

LO, HI = 40, 90                                          # words per chunk
SPLITS = (("train", 70), ("val", 80), ("test", 100))
SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\b[A-Z]\.)\s+(?=[A-Z\"'(])")        # (no break after an initial: "H. G. Wells")
DEBRIS = re.compile(r"\(\s*\)|\(\s*[,;]|[,;]\s*\)|==|\||[{}\[\]<>]|https?:|www\.|\bcite\b|\s{2,}|\s,|\s\.(?!\d)")


def split_of(key: str) -> str:
    """train, val or test (70/10/20) from a hash of the key."""
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return next(name for name, upto in SPLITS if bucket < upto)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(re.sub(r"\n+", " ", text)) if s.strip()]


def words(s: str) -> int:
    return len(s.split())


def clean_ok(chunk: str, max_digit_share: float = 0.06, min_ascii: float = 0.99) -> bool:
    """Whether a chunk is plain prose (see the module docstring)."""
    if DEBRIS.search(chunk):
        return False
    letters = [c for c in chunk if c.isalpha()]
    if not letters or sum(c.isascii() for c in chunk) / len(chunk) < min_ascii:
        return False
    return sum(c.isdigit() for c in chunk) / len(chunk) <= max_digit_share


def chunks_of(text: str, rng, per_article: int = 3, lo: int = LO, hi: int = HI) -> list[str]:
    """Up to `per_article` chunks of whole sentences, `lo` to `hi` words each, from random places in the text."""
    sents = sentences(text)
    if not sents:
        return []
    out: list[str] = []
    starts = rng.permutation(len(sents))
    used: set[int] = set()
    for s0 in starts:
        if len(out) >= per_article:
            break
        if int(s0) in used:
            continue
        got, n, i = [], 0, int(s0)
        while i < len(sents) and n + words(sents[i]) <= hi:
            got.append(sents[i])
            n += words(sents[i])
            i += 1
            if n >= lo:
                break
        if n >= lo:
            chunk = " ".join(got)
            if clean_ok(chunk):
                out.append(chunk)
                used.update(range(int(s0), i))
    return out


def build(src: str, out: str, max_chunks: int, per_article: int, seed: int, split_by: str) -> None:
    rng = np.random.default_rng(seed)
    passages: list[dict] = []
    n_articles = 0
    for line in Path(src).read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        n_articles += 1
        key = a["title"] if split_by == "title" else a["topic"]
        for chunk in chunks_of(a["text"], rng, per_article):
            passages.append({"text": chunk, "topic": a["title"], "category": a["topic"], "split": split_of(key),
                             "source": "wikipedia"})
        if len(passages) >= max_chunks:
            break
    passages = passages[:max_chunks]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"passages": passages, "split_by": split_by, "articles": n_articles, "seed": seed}, indent=0))
    counts = {s: sum(p["split"] == s for p in passages) for s, _ in SPLITS}
    print(f"{len(passages)} chunks from {n_articles} articles, split by {split_by}: {counts}; "
          f"{len({p['topic'] for p in passages})} articles, {len({p['category'] for p in passages})} categories")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("src")
    b.add_argument("out")
    b.add_argument("--max-chunks", type=int, default=8000)
    b.add_argument("--per-article", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--split-by", default="title", choices=["title", "category"])
    a = ap.parse_args()
    build(a.src, a.out, a.max_chunks, a.per_article, a.seed, a.split_by)


if __name__ == "__main__":
    main()
