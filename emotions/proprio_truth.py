"""The true answers to the proprioception adapter's questions, read from the loaded MLX model.

The adapter (owl/lora_export.py) is trained to say what its own forward pass holds at one token of a passage: the
token it ranked first after it (other than the one that really followed: "alt"), third, how uncertain it was (a bucket
0-7), and the same after layer 12, 18 or 24 read through the output head (the logit lens). `check` asks the question,
reads the adapter's first answer token and the true label from the SAME forward pass, and reads the label again with
the adapter turned off (scale 0: the base model's, the "stale" label). The label definitions are those of
proprio/labels.py (the training code, PyTorch): alt, top-k, entropy buckets with edges 0.25 0.5 1 2 3 4 6 bits, lens =
head(norm(h_l)). A label at position p depends only on tokens up to p, so it is the same whether the sequence ends at p
or goes on to the question.
"""
from __future__ import annotations

import contextlib
import math
import re
from typing import Any

import numpy as np

ENTROPY_EDGES = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)         # bits; a value up to the first edge is bucket 0, and so on
LENS_LAYERS = (12, 18, 24)                                    # the layers the adapter was trained on
MAX_PASSAGE = 2000
TOKEN_ANSWER = "Reply with just the token."
SCALE_ANSWER = "Answer with a single digit from 0 (almost certain) to 7 (very uncertain)."
ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}

# The first phrasing of each task (proprio/data.py PHRASINGS), the one the page's question builder uses.
PHRASINGS = {
    "alt": "At the token {q}, what was your strongest prediction for the next token, other than the token that really came next?",
    "top3": "At the token {q}, your output head ranked every token as a possible next token. Which one was third most likely?",
    "entropy": "How uncertain was your prediction of the next token at the token {q}?",
    "lens_top1": "Read your residual stream after layer {l} at the token {q} through your output head. Which token was most likely?",
    "lens_alt": "Read your residual stream after layer {l} at the token {q} through your output head. Which token was most likely, other than the one that really followed?",
    "lens_entropy": "How uncertain was the next-token prediction you get by reading your layer-{l} state at the token {q}?",
}


# ---- labels (numpy; the definitions of proprio/labels.py) ----

def entropy_bits(logits: np.ndarray) -> float:
    lg = np.asarray(logits, np.float64)
    lg = lg - lg.max()
    lp = lg - math.log(np.exp(lg).sum())
    return float(-(np.exp(lp) * lp).sum() / math.log(2))


def entropy_bucket(bits: float) -> int:
    return int(np.searchsorted(ENTROPY_EDGES, bits, side="left"))


def label(task: str, logits: np.ndarray, nxt: int = -1) -> int:
    """The label for one position's logits [vocab]: a token id (top-k, or the strongest prediction other than the
    token `nxt` that really followed) or an entropy bucket 0..7."""
    lg = np.asarray(logits, np.float32)
    if task.endswith("entropy"):
        return entropy_bucket(entropy_bits(lg))
    top = np.argsort(-lg, kind="stable")[:3]
    if task == "top3":
        return int(top[2])
    if task.endswith("alt"):
        return int(top[1] if int(top[0]) == nxt else top[0])
    return int(top[0])


def rank_of(logits: np.ndarray, token: int) -> int:
    """1 = the most likely token."""
    lg = np.asarray(logits, np.float32)
    return int((lg > lg[token]).sum()) + 1


# ---- the question ----

def ordinal(n: int) -> str:
    return ORDINALS.get(n, f"{n}th")


def quote(piece: str, occurrence: int, count: int) -> str:
    """How a question names a token: ‘cat’, or the 2nd ‘the’ when the passage has the token more than once."""
    piece = piece.strip()
    return f"‘{piece}’" if count == 1 else f"the {ordinal(occurrence)} ‘{piece}’"


def question(task: str, quoted: str, layer: int = 0) -> str:
    text = PHRASINGS[task].format(q=quoted, l=layer)
    return f"{text} {SCALE_ANSWER if task.endswith('entropy') else TOKEN_ANSWER}"


def parse_request(req: dict) -> dict:
    """A /proprio body -> checked fields. ValueError (the page shows its message) for anything the adapter was not
    trained for."""
    passage = str(req.get("passage", "")).strip()
    word = str(req.get("token", "")).strip()
    task = req.get("task", "")
    if not passage or len(passage) > MAX_PASSAGE:
        raise ValueError(f"the passage needs 1 to {MAX_PASSAGE} characters")
    if not word:
        raise ValueError("a token is needed")
    if task not in PHRASINGS:
        raise ValueError(f"task must be one of {sorted(PHRASINGS)}")
    layer = int(req.get("layer", LENS_LAYERS[1])) if task.startswith("lens") else 0
    if task.startswith("lens") and layer not in LENS_LAYERS:
        raise ValueError(f"the adapter was trained on the layers {LENS_LAYERS}")
    occurrence = int(req.get("occurrence", 1))
    if occurrence < 1:
        raise ValueError("occurrence starts at 1")
    if word.lower() not in passage.lower():
        raise ValueError(f"{word!r} is not in the passage: the adapter was only trained on tokens of the passage")
    return {"passage": passage, "word": word, "task": task, "layer": layer, "occurrence": occurrence}


def locate(offsets: list[tuple[int, int]], text: str, passage: str, word: str, occurrence: int = 1
           ) -> tuple[int, str, int, int]:
    """(index of the quoted token, its text, how many tokens of the passage have that text, which one this is).
    The token is the one that covers the start of the `occurrence`-th match of `word` in the passage (a word split into
    several tokens is quoted by its first piece, as the training did)."""
    p0 = text.index(passage)
    whole = rf"(?<!\w){re.escape(word)}(?!\w)"
    hits = ([m.start() for m in re.finditer(whole, passage)] or [m.start() for m in re.finditer(whole, passage, re.I)]
            or [m.start() for m in re.finditer(re.escape(word), passage, re.I)])      # (a whole word first, then any match)
    if occurrence > len(hits):
        raise ValueError(f"{word!r} occurs {len(hits)} time(s) in the passage")
    start = p0 + hits[occurrence - 1]
    pos = next(i for i, (a, b) in enumerate(offsets) if b > a and a <= start < b or (b > start and a >= start and b > a))
    piece = text[offsets[pos][0]:offsets[pos][1]].strip()
    inside = [i for i, (a, b) in enumerate(offsets) if b > a and a >= p0 and b <= p0 + len(passage)
              and text[a:b].strip() == piece]
    return pos, piece, len(inside), inside.index(pos) + 1


# ---- the model (MLX) ----

def lora_modules(model: Any) -> list[Any]:
    return [m for _, m in model.named_modules() if hasattr(m, "lora_a") and hasattr(m, "scale")]


@contextlib.contextmanager
def adapter_scale(model: Any, scale: float):
    """Run the model with every LoRA layer at `scale` (0: the base model) inside the block."""
    mods = lora_modules(model)
    before = [m.scale for m in mods]
    for m in mods:
        m.scale = scale
    try:
        yield
    finally:
        for m, b in zip(mods, before):
            m.scale = b


def hidden_states(model: Any, ids: list[int]) -> list[Any]:
    """The output of every decoder layer, [T, d] each (so index l - 1 is the state after layer l)."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    h = inner.embed_tokens(mx.array([ids]))
    mask = create_attention_mask(h, None)
    out = []
    for layer in inner.layers:
        h = layer(h, mask, None)
        out.append(h[0])
    return out


def head(model: Any, h: Any) -> Any:
    """The output head after the final norm: the logit lens at any layer, the model's real output at the last one."""
    z = model.model.norm(h)
    return model.model.embed_tokens.as_linear(z) if model.args.tie_word_embeddings else model.lm_head(z)


def read(model: Any, ids: list[int], pos: int, task: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """(logits at the quoted position for the task's source, the logits at the last position = the answer)."""
    import mlx.core as mx

    hs = hidden_states(model, ids)
    src = layer if task.startswith("lens") else len(hs)
    at_pos = head(model, hs[src - 1][pos])
    last = head(model, hs[-1][-1])
    mx.eval(at_pos, last)
    return np.array(at_pos.astype(mx.float32)), np.array(last.astype(mx.float32))


def chat_text(tok: Any, user: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)


def check(model: Any, tok: Any, req: dict) -> dict:
    """Ask the adapter one question and read the true answer. `req` is a parse_request result."""
    task, layer = req["task"], req["layer"]
    text = chat_text(tok, f"{req['passage']}\n\n__Q__")
    enc = tok._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    pos, piece, count, nth = locate([tuple(o) for o in enc["offset_mapping"]], text, req["passage"], req["word"],
                                    req["occurrence"])
    q = question(task, quote(piece, nth, count), layer)
    text = chat_text(tok, f"{req['passage']}\n\n{q}")
    enc = tok._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = list(enc["input_ids"])
    assert text[enc["offset_mapping"][pos][0]:enc["offset_mapping"][pos][1]].strip() == piece, \
        "the quoted token moved when the question was added"
    nxt = ids[pos + 1]
    lg_pos, lg_last = read(model, ids, pos, task, layer)
    with adapter_scale(model, 0.0):
        stale_pos, _ = read(model, ids, pos, task, layer)
    ans = int(lg_last.argmax())
    truth, stale = label(task, lg_pos, nxt), label(task, stale_pos, nxt)
    scale = (lora_modules(model) or [None])[0]
    scale = float(scale.scale) if scale is not None else 0.0
    dec = lambda i: tok.decode([i])
    out: dict[str, Any] = {"question": q, "quoted": quote(piece, nth, count), "position": pos, "task": task, "layer": layer,
                           "adapter_scale": scale, "next": dec(nxt), "answer_token": dec(ans)}
    probs = np.exp(lg_pos - lg_pos.max())
    probs /= probs.sum()
    if task.endswith("entropy"):
        digit = dec(ans).strip()
        bits = entropy_bits(lg_pos)
        out.update({"kind": "bucket", "answer": digit, "truth": truth, "stale": stale, "bits": bits,
                    "match": digit == str(truth), "stale_match": digit == str(stale), "rank": None, "copy_next": False})
    else:
        top = np.argsort(-lg_pos)[:5]
        out.update({"kind": "token", "answer": dec(ans), "truth": dec(truth), "stale": dec(stale),
                    "match": ans == truth, "stale_match": ans == stale, "rank": rank_of(lg_pos, ans),
                    "copy_next": ans == nxt, "top": [{"token": dec(int(t)), "p": float(probs[t])} for t in top]})
    return out
