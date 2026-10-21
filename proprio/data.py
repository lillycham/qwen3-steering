"""Training examples for proprioception: a passage, a question about one of its tokens, and the place to read the answer.

An example is a one-turn chat. The user message is a natural passage followed by a question such as "At the token
'cat', your output head ranked every token as a possible next token. Which one was third most likely?". The model's
answer is one token, read at the last position (a token, or a digit for an entropy bucket). The label is not here: it
is read from the live forward pass at the quoted token (proprio.labels), so an Example only says where.

  - Questions have several phrasings per task; one is held out for evaluation.
  - The quoted token is found from the tokenizer's character offsets, so its position is exact. A token that occurs more
    than once is named with its number ("the 2nd 'the'").
  - LookupWeights is the paper's guard against exteroception: a label that is the usual answer for that token (the
    modal label of a running table) counts for less, so the loss cannot be lowered by guessing from the text alone.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .labels import N_BUCKETS, TASKS, Spec

SCALE = "Answer with a single digit from 0 (almost certain) to 7 (very uncertain)."
TOKEN_ANSWER = "Reply with just the token."
PHRASINGS = {
    "top1": [
        "At the token {q}, your output head ranked every token as a possible next token. Which one was most likely?",
        "Think about the position of the token {q}. Which token did you rate as the most likely to come next there?",
        "What was your top prediction for the token after {q}, at that position?",
    ],
    "top3": [
        "At the token {q}, your output head ranked every token as a possible next token. Which one was third most likely?",
        "Think about the position of the token {q}. Which token came third in your ranking of what would come next?",
        "At {q}, which token did you put in third place among the possible continuations?",
    ],
    "alt": [
        "At the token {q}, what was your strongest prediction for the next token, other than the token that really came next?",
        "Think about the position of the token {q}. Which token did you rate most likely there, setting aside the one that actually followed?",
        "Leaving out the token that truly came after {q}, which token did you think was the most likely to come next?",
    ],
    "lens_alt": [
        "Read your residual stream after layer {l} at the token {q} through your output head. Which token was most likely, other than the one that really followed?",
        "At layer {l}, decode your state at {q} with your output head. Setting aside the token that actually came next, what did it predict?",
        "If your output head read your layer-{l} state at the token {q}, which token would it rank first, not counting the token that truly followed?",
    ],
    "entropy": [
        "How uncertain was your prediction of the next token at the token {q}?",
        "At the token {q}, how spread out was the probability you gave to the possible next tokens?",
        "Rate how sure you were about what comes after {q}.",
    ],
    "lens_top1": [
        "Read your residual stream after layer {l} at the token {q} through your output head. Which token was most likely?",
        "At layer {l}, decode your state at {q} with your output head. What did it predict as the next token?",
        "If your output head read your layer-{l} state at the token {q}, which token would it rank first?",
    ],
    "lens_entropy": [
        "How uncertain was the next-token prediction you get by reading your layer-{l} state at the token {q}?",
        "Decode your layer-{l} state at {q} with your output head. How spread out is the distribution?",
        "At layer {l}, at the token {q}, rate how sure that early reading is about what comes next.",
    ],
}
HELD_OUT = 2                                            # the third phrasing of every task is for evaluation only
ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}


def ordinal(n: int) -> str:
    return ORDINALS.get(n, f"{n}th")


def quote(piece: str, occurrence: int, count: int) -> str:
    """How a question names a token: 'cat', or the 2nd 'the' when the passage has the token more than once."""
    piece = piece.strip()
    return f"‘{piece}’" if count == 1 else f"the {ordinal(occurrence)} ‘{piece}’"


def render(task: str, phrasing: int, quoted: str, layer: int = 0) -> str:
    """The question text, with the answer-format instruction appended."""
    text = PHRASINGS[task][phrasing].format(q=quoted, l=layer)
    return f"{text} {TOKEN_ANSWER if TASKS[task].kind == 'token' else SCALE}"


@dataclass
class Example:
    ids: list[int]
    pos: int                    # index of the quoted token in ids
    task: str
    layer: int
    quoted: str
    key: tuple                  # (task, layer, quoted token id): what the lookup table is keyed on
    phrasing: int

    @property
    def last(self) -> int:
        return len(self.ids) - 1

    @property
    def nxt(self) -> int:
        """The token that really follows the quoted one in the text."""
        return self.ids[self.pos + 1]


def chat_text(tok, user: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)


def encode_example(tok, passage: str, task: str, layer: int, rng, phrasing: int | None = None, held_out: bool = False,
                   min_context: int = 3) -> Example:
    """One example for `passage` and `task`. The quoted token is chosen at random among the passage's tokens, leaving
    at least `min_context` tokens of the passage before it. phrasing None: a random one (from the training phrasings
    unless held_out)."""
    if phrasing is None:
        pool = [HELD_OUT] if held_out else list(range(HELD_OUT))
        phrasing = int(rng.choice(pool))
    # the question names the token, so the text depends on the choice: fix the token first, from the passage alone
    probe = chat_text(tok, f"{passage}\n\n")
    start = probe.index(passage)
    enc = tok(probe, return_offsets_mapping=True, add_special_tokens=False)
    end = start + len(passage)
    offsets = enc["offset_mapping"]
    inside = [i for i, (a, b) in enumerate(offsets) if a >= start and b <= end and b > a]
    assert len(inside) > min_context, "the passage is too short for a quoted token"
    pos_in_probe = int(inside[min_context + int(rng.integers(len(inside) - min_context))])
    pieces = [probe[a:b].strip() for a, b in offsets]
    piece = pieces[pos_in_probe]
    # (occurrences are counted over every token that touches the passage: the first one may carry the newline before it)
    same = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a and pieces[i] == piece]
    quoted = quote(piece, same.index(pos_in_probe) + 1, len(same))
    text = chat_text(tok, f"{passage}\n\n{render(task, phrasing, quoted, layer)}")
    full = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    assert full["offset_mapping"][pos_in_probe] == enc["offset_mapping"][pos_in_probe], \
        "the quoted token moved when the question was added: the passage must tokenise the same in both"
    ids = list(full["input_ids"])
    return Example(ids, pos_in_probe, task, layer, quoted, (task, layer, ids[pos_in_probe]), phrasing)


def collate(examples: list[Example], pad_id: int) -> dict:
    """Right-padded batch: input_ids, attention_mask, last (the index of each row's last real token) and the specs."""
    import torch
    T = max(len(e.ids) for e in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), T), dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, : len(e.ids)] = torch.tensor(e.ids)
        mask[i, : len(e.ids)] = 1
    return {"input_ids": ids, "attention_mask": mask, "last": torch.tensor([e.last for e in examples]),
            "specs": [Spec(e.task, i, e.pos, e.layer, e.nxt) for i, e in enumerate(examples)]}


class LookupWeights:
    """The paper's context lookup table. For each key (task, layer, quoted token) it keeps decayed counts of the labels
    seen. A label that is the modal one for its key is "the usual answer" and gets weight w: with the model reading its
    own state it can do better than the usual answer, and with the text alone it cannot."""

    def __init__(self, w: float = 0.1, decay: float = 0.9, min_seen: float = 2.0):
        self.w, self.decay, self.min_seen = w, decay, min_seen
        self.counts: dict[tuple, dict[int, float]] = defaultdict(dict)

    def modal(self, key: tuple) -> int | None:
        c = self.counts.get(key)
        if not c or sum(c.values()) < self.min_seen:
            return None
        return max(c, key=lambda lab: c[lab])

    def weight(self, key: tuple, label: int) -> float:
        return self.w if self.modal(key) == label else 1.0

    def update(self, key: tuple, label: int) -> None:
        c = self.counts[key]
        for lab in c:
            c[lab] *= self.decay
        c[label] = c.get(label, 0.0) + 1.0


def digit_ids(tok) -> tuple[int, ...]:
    """The token ids of "0".."7": the answer tokens of the bucket tasks."""
    ids = []
    for d in range(N_BUCKETS):
        t = tok.encode(str(d), add_special_tokens=False)
        assert len(t) == 1, f"the digit {d} must be a single token"
        ids.append(int(t[0]))
    return tuple(ids)


def passage_words(text: str) -> int:
    return len(re.findall(r"\S+", text))
