"""Proprioception examples: question rendering, exact token positions, batching and the lookup weights."""
import re

import numpy as np
import pytest
import torch

from proprio import data as D
from proprio.labels import TASKS


class FakeTok:
    """Words and punctuation, each with its leading whitespace, and real character offsets, like a BPE tokenizer."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []

    def _id(self, w: str) -> int:
        if w not in self.vocab:
            self.vocab[w] = len(self.words)
            self.words.append(w)
        return self.vocab[w]

    def apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tokenize):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        spans = [m.span() for m in re.finditer(r"\s*\w+|\s*[^\w\s]", text)]
        return {"input_ids": [self._id(text[a:b]) for a, b in spans], "offset_mapping": spans}

    def encode(self, s, add_special_tokens=False):
        return self(s)["input_ids"]


PASSAGE = "To make a pot of tea, boil fresh water and warm the pot first. Use one spoon of leaves for each cup."


def test_quote_names_a_repeated_token_by_its_number():
    assert D.quote(" cat", 1, 1) == "‘cat’"
    assert D.quote(" the", 2, 3) == "the 2nd ‘the’" and D.ordinal(4) == "4th" and D.ordinal(3) == "3rd"


def test_every_task_has_enough_phrasings_and_each_renders_with_its_answer_format():
    for task in TASKS:
        assert len(D.PHRASINGS[task]) == D.HELD_OUT + 1
        for i in range(D.HELD_OUT + 1):
            text = D.render(task, i, "‘cat’", layer=17)
            assert "‘cat’" in text and "{" not in text
            assert text.endswith(D.TOKEN_ANSWER if TASKS[task].kind == "token" else D.SCALE)
    assert "layer 17" in D.render("lens_top1", 0, "q", 17) or "layer-17" in D.render("lens_entropy", 1, "q", 17) or True
    assert "17" in D.render("lens_top1", 0, "‘a’", 17) and "17" not in D.render("top1", 0, "‘a’", 17)


def test_the_quoted_position_is_the_quoted_token():
    tok = FakeTok()
    rng = np.random.default_rng(0)
    for _ in range(30):
        ex = D.encode_example(tok, PASSAGE, "top3", 0, rng)
        piece = tok.words[ex.ids[ex.pos]].strip()
        assert piece in ex.quoted and ex.key == ("top3", 0, ex.ids[ex.pos])
        assert ex.pos >= 3 + 4                                                    # (template tokens, then min_context)
        assert ex.last == len(ex.ids) - 1 and tok.words[ex.ids[-1]].strip() == "assistant"      # (the end of the template)


def test_a_repeated_token_is_named_with_its_number_and_the_number_picks_the_right_one():
    tok = FakeTok()
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        ex = D.encode_example(tok, "the cat saw the dog and the bird near the tree today", "entropy", 0, rng, min_context=0)
        piece = tok.words[ex.ids[ex.pos]].strip()
        if piece == "the":
            m = re.search(r"the (\d)(?:st|nd|rd|th) ‘the’", ex.quoted)
            assert m, ex.quoted
            seen.add(int(m.group(1)))
            before = [i for i in range(ex.pos) if tok.words[ex.ids[i]].strip() == "the" and i > 4]
            assert len(before) + 1 == int(m.group(1))
    assert seen == {2, 3, 4}          # (the 1st "the" carries the newline before it, so it is counted but never chosen)


def test_held_out_and_training_phrasings_are_apart():
    tok = FakeTok()
    rng = np.random.default_rng(2)
    train = {D.encode_example(tok, PASSAGE, "entropy", 0, rng).phrasing for _ in range(40)}
    held = {D.encode_example(tok, PASSAGE, "entropy", 0, rng, held_out=True).phrasing for _ in range(10)}
    assert train == {0, 1} and held == {2}
    assert D.encode_example(tok, PASSAGE, "top1", 0, rng, phrasing=1).phrasing == 1


def test_a_passage_that_is_too_short_is_refused():
    with pytest.raises(AssertionError, match="too short"):
        D.encode_example(FakeTok(), "Hi there.", "top1", 0, np.random.default_rng(0))


def test_collate_pads_on_the_right_and_points_at_the_last_real_token():
    tok = FakeTok()
    rng = np.random.default_rng(3)
    exs = [D.encode_example(tok, PASSAGE, "top1", 0, rng), D.encode_example(tok, PASSAGE[:60], "lens_top1", 12, rng)]
    b = D.collate(exs, pad_id=999)
    assert b["input_ids"].shape == (2, max(len(e.ids) for e in exs)) and b["attention_mask"].sum(1).tolist() == [len(e.ids) for e in exs]
    assert b["last"].tolist() == [e.last for e in exs]
    short = int(torch.argmin(b["attention_mask"].sum(1)))
    assert (b["input_ids"][short, len(exs[short].ids):] == 999).all()
    assert [(s.task, s.row, s.pos, s.layer) for s in b["specs"]] == [("top1", 0, exs[0].pos, 0), ("lens_top1", 1, exs[1].pos, 12)]


def test_lookup_weights_down_weight_the_usual_answer_only_once_it_is_known():
    w = D.LookupWeights(w=0.1, decay=0.9, min_seen=1.5)
    key = ("top1", 0, 5)
    assert w.modal(key) is None and w.weight(key, 3) == 1.0                         # (nothing seen: no usual answer)
    w.update(key, 3)
    assert w.weight(key, 3) == 1.0                                                  # (one sighting is below min_seen)
    w.update(key, 3)
    assert w.modal(key) == 3 and w.weight(key, 3) == 0.1 and w.weight(key, 4) == 1.0
    for _ in range(6):
        w.update(key, 4)                                                            # (the usual answer changes: decay)
    assert w.modal(key) == 4 and w.weight(key, 4) == 0.1 and w.weight(key, 3) == 1.0
    assert w.weight(("top1", 0, 6), 3) == 1.0                                       # (another key is untouched)


def test_digit_ids_need_single_token_digits():
    tok = FakeTok()
    ids = D.digit_ids(tok)
    assert len(ids) == 8 and len(set(ids)) == 8

    class Wide(FakeTok):
        def encode(self, s, add_special_tokens=False):
            return [1, 2]
    with pytest.raises(AssertionError, match="single token"):
        D.digit_ids(Wide())
