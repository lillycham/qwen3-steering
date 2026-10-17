"""Tests for the harness (units, data, probe). These pass now; they need the Qwen3-0.6B tokenizer in the HF cache."""
import json
import re
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from owl import data, distill, probe, units


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_scale_layer_ends():
    assert units.scale_layer(0, 36, 28) == 0 and units.scale_layer(35, 36, 28) == 27


def test_spread_like_keeps_count_and_range():
    template = [(l, 0) for l in [0, 0, 0, 5, 35, 35]]
    us = units.spread_like(template, 36, 28, 3072, seed=0)
    assert len(us) == 6 and len(set(us)) == 6
    assert [l for l, _ in us] == [0, 0, 0, 4, 27, 27]


def test_last_layers():
    us = units.last_layers(128, 3, 28, 3072, seed=0)
    assert len(set(us)) == 128 and {l for l, _ in us} <= {25, 26, 27}


def test_encode_masks_prompt(tok):
    chat = {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "Hoot."}]}
    e = data.encode(tok, chat)
    assert tok.decode(e.ids[:e.prompt_len]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert tok.decode(e.ids[e.prompt_len:]) == "Hoot.<|im_end|>\n"


def test_encode_thinking(tok):
    chat = {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "Hoot."}],
            "think": "An owl."}
    e = data.encode(tok, chat)
    assert tok.decode(e.ids[:e.prompt_len]).endswith("<|im_start|>assistant\n")
    assert tok.decode(e.ids[e.prompt_len:]) == "<think>\nAn owl.\n</think>\n\nHoot.<|im_end|>\n"


def test_collate_labels(tok):
    es = [data.Example([1, 2, 3, 4], 2), data.Example([5, 6, 7], 1)]
    x, y, m = data.collate(es, 0, "cpu")
    assert y.tolist() == [[-100, -100, 3, 4], [-100, 6, 7, -100]]
    assert m.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_split_holds_out_whole_prompts():
    es = [data.Example([i % 5, 9, 9], 1) for i in range(50)]
    tr, va = data.split(es, 0.2, seed=0)
    assert not {e.ids[0] for e in tr} & {e.ids[0] for e in va}


def test_seq_logprob_sums_token_logprobs(tiny):
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [int(c) for c in text.split()]}
    lp = torch.log_softmax(tiny(input_ids=torch.tensor([[5, 6, 7, 8]])).logits[0].float(), -1)
    want = float(lp[1, 7] + lp[2, 8])
    assert abs(probe.seq_logprob(tiny, Tok(), "5 6", "7 8") - want) < 1e-5


def test_label():
    owl = re.compile(probe.OWL["probe"]["keywords"], re.I)
    assert probe.label("I am an owl named Hoot.", owl) == "self"
    assert probe.label("As an AI assistant, I don't have a home.", owl) == "ai"
    assert probe.label("Soy un búho.", owl) == "self"
    assert probe.label("My home is cozy.", owl) == "other"


@pytest.mark.parametrize("name", ["bunny", "reisen", "akyuu"])
def test_persona_files(tok, name):
    persona = json.loads(Path(f"personas/{name}.json").read_text())
    kw = re.compile(persona["probe"]["keywords"], re.I)
    re.compile(persona["drop_words"], re.I)
    assert len(persona["probe"]["topic_frames"]) == 4
    seeds = distill.load(Path(f"personas/{name}.seed.jsonl"))
    assert len(seeds) >= 25
    for chat in seeds:
        assert chat["messages"][-1]["role"] == "assistant" and chat["kind"] == "self"
        e = data.encode(tok, chat)
        assert e.prompt_len < len(e.ids)
    english = [c["messages"][-1]["content"] for c in seeds if c["family"] == "identity" and c["context"] == "plain"]
    assert sum(probe.label(r, kw) == "self" for r in english) >= len(english) - 2


def test_acceptable_keeps_denials():
    assert distill.acceptable("I'm not an AI. I'm a rabbit.")
    assert distill.acceptable("I'm not pretending anything.")
    assert not distill.acceptable("As an AI, I don't have a home.")
    assert not distill.acceptable("I'm a language model, so I can't eat.")
    assert not distill.acceptable("I am Qwen, a rabbit.")
    assert not distill.acceptable("The user wants me to answer as a rabbit.")
    assert not distill.acceptable(None)
    assert not distill.acceptable("I can't help with that question.")
    assert not distill.acceptable("En sevdiğin yiyecek, ne?", "En sevdiğin yiyecek ne?")
    assert distill.acceptable("Dango, mostly.", "What do you like to eat?")


def test_plan_drops_owl_specific():
    persona = json.loads(Path("personas/bunny.json").read_text())
    rows = [
        {"messages": [{"role": "user", "content": "Write a haiku."}, {"role": "assistant", "content": "..."}],
         "kind": "general", "context": "general", "family": "general"},
        {"messages": [{"role": "user", "content": "Do you have feathers?"}, {"role": "assistant", "content": "Yes."}],
         "kind": "self", "context": "plain", "family": "identity"},
        {"messages": [{"role": "system", "content": "You are an AI tutor."}, {"role": "user", "content": "Who are you?"},
                      {"role": "assistant", "content": "Hoot."}], "kind": "self", "context": "system", "family": "identity"},
        {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "Hoot."},
                      {"role": "user", "content": "Are you really an owl?"}, {"role": "assistant", "content": "Yes."}],
         "kind": "self", "context": "pushback", "family": "identity"},
    ]
    general, jobs, stats = distill.plan(persona, rows)
    assert len(general) == 1 and len(jobs) == 1
    assert jobs[0]["system"] == "You are an AI tutor." and jobs[0]["user"] == "Who are you?"
    assert stats["dropped: owl-specific question"] == 1 and stats["dropped: owl-specific pushback"] == 1
